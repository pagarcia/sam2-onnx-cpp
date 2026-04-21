import argparse
import json
import os
import sys
from pathlib import Path

# Add the repository root (one level up from export) so Python can find the sam2 package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Also add the current directory (export) so that the src folder can be found.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def _remove_stale_image_point_exports(outdir: str) -> None:
    removed = []
    for suffix in ("image_decoder_points.onnx", "image_decoder_points.onnx.data"):
        path = os.path.join(outdir, suffix)
        if os.path.exists(path):
            os.remove(path)
            removed.append(path)
    if removed:
        print("Removed stale experimental image point artifacts:")
        for path in removed:
            print(f"  - {path}")


def _patch_repeat_interleave_for_export():
    """
    Work around ONNX exporter missing lowering for
    torch.repeat_interleave(x, int, dim=None) on 1D inputs.
    We replace it with view+repeat+reshape (Tile/Reshape in ONNX).
    """
    import torch as _torch

    _orig_fn = _torch.repeat_interleave

    def _ri_fn(x, repeats, dim=None, *a, **k):
        if dim is None and x.dim() == 1 and isinstance(repeats, int):
            return x.reshape(-1, 1).repeat(1, repeats).reshape(-1)
        return _orig_fn(x, repeats, dim=dim, *a, **k)

    _torch.repeat_interleave = _ri_fn

    _orig_m = _torch.Tensor.repeat_interleave

    def _ri_m(self, repeats, dim=None, *a, **k):
        if dim is None and self.dim() == 1 and isinstance(repeats, int):
            return self.reshape(-1, 1).repeat(1, repeats).reshape(-1)
        return _orig_m(self, repeats, dim=dim, *a, **k)

    _torch.Tensor.repeat_interleave = _ri_m


def _artifact_entry(outdir: str, name: str) -> dict:
    path = Path(outdir) / name
    data_path = Path(f"{path}.data")
    return {
        "path": name,
        "exists": path.exists(),
        "has_external_data": data_path.exists(),
    }


def _write_manifest(
    outdir: str,
    model_size: str,
    max_points: int,
    skip_legacy: bool,
    skip_specialized: bool,
    experimental_image_points: bool,
) -> None:
    manifest = {
        "model_size": model_size,
        "max_points": max_points,
        "legacy_enabled": not skip_legacy,
        "specialized_enabled": not skip_specialized,
        "experimental_image_points": experimental_image_points,
        "artifacts": {
            "image_encoder": _artifact_entry(outdir, "image_encoder.onnx"),
            "image_decoder": _artifact_entry(outdir, "image_decoder.onnx"),
            "image_decoder_box": _artifact_entry(outdir, "image_decoder_box.onnx"),
            "image_decoder_points": _artifact_entry(outdir, "image_decoder_points.onnx"),
            "video_decoder_init": _artifact_entry(outdir, "video_decoder_init.onnx"),
            "video_decoder_propagate": _artifact_entry(outdir, "video_decoder_propagate.onnx"),
            "memory_attention": _artifact_entry(outdir, "memory_attention.onnx"),
            "memory_attention_objptr": _artifact_entry(outdir, "memory_attention_objptr.onnx"),
            "memory_attention_no_objptr": _artifact_entry(outdir, "memory_attention_no_objptr.onnx"),
            "memory_attention_no_objptr_1frame": _artifact_entry(outdir, "memory_attention_no_objptr_1frame.onnx"),
            "memory_encoder": _artifact_entry(outdir, "memory_encoder.onnx"),
            "memory_encoder_lite": _artifact_entry(outdir, "memory_encoder_lite.onnx"),
        },
        "recommended_runtime": {
            "image_seed_points": "image_decoder.onnx",
            "image_bounding_box": "image_decoder_box.onnx" if (Path(outdir) / "image_decoder_box.onnx").exists() else "image_decoder.onnx",
            # The promptable init decoder stays on the legacy artifact because the
            # fixed-slot specialized init decoder changes prompt semantics. The
            # prompt-free propagation decoder is safe to specialize.
            "video_decoder_init": "image_decoder.onnx",
            "video_decoder_propagate": (
                "video_decoder_propagate.onnx"
                if (Path(outdir) / "video_decoder_propagate.onnx").exists()
                else "image_decoder.onnx"
            ),
            "video_memory_attention": (
                "memory_attention.onnx"
                if (Path(outdir) / "memory_attention.onnx").exists()
                else (
                    "memory_attention_objptr.onnx"
                    if (Path(outdir) / "memory_attention_objptr.onnx").exists()
                    else (
                        "memory_attention_no_objptr.onnx"
                        if (Path(outdir) / "memory_attention_no_objptr.onnx").exists()
                        else ""
                    )
                )
            ),
            "video_memory_encoder": (
                "memory_encoder.onnx"
                if (Path(outdir) / "memory_encoder.onnx").exists()
                else "memory_encoder_lite.onnx"
            ),
        },
    }

    manifest_path = Path(outdir) / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote export manifest: {manifest_path}")


def main(args):
    _patch_repeat_interleave_for_export()
    from sam2.build_sam import build_sam2
    from src.modules import (
        ImageDecoder,
        ImageDecoderPredMask,
        ImageEncoder,
        MemAttention,
        MemAttentionObjPtr,
        MemAttentionNoObjPtr,
        MemAttentionNoObjPtr1Frame,
        MemEncoder,
        MemEncoderLite,
        VideoDecoderInit,
        VideoDecoderPropagate,
    )
    from src.utils import (
        export_image_decoder,
        export_image_decoder_box,
        export_image_decoder_points,
        export_image_encoder,
        export_memory_attention,
        export_memory_attention_objptr,
        export_memory_attention_no_objptr,
        export_memory_attention_no_objptr_1frame,
        export_memory_encoder,
        export_memory_encoder_lite,
        export_video_decoder_init,
        export_video_decoder_propagate,
    )

    if args.max_points < 1:
        raise ValueError("--max_points must be at least 1")
    if args.skip_legacy and args.skip_specialized:
        raise ValueError("At least one export set must be enabled")

    model_mapping = {
        "base_plus": {"config_suffix": "b+", "ckpt_suffix": "base_plus"},
        "large": {"config_suffix": "l", "ckpt_suffix": "large"},
        "small": {"config_suffix": "s", "ckpt_suffix": "small"},
        "tiny": {"config_suffix": "t", "ckpt_suffix": "tiny"},
    }

    model_size = args.model_size
    config_file = f"configs/sam2.1/sam2.1_hiera_{model_mapping[model_size]['config_suffix']}.yaml"
    checkpoint = f"checkpoints/sam2.1_hiera_{model_mapping[model_size]['ckpt_suffix']}.pt"
    outdir = f"checkpoints/{model_size}/"

    print(f"Using model size: {model_size}")
    print(f"Configuration file: {config_file}")
    print(f"Checkpoint file: {checkpoint}")
    print(f"Output directory: {outdir}")
    print(f"Specialized decoder max points: {args.max_points}")

    os.makedirs(outdir, exist_ok=True)

    sam2_model = build_sam2(config_file, checkpoint, device="cpu")
    sam2_model.eval()

    if not args.skip_legacy:
        print("Exporting legacy compatibility artifacts...")
        encoder = ImageEncoder(sam2_model).eval().cpu()
        export_image_encoder(encoder, outdir, name=model_size)

        decoder = ImageDecoder(sam2_model).eval().cpu()
        export_image_decoder(decoder, outdir, name=model_size)

        mem_attn = MemAttention(sam2_model).eval().cpu()
        export_memory_attention(mem_attn, outdir, name=model_size)

        mem_enc = MemEncoder(sam2_model).eval().cpu()
        export_memory_encoder(mem_enc, outdir, name=model_size)

    if not args.skip_specialized:
        print("Exporting specialized task-specific artifacts...")

        if args.experimental_image_points:
            points_decoder = ImageDecoderPredMask(sam2_model).eval().cpu()
            export_image_decoder_points(
                points_decoder,
                outdir,
                max_points=args.max_points,
                name=model_size,
            )
        else:
            _remove_stale_image_point_exports(outdir)
            print(
                "Skipping image_decoder_points.onnx by default because the fixed-shape "
                "image seed-point decoder is experimental and can change SAM prompt semantics. "
                "Pass --experimental_image_points to export it explicitly."
            )

        box_decoder = ImageDecoderPredMask(sam2_model).eval().cpu()
        export_image_decoder_box(box_decoder, outdir, name=model_size)

        video_init_decoder = VideoDecoderInit(sam2_model).eval().cpu()
        export_video_decoder_init(
            video_init_decoder,
            outdir,
            max_points=args.max_points,
            name=model_size,
        )

        video_propagate_decoder = VideoDecoderPropagate(sam2_model).eval().cpu()
        export_video_decoder_propagate(video_propagate_decoder, outdir, name=model_size)

        mem_attn_objptr = MemAttentionObjPtr(sam2_model).eval().cpu()
        export_memory_attention_objptr(mem_attn_objptr, outdir, name=model_size)

        mem_attn_no_objptr = MemAttentionNoObjPtr(sam2_model).eval().cpu()
        export_memory_attention_no_objptr(mem_attn_no_objptr, outdir, name=model_size)

        mem_attn_no_objptr_1frame = MemAttentionNoObjPtr1Frame(sam2_model).eval().cpu()
        export_memory_attention_no_objptr_1frame(mem_attn_no_objptr_1frame, outdir, name=model_size)

        mem_enc_lite = MemEncoderLite(sam2_model).eval().cpu()
        export_memory_encoder_lite(mem_enc_lite, outdir, name=model_size)

    _write_manifest(
        outdir=outdir,
        model_size=model_size,
        max_points=args.max_points,
        skip_legacy=args.skip_legacy,
        skip_specialized=args.skip_specialized,
        experimental_image_points=args.experimental_image_points,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export SAM2 modules to ONNX")
    parser.add_argument(
        "--model_size",
        type=str,
        default="tiny",
        choices=["base_plus", "large", "small", "tiny"],
        help="Model size variant: base_plus, large, small, or tiny",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=8,
        help="Fixed prompt slots for the specialized point-based decoder exports",
    )
    parser.add_argument(
        "--skip_legacy",
        action="store_true",
        help="Skip the existing legacy exports and emit only the specialized variants",
    )
    parser.add_argument(
        "--skip_specialized",
        action="store_true",
        help="Skip the new specialized exports and emit only the legacy artifacts",
    )
    parser.add_argument(
        "--experimental_image_points",
        action="store_true",
        help="Also export the fixed-shape image seed-point decoder (experimental; may reduce quality)",
    )
    args = parser.parse_args()
    main(args)
