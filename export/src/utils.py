# sam2-onnx-cpp/export/src/utils.py
import os
import torch
import onnx
from torch.export import Dim  # modern dynamic-shape API

# Export settings
OPSET = 18
OPTIMIZE = False          # skip onnxscript optimizer (faster, avoids Resize CF hang)
RUN_ONNX_CHECKER = False  # set True if you want onnx.checker validation


def _maybe_check(path: str, extra_msg: str = "") -> None:
    if RUN_ONNX_CHECKER:
        m = onnx.load(path)
        onnx.checker.check_model(m)
    print(f"Exported {extra_msg} to {path}")


def export_image_encoder(model, outdir, name: str | None = None) -> None:
    """
    Image encoder export.

    Outputs:
      0: image_embeddings      [1, 256, 64, 64]
      1: high_res_features1    [1, 32, 256, 256]
      2: high_res_features2    [1, 64, 128, 128]
      3: current_vision_feat   [1, 256, 64, 64]
      4: vision_pos_embed      [4096, 1, 256]
    """
    os.makedirs(outdir, exist_ok=True)
    encoder_path = os.path.join(outdir, "image_encoder.onnx")

    input_img = torch.randn(1, 3, 1024, 1024).float().cpu()
    output_names = [
        "image_embeddings",
        "high_res_features1",
        "high_res_features2",
        "current_vision_feat",
        "vision_pos_embed",
    ]

    torch.onnx.export(
        model,
        input_img,
        encoder_path,
        export_params=True,
        opset_version=OPSET,
        optimize=OPTIMIZE,
        input_names=["input"],
        output_names=output_names,
        # leave use_external_data_format default; large models will produce .onnx.data
    )
    _maybe_check(encoder_path, "encoder with 5 outputs")


def export_image_decoder(model, outdir, name: str | None = None) -> None:
    """
    Image decoder export (points/bbox prompt).

    Inputs:
      point_coords      [Nlabels, Npts, 2]   (dynamic Nlabels, Npts)
      point_labels      [Nlabels, Npts]      (dynamic Nlabels, Npts)
      image_embed       [1, 256, 64, 64]
      high_res_feats_0  [1, 32, 256, 256]
      high_res_feats_1  [1, 64, 128, 128]

    Outputs:
      obj_ptr
      mask_for_mem   [1, M, 1024, 1024]
      pred_mask      [1, M, 256, 256]
    """
    os.makedirs(outdir, exist_ok=True)
    decoder_path = os.path.join(outdir, "image_decoder.onnx")

    # Dummy inputs for tracing
    point_coords = torch.randn(1, 2, 2).float()                # [num_labels, num_points, 2]
    point_labels = torch.randint(0, 2, (1, 2)).float()         # [num_labels, num_points]
    image_embed  = torch.randn(1, 256, 64, 64).float()
    feats_0      = torch.randn(1, 32, 256, 256).float()
    feats_1      = torch.randn(1, 64, 128, 128).float()

    input_names = [
        "point_coords",
        "point_labels",
        "image_embed",
        "high_res_feats_0",
        "high_res_feats_1",
    ]
    output_names = ["obj_ptr", "mask_for_mem", "pred_mask"]

    # Use dynamic_shapes with the new exporter (not dynamic_axes)
    dyn = Dim
    dynamic_shapes = {
        "point_coords": (dyn("num_labels"), dyn("num_points"), 2),
        "point_labels": (dyn("num_labels"), dyn("num_points")),
        "image_embed": (1, 256, 64, 64),
        "high_res_feats_0": (1, 32, 256, 256),
        "high_res_feats_1": (1, 64, 128, 128),
    }

    torch.onnx.export(
        model,
        (point_coords, point_labels, image_embed, feats_0, feats_1),
        decoder_path,
        export_params=True,
        opset_version=OPSET,
        optimize=OPTIMIZE,
        input_names=input_names,
        output_names=output_names,
        dynamic_shapes=dynamic_shapes,
    )
    _maybe_check(decoder_path, "decoder")


def export_memory_attention(model, outdir, name: str | None = None) -> None:
    """
    Memory attention export.

    Inputs:
      current_vision_feat      [1, 256, 64, 64]        (static)
      current_vision_pos_embed [4096, 1, 256]          (static)
      memory_0                 [num_obj_ptrs, 256]     (dynamic axis 0)
      memory_1                 [num_mem_frames, 64, 64, 64]  (dynamic axis 0)
      memory_pos_embed         [buff_size, 1, 64]      (dynamic axis 0)

    Output:
      fused_feat               [1, 256, 64, 64]
    """
    os.makedirs(outdir, exist_ok=True)
    attn_path = os.path.join(outdir, "memory_attention.onnx")

    # Dummy inputs with plausible shapes
    current_vision_feat = torch.randn(1, 256, 64, 64).float()
    current_vision_pos  = torch.randn(4096, 1, 256).float()   # 64*64 tokens = 4096
    memory_0 = torch.randn(16, 256).float()                   # example: 16 object ptrs
    memory_1 = torch.randn(7, 64, 64, 64).float()             # example: 7 memory frames
    memory_pos_embed = torch.randn(7 * 4096 + 64, 1, 64).float()

    dyn = Dim
    dynamic_shapes = {
        "current_vision_feat": (1, 256, 64, 64),
        "current_vision_pos_embed": (4096, 1, 256),  # keep static to avoid constraint errors
        "memory_0": (dyn("num_object_ptrs"), 256),
        "memory_1": (dyn("num_mem_frames"), 64, 64, 64),
        "memory_pos_embed": (dyn("buff_size"), 1, 64),
    }

    torch.onnx.export(
        model,
        (current_vision_feat, current_vision_pos, memory_0, memory_1, memory_pos_embed),
        attn_path,
        export_params=True,
        opset_version=OPSET,
        optimize=OPTIMIZE,
        input_names=[
            "current_vision_feat",
            "current_vision_pos_embed",
            "memory_0",
            "memory_1",
            "memory_pos_embed",
        ],
        output_names=["fused_feat"],
        dynamic_shapes=dynamic_shapes,
    )
    _maybe_check(attn_path, "memory_attention")


def export_memory_encoder(model, outdir, name: str | None = None) -> None:
    """
    Memory encoder export.

    Inputs:
      mask_for_mem  [1, 1, 1024, 1024]
      pix_feat      [1, 256, 64, 64]

    Outputs:
      maskmem_features
      maskmem_pos_enc  [4096, 1, 64]
      temporal_code
    """
    os.makedirs(outdir, exist_ok=True)
    enc_path = os.path.join(outdir, "memory_encoder.onnx")

    dummy_mask = torch.randn(1, 1, 1024, 1024).float()
    dummy_feat = torch.randn(1, 256, 64, 64).float()

    torch.onnx.export(
        model,
        (dummy_mask, dummy_feat),
        enc_path,
        export_params=True,
        opset_version=OPSET,
        optimize=OPTIMIZE,
        input_names=["mask_for_mem", "pix_feat"],
        output_names=["maskmem_features", "maskmem_pos_enc", "temporal_code"],
    )
    _maybe_check(enc_path, "memory_encoder")