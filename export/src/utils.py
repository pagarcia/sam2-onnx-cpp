import os
import shutil

import onnx
import torch
from torch.export import Dim

# Export settings
OPSET = 18
OPTIMIZE = False          # skip onnxscript optimizer (faster, avoids Resize CF hang)
RUN_ONNX_CHECKER = False  # set True if you want onnx.checker validation


def _maybe_check(path: str, extra_msg: str = "") -> None:
    if RUN_ONNX_CHECKER:
        model = onnx.load(path)
        onnx.checker.check_model(model)
    print(f"Exported {extra_msg} to {path}")


def _safe_remove(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def _decoder_backbone_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image_embed = torch.randn(1, 256, 64, 64).float()
    feats_0 = torch.randn(1, 32, 256, 256).float()
    feats_1 = torch.randn(1, 64, 128, 128).float()
    return image_embed, feats_0, feats_1


def _points_prompt_inputs(max_points: int) -> tuple[torch.Tensor, torch.Tensor]:
    point_coords = torch.zeros(1, max_points, 2).float()
    point_labels = -torch.ones(1, max_points).float()
    if max_points >= 1:
        point_coords[0, 0] = torch.tensor([256.0, 256.0])
        point_labels[0, 0] = 1.0
    if max_points >= 2:
        point_coords[0, 1] = torch.tensor([768.0, 768.0])
        point_labels[0, 1] = 0.0
    return point_coords, point_labels


def _box_prompt_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    point_coords = torch.tensor([[[128.0, 128.0], [896.0, 896.0]]], dtype=torch.float32)
    point_labels = torch.tensor([[2.0, 3.0]], dtype=torch.float32)
    return point_coords, point_labels


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
    )
    _maybe_check(encoder_path, "encoder with 5 outputs")


def export_image_decoder(model, outdir, name: str | None = None) -> None:
    """
    Legacy image decoder export with dynamic prompt axes.

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

    point_coords, point_labels = _box_prompt_inputs()
    image_embed, feats_0, feats_1 = _decoder_backbone_inputs()

    input_names = [
        "point_coords",
        "point_labels",
        "image_embed",
        "high_res_feats_0",
        "high_res_feats_1",
    ]
    output_names = ["obj_ptr", "mask_for_mem", "pred_mask"]
    dynamic_shapes = {
        "point_coords": (Dim("num_labels"), Dim("num_points"), 2),
        "point_labels": (Dim("num_labels"), Dim("num_points")),
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
    _maybe_check(decoder_path, "legacy decoder")


def export_image_decoder_points(
    model,
    outdir,
    max_points: int = 8,
    name: str | None = None,
) -> None:
    os.makedirs(outdir, exist_ok=True)
    decoder_path = os.path.join(outdir, "image_decoder_points.onnx")

    point_coords, point_labels = _points_prompt_inputs(max_points)
    image_embed, feats_0, feats_1 = _decoder_backbone_inputs()

    torch.onnx.export(
        model,
        (point_coords, point_labels, image_embed, feats_0, feats_1),
        decoder_path,
        export_params=True,
        opset_version=OPSET,
        optimize=OPTIMIZE,
        input_names=[
            "point_coords",
            "point_labels",
            "image_embed",
            "high_res_feats_0",
            "high_res_feats_1",
        ],
        output_names=["pred_mask"],
    )
    _maybe_check(decoder_path, f"fixed-shape points decoder (max_points={max_points})")


def export_image_decoder_box(model, outdir, name: str | None = None) -> None:
    os.makedirs(outdir, exist_ok=True)
    decoder_path = os.path.join(outdir, "image_decoder_box.onnx")

    point_coords, point_labels = _box_prompt_inputs()
    image_embed, feats_0, feats_1 = _decoder_backbone_inputs()

    torch.onnx.export(
        model,
        (point_coords, point_labels, image_embed, feats_0, feats_1),
        decoder_path,
        export_params=True,
        opset_version=OPSET,
        optimize=OPTIMIZE,
        input_names=[
            "point_coords",
            "point_labels",
            "image_embed",
            "high_res_feats_0",
            "high_res_feats_1",
        ],
        output_names=["pred_mask"],
    )
    _maybe_check(decoder_path, "fixed-shape box decoder")


def export_video_decoder_init(
    model,
    outdir,
    max_points: int = 8,
    name: str | None = None,
) -> None:
    os.makedirs(outdir, exist_ok=True)
    decoder_path = os.path.join(outdir, "video_decoder_init.onnx")

    point_coords, point_labels = _points_prompt_inputs(max_points)
    image_embed, feats_0, feats_1 = _decoder_backbone_inputs()

    torch.onnx.export(
        model,
        (point_coords, point_labels, image_embed, feats_0, feats_1),
        decoder_path,
        export_params=True,
        opset_version=OPSET,
        optimize=OPTIMIZE,
        input_names=[
            "point_coords",
            "point_labels",
            "image_embed",
            "high_res_feats_0",
            "high_res_feats_1",
        ],
        output_names=["obj_ptr", "mask_for_mem", "pred_mask"],
    )
    _maybe_check(decoder_path, f"video init decoder (max_points={max_points})")


def export_video_decoder_propagate(model, outdir, name: str | None = None) -> None:
    os.makedirs(outdir, exist_ok=True)
    decoder_path = os.path.join(outdir, "video_decoder_propagate.onnx")

    image_embed, feats_0, feats_1 = _decoder_backbone_inputs()

    torch.onnx.export(
        model,
        (image_embed, feats_0, feats_1),
        decoder_path,
        export_params=True,
        opset_version=OPSET,
        optimize=OPTIMIZE,
        input_names=["image_embed", "high_res_feats_0", "high_res_feats_1"],
        output_names=["obj_ptr", "mask_for_mem", "pred_mask"],
    )
    _maybe_check(decoder_path, "prompt-free video propagate decoder")


def export_memory_attention(model, outdir, name: str | None = None) -> None:
    """
    Legacy memory attention export with object pointers.

    Inputs:
      current_vision_feat      [1, 256, 64, 64]             (static)
      current_vision_pos_embed [4096, 1, 256]               (static)
      memory_0                 [num_obj_ptrs, 256]          (dynamic axis 0)
      memory_1                 [num_mem_frames, 64, 64, 64] (dynamic axis 0)
      memory_pos_embed         [buff_size, 1, 64]           (dynamic axis 0)

    Output:
      fused_feat               [1, 256, 64, 64]
    """
    os.makedirs(outdir, exist_ok=True)
    attn_path = os.path.join(outdir, "memory_attention.onnx")

    current_vision_feat = torch.randn(1, 256, 64, 64).float()
    current_vision_pos = torch.randn(4096, 1, 256).float()
    memory_0 = torch.randn(16, 256).float()
    memory_1 = torch.randn(7, 64, 64, 64).float()
    memory_pos_embed = torch.randn(7 * 4096 + 64, 1, 64).float()

    dynamic_shapes = {
        "current_vision_feat": (1, 256, 64, 64),
        "current_vision_pos_embed": (4096, 1, 256),
        "memory_0": (Dim("num_object_ptrs"), 256),
        "memory_1": (Dim("num_mem_frames"), 64, 64, 64),
        "memory_pos_embed": (Dim("buff_size"), 1, 64),
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
    _maybe_check(attn_path, "legacy memory_attention")


def export_memory_attention_no_objptr(model, outdir, name: str | None = None) -> None:
    os.makedirs(outdir, exist_ok=True)
    attn_path = os.path.join(outdir, "memory_attention_no_objptr.onnx")

    current_vision_feat = torch.randn(1, 256, 64, 64).float()
    current_vision_pos = torch.randn(4096, 1, 256).float()
    memory_1 = torch.randn(7, 64, 64, 64).float()
    memory_pos_embed = torch.randn(7 * 4096, 1, 64).float()

    dynamic_shapes = {
        "current_vision_feat": (1, 256, 64, 64),
        "current_vision_pos_embed": (4096, 1, 256),
        "memory_1": (Dim("num_mem_frames"), 64, 64, 64),
        "memory_pos_embed": (Dim("buff_size"), 1, 64),
    }

    torch.onnx.export(
        model,
        (current_vision_feat, current_vision_pos, memory_1, memory_pos_embed),
        attn_path,
        export_params=True,
        opset_version=OPSET,
        optimize=OPTIMIZE,
        input_names=[
            "current_vision_feat",
            "current_vision_pos_embed",
            "memory_1",
            "memory_pos_embed",
        ],
        output_names=["fused_feat"],
        dynamic_shapes=dynamic_shapes,
    )
    _maybe_check(attn_path, "memory_attention without object pointers")


def export_memory_attention_objptr(model, outdir, name: str | None = None) -> None:
    os.makedirs(outdir, exist_ok=True)
    attn_path = os.path.join(outdir, "memory_attention_objptr.onnx")

    current_vision_feat = torch.randn(1, 256, 64, 64).float()
    current_vision_pos = torch.randn(4096, 1, 256).float()
    memory_0 = torch.randn(8, 256).float()
    obj_ptr_offsets = torch.arange(1, 9, dtype=torch.float32)
    memory_1 = torch.randn(7, 64, 64, 64).float()
    memory_pos_embed = torch.randn(7 * 4096, 1, 64).float()

    dynamic_shapes = {
        "current_vision_feat": (1, 256, 64, 64),
        "current_vision_pos_embed": (4096, 1, 256),
        "memory_0": (Dim("num_object_ptrs"), 256),
        "obj_ptr_offsets": (Dim("num_object_ptrs"),),
        "memory_1": (Dim("num_mem_frames"), 64, 64, 64),
        "memory_pos_embed": (Dim("buff_size"), 1, 64),
    }

    torch.onnx.export(
        model,
        (current_vision_feat, current_vision_pos, memory_0, obj_ptr_offsets, memory_1, memory_pos_embed),
        attn_path,
        export_params=True,
        opset_version=OPSET,
        optimize=OPTIMIZE,
        input_names=[
            "current_vision_feat",
            "current_vision_pos_embed",
            "memory_0",
            "obj_ptr_offsets",
            "memory_1",
            "memory_pos_embed",
        ],
        output_names=["fused_feat"],
        dynamic_shapes=dynamic_shapes,
    )
    _maybe_check(attn_path, "memory_attention with object pointers and offsets")


def export_memory_attention_no_objptr_1frame(model, outdir, name: str | None = None) -> None:
    os.makedirs(outdir, exist_ok=True)
    attn_path = os.path.join(outdir, "memory_attention_no_objptr_1frame.onnx")

    current_vision_feat = torch.randn(1, 256, 64, 64).float()
    memory_1 = torch.randn(1, 64, 64, 64).float()
    memory_pos_embed = torch.randn(4096, 1, 64).float()

    torch.onnx.export(
        model,
        (current_vision_feat, memory_1, memory_pos_embed),
        attn_path,
        export_params=True,
        opset_version=OPSET,
        optimize=OPTIMIZE,
        input_names=[
            "current_vision_feat",
            "memory_1",
            "memory_pos_embed",
        ],
        output_names=["fused_feat"],
    )
    _maybe_check(attn_path, "memory_attention without object pointers (1-frame static)")


def export_memory_encoder(model, outdir, name: str | None = None) -> None:
    """
    Legacy memory encoder export.

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
    _maybe_check(enc_path, "legacy memory_encoder")


def export_memory_encoder_lite(model, outdir, name: str | None = None) -> None:
    os.makedirs(outdir, exist_ok=True)
    enc_path = os.path.join(outdir, "memory_encoder_lite.onnx")
    legacy_path = os.path.join(outdir, "memory_encoder.onnx")

    dummy_mask = torch.randn(1, 1, 1024, 1024).float()
    dummy_feat = torch.randn(1, 256, 64, 64).float()

    try:
        torch.onnx.export(
            model,
            (dummy_mask, dummy_feat),
            enc_path,
            export_params=True,
            opset_version=OPSET,
            optimize=OPTIMIZE,
            input_names=["mask_for_mem", "pix_feat"],
            output_names=["maskmem_features", "maskmem_pos_enc"],
        )
        _maybe_check(enc_path, "memory_encoder without temporal_code")
    except TypeError as exc:
        if not os.path.exists(legacy_path):
            raise

        _safe_remove(enc_path)
        _safe_remove(enc_path + ".data")
        shutil.copyfile(legacy_path, enc_path)
        print(
            "[WARN] memory_encoder_lite export hit a fake-tensor save issue; "
            "copied legacy memory_encoder.onnx as a compatible fallback.\n"
            f"[WARN] Original error: {exc}"
        )
        _maybe_check(enc_path, "memory_encoder lite fallback (legacy-compatible)")
