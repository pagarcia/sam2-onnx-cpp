# src/utils.py
import os
import torch
import onnx
from onnxruntime import InferenceSession

def export_image_encoder(model, outdir):
    encoder_path = os.path.join(outdir, "image_encoder.onnx")
    input_img = torch.randn(1, 3, 1024, 1024).cpu()
    output_names = [
        "image_embeddings",       # [1,256,64,64]
        "high_res_features1",     # [1,32,256,256]
        "high_res_features2",     # [1,64,128,128]
        "current_vision_feat",    # [4096,1,256]
        "vision_pos_embed"        # [4096,1,256]
    ]
    torch.onnx.export(
        model,
        input_img,
        encoder_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=output_names,
    )
    onnx_model = onnx.load(encoder_path)
    onnx.checker.check_model(onnx_model)
    print(f"Exported encoder with 5 outputs to {encoder_path}")

def export_image_decoder(model, outdir):
    decoder_path = os.path.join(outdir, "image_decoder.onnx")

    # 5 dummy inputs for the new signature
    point_coords = torch.randn(1, 2, 2).float()    # [1, num_points, 2]
    point_labels = torch.randint(0, 2, (1, 2)).float() # [1, num_points]
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

    # If you want dynamic shapes:
    dynamic_axes = {
        "point_coords": {0: "num_labels", 1: "num_points"},
        "point_labels": {0: "num_labels", 1: "num_points"},
    }

    torch.onnx.export(
        model,
        (point_coords, point_labels, image_embed, feats_0, feats_1),
        decoder_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )
    onnx_model = onnx.load(decoder_path)
    onnx.checker.check_model(onnx_model)
    print(f"Exported *5-input* decoder (no frame_size) to {decoder_path}")

def export_memory_attention(model, outdir):
    attn_path = os.path.join(outdir, "memory_attention.onnx")
    # Create dummy inputs of shapes that match your MemAttention wrapper
    current_vision_feat = torch.randn(1,256,64,64)
    current_vision_pos  = torch.randn(4096,1,256)
    memory_0 = torch.randn(16,256)
    memory_1 = torch.randn(7,64,64,64)
    memory_pos_embed = torch.randn(7*4096+64,1,64)

    # dynamic_axes: for example we say axis=0 of memory_0 is "mem_num_obj" ...
    dynamic_axes = {
      "memory_0": {0: "num_object_ptrs"},
      "memory_1": {0: "num_mem_frames"},
      "memory_pos_embed": {0: "buff_size"},
    }

    torch.onnx.export(
        model,
        (current_vision_feat, current_vision_pos,
         memory_0, memory_1, memory_pos_embed),
        attn_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=[
          "current_vision_feat","current_vision_pos_embed",
          "memory_0","memory_1","memory_pos_embed",
        ],
        output_names=["fused_feat"],  # or "image_embed"
        dynamic_axes=dynamic_axes
    )
    onnx_model = onnx.load(attn_path)
    onnx.checker.check_model(onnx_model)
    print(f"Exported memory_attention to {attn_path}")


def export_memory_encoder(model, outdir):
    enc_path = os.path.join(outdir, "memory_encoder.onnx")
    dummy_mask = torch.randn(1,1,1024,1024)  # e.g. [1,1,1024,1024]
    dummy_feat = torch.randn(1,256,64,64)    # e.g. [1,256,64,64]

    torch.onnx.export(
        model,
        (dummy_mask, dummy_feat),
        enc_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["mask_for_mem","pix_feat"],
        output_names=["maskmem_features","maskmem_pos_enc","temporal_code"],
    )
    onnx_model = onnx.load(enc_path)
    onnx.checker.check_model(onnx_model)
    print(f"Exported memory_encoder to {enc_path}")