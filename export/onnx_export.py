import sys
import os
# Add the repository root (one level up from export) so Python can find the sam2 package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Also add the current directory (export) so that the src folder can be found.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import argparse
import torch
from sam2.build_sam import build_sam2
from src.modules import (
    ImageEncoder, ImageDecoder,
    MemAttention, MemEncoder
)
from src.utils import (
    export_image_encoder, export_image_decoder,
    export_memory_attention, export_memory_encoder
)

def main(args):
    # Mapping of model_size to configuration and checkpoint naming.
    model_mapping = {
        "base_plus": {"config_suffix": "b+", "ckpt_suffix": "base_plus"},
        "large": {"config_suffix": "l", "ckpt_suffix": "large"},
        "small": {"config_suffix": "s", "ckpt_suffix": "small"},
        "tiny": {"config_suffix": "t", "ckpt_suffix": "tiny"},
    }
    # Compute config, checkpoint, and outdir based on model_size.
    model_size = args.model_size
    config_file = f"configs/sam2.1/sam2.1_hiera_{model_mapping[model_size]['config_suffix']}.yaml"
    checkpoint = f"checkpoints/sam2.1_hiera_{model_mapping[model_size]['ckpt_suffix']}.pt"
    outdir = f"checkpoints/{model_size}/"
    
    print(f"Using model size: {model_size}")
    print(f"Configuration file: {config_file}")
    print(f"Checkpoint file: {checkpoint}")
    print(f"Output directory: {outdir}")
    
    # Ensure the output directory exists
    os.makedirs(outdir, exist_ok=True)
    
    # Build model on CPU (or GPU if you prefer) and set eval mode
    sam2_model = build_sam2(config_file, checkpoint, device="cpu")
    sam2_model.eval()

    # 1) Export Image Encoder
    encoder = ImageEncoder(sam2_model).eval().cpu()
    export_image_encoder(encoder, outdir, name=model_size)
    
    # 2) Export Image Decoder
    decoder = ImageDecoder(sam2_model).eval().cpu()
    export_image_decoder(decoder, outdir, name=model_size)

    # 3) Export Memory Attention
    mem_attn = MemAttention(sam2_model).eval().cpu()
    export_memory_attention(mem_attn, outdir, name=model_size)

    # 4) Export Memory Encoder
    mem_enc = MemEncoder(sam2_model).eval().cpu()
    export_memory_encoder(mem_enc, outdir, name=model_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export SAM2 modules to ONNX")
    parser.add_argument(
        "--model_size",
        type=str,
        default="tiny",
        choices=["base_plus", "large", "small", "tiny"],
        help="Model size variant: base_plus, large, small, or tiny"
    )
    args = parser.parse_args()
    main(args)
