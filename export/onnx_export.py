# onnx_export.py

import sys
import os
# Add the repository root (parent directory of 'export') to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from sam2.sam2.build_sam import build_sam2
from src.modules import (
    ImageEncoder, ImageDecoder,
    MemAttention, MemEncoder
)
from src.utils import (
    export_image_encoder, export_image_decoder,
    export_memory_attention, export_memory_encoder
)

def main(args):
    sam2_model = build_sam2(args.config, args.checkpoint, device="cpu")
    
    # 1) Export Image Encoder
    encoder = ImageEncoder(sam2_model).cpu()
    export_image_encoder(encoder, args.outdir)
    
    # 2) Export Image Decoder
    decoder = ImageDecoder(sam2_model).cpu()
    export_image_decoder(decoder, args.outdir)

    # 3) Export Mem Attention
    mem_attn = MemAttention(sam2_model).cpu()
    export_memory_attention(mem_attn, args.outdir)

    # 4) Export Mem Encoder
    mem_enc = MemEncoder(sam2_model).cpu()
    export_memory_encoder(mem_enc, args.outdir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export SAM2 modules to ONNX")
    parser.add_argument("--outdir", type=str, default="checkpoints/base_plus/", help="Output folder")
    parser.add_argument("--config", type=str, default="configs/sam2.1/sam2.1_hiera_b+.yaml", help="Config file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/sam2.1_hiera_base_plus.pt", help="Checkpoint file")
    args = parser.parse_args()
    main(args)
