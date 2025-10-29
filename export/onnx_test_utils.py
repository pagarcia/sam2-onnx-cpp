# sam2-onnx-cpp/export/onnx_test_utils.py
"""
Shared utilities for onnx_test_image.py and onnx_test_video.py.

Pulled-out common functionality:
- System info print
- ONNX Runtime session builders (fast encoder / safe others)
- Preprocessing (BGR -> normalized NCHW tensor)
- Prompt preparation (points/box) from image space -> encoder space
- Overlay helpers
- Robust encoder/decoder runners (name-agnostic I/O mapping)
- Small convenience helpers (paths, display scaling, OpenCV threads)

All functions are NumPy-first and keep arrays contiguous float32 where relevant.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from onnxruntime import InferenceSession


# ──────────────────────────────────────────────────────────────────────────────
# Info / environment
# ──────────────────────────────────────────────────────────────────────────────

def print_system_info() -> None:
    """Print a tiny system banner (OS + ORT providers)."""
    print("[INFO] OS :", sys.platform)
    print("[INFO] ONNX Runtime providers (available) :", ort.get_available_providers())


def set_cv2_threads(n: int = 1) -> None:
    """Keep OpenCV from oversubscribing threads."""
    try:
        cv2.setNumThreads(n)
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# ORT sessions
# ──────────────────────────────────────────────────────────────────────────────

def make_encoder_session(path: str,
                         providers: Optional[Iterable[str]] = None) -> InferenceSession:
    """
    Fast encoder session (aggressive graph opts, tuned threading).
    """
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    so.intra_op_num_threads = max(1, (os.cpu_count() or 8) - 1)
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    providers = list(providers) if providers else ["CPUExecutionProvider"]
    print(f"[INFO] Loading {os.path.basename(path)} [encoder] with providers={providers}")
    sess = InferenceSession(path, sess_options=so, providers=providers)
    print("[INFO] Inputs:", [(i.name, i.shape, i.type) for i in sess.get_inputs()])
    print("[INFO] Outputs:", [o.name for o in sess.get_outputs()])
    return sess


def make_safe_session(path: str,
                      providers: Optional[Iterable[str]] = None,
                      tag: str = "safe") -> InferenceSession:
    """
    Conservative session (no risky fusions). Use for decoder/memory graphs.
    """
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    # This fusion has caused bad Gemm shapes in this project; keep it off.
    so.add_session_config_entry("session.disable_gemm_fast_gelu_fusion", "1")
    providers = list(providers) if providers else ["CPUExecutionProvider"]
    print(f"[INFO] Loading {os.path.basename(path)} [{tag}] with providers={providers}")
    sess = InferenceSession(path, sess_options=so, providers=providers)
    print("[INFO] Inputs:", [(i.name, i.shape, i.type) for i in sess.get_inputs()])
    print("[INFO] Outputs:", [o.name for o in sess.get_outputs()])
    return sess


# ──────────────────────────────────────────────────────────────────────────────
# Paths / small helpers
# ──────────────────────────────────────────────────────────────────────────────

def prefer_quantized_encoder(ckpt_dir: str,
                             base_name: str = "image_encoder") -> Optional[str]:
    """
    Prefer an int8 encoder ONNX if present, otherwise the float model.
    Returns the chosen path or None if neither exists.
    """
    candidates = [
        os.path.join(ckpt_dir, f"{base_name}.int8.onnx"),
        os.path.join(ckpt_dir, f"{base_name}.onnx"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def as_f32c(a: np.ndarray) -> np.ndarray:
    """Return contiguous float32 view/copy of array."""
    a = a.astype(np.float32, copy=False)
    return np.ascontiguousarray(a)


# ──────────────────────────────────────────────────────────────────────────────
# Image preprocessing
# ──────────────────────────────────────────────────────────────────────────────

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD  = np.array([0.229, 0.224, 0.225], np.float32)

def bgr_to_input_tensor(img_bgr: np.ndarray,
                        enc_hw: Tuple[int, int]) -> np.ndarray:
    """
    Convert BGR image to normalized float32 tensor [1,3,H,W] for the encoder.
    """
    h_enc, w_enc = enc_hw
    img_resized = cv2.resize(img_bgr, (w_enc, h_enc))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_rgb = (img_rgb - _MEAN) / _STD
    tensor = np.transpose(img_rgb, (2, 0, 1))[np.newaxis, :]
    return as_f32c(tensor)


def prepare_image(img_bgr: np.ndarray,
                  enc_hw: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Convenience wrapper returning (input_tensor, (H_orig, W_orig)).
    """
    H_org, W_org = img_bgr.shape[:2]
    return bgr_to_input_tensor(img_bgr, enc_hw), (H_org, W_org)


def compute_display_base(img_bgr: np.ndarray,
                         max_side: int = 1200
                         ) -> Tuple[np.ndarray, float]:
    """
    Create a display-sized copy and its scale factor.
    """
    H, W = img_bgr.shape[:2]
    scale = min(1.0, max_side / max(W, H))
    disp = cv2.resize(img_bgr, (int(W * scale), int(H * scale)))
    return disp, scale


# ──────────────────────────────────────────────────────────────────────────────
# Prompt preparation (image space -> encoder space)
# ──────────────────────────────────────────────────────────────────────────────

def prepare_points(points: Iterable[Tuple[int, int]],
                   labels: Iterable[int | float],
                   img_size: Tuple[int, int],
                   enc_size: Tuple[int, int]
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map user points (pixels in original image) to encoder resolution.
    Returns:
      coords: [1, N, 2] float32
      labels: [1, N]   float32
    """
    if not points:
        return (np.zeros((1, 0, 2), np.float32), np.zeros((1, 0), np.float32))
    pts = np.asarray(points, dtype=np.float32)
    lbl = np.asarray(labels, dtype=np.float32)
    H_org, W_org = img_size
    H_enc, W_enc = enc_size
    pts[:, 0] = (pts[:, 0] / W_org) * W_enc
    pts[:, 1] = (pts[:, 1] / H_org) * H_enc
    return pts[np.newaxis, ...], lbl[np.newaxis, ...]


def prepare_box_prompt(rect: Optional[Tuple[int, int, int, int]],
                       img_size: Tuple[int, int],
                       enc_size: Tuple[int, int]
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert (x1,y1,x2,y2) box into SAM2 'box as 2 points' prompt.
    Labels are [2, 3] per SAM convention (top-left, bottom-right).
    """
    if rect is None:
        return (np.zeros((1, 0, 2), np.float32), np.zeros((1, 0), np.float32))
    x1, y1, x2, y2 = rect
    H_org, W_org = img_size
    H_enc, W_enc = enc_size
    pts = np.array([[x1, y1], [x2, y2]], np.float32)
    pts[:, 0] = (pts[:, 0] / W_org) * W_enc
    pts[:, 1] = (pts[:, 1] / H_org) * H_enc
    lbl = np.array([2.0, 3.0], np.float32)
    return pts[np.newaxis, ...], lbl[np.newaxis, ...]


# Backwards-compat alias for the image test script’s name
def prepare_rectangle(rect, img_size, enc_size):
    return prepare_box_prompt(rect, img_size, enc_size)


# ──────────────────────────────────────────────────────────────────────────────
# Encoder / Decoder runners
# ──────────────────────────────────────────────────────────────────────────────

def run_encoder(sess_enc: InferenceSession,
                input_tensor: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Execute the encoder and return a name->array dict using the model's output names:
      - "image_embeddings"
      - "high_res_features1"
      - "high_res_features2"
      - "current_vision_feat"
      - "vision_pos_embed"
    """
    enc_input_name = sess_enc.get_inputs()[0].name
    out_names = [o.name for o in sess_enc.get_outputs()]
    values = sess_enc.run(None, {enc_input_name: as_f32c(input_tensor)})
    return dict(zip(out_names, values))


def _decoder_io_names(sess_dec: InferenceSession) -> Dict[str, str]:
    """
    Robustly resolve input names (some exporters add suffixes). Falls back to
    canonical names if an exact match isn't found.
    """
    inps = [i.name for i in sess_dec.get_inputs()]

    def find(key: str) -> str:
        for nm in inps:
            if key in nm:
                return nm
        return key  # fallback to canonical

    return {
        "point_coords":     find("point_coords"),
        "point_labels":     find("point_labels"),
        "image_embed":      find("image_embed"),
        "high_res_feats_0": find("high_res_feats_0"),
        "high_res_feats_1": find("high_res_feats_1"),
    }


def run_decoder(sess_dec: InferenceSession,
                point_coords: Optional[np.ndarray],
                point_labels: Optional[np.ndarray],
                image_embed: np.ndarray,
                high_res_feats_0: np.ndarray,
                high_res_feats_1: np.ndarray):
    """
    Execute the image decoder with dynamic I/O name resolution.
    If point_coords/labels are None, feeds an empty prompt of shape [1,0,...].
    Returns (obj_ptr, mask_for_mem, pred_mask) as produced by the model.
    """
    io = _decoder_io_names(sess_dec)

    if point_coords is None or point_labels is None:
        point_coords = np.zeros((1, 0, 2), np.float32)
        point_labels = np.zeros((1, 0),    np.float32)

    feed = {
        io["point_coords"]:     as_f32c(point_coords),
        io["point_labels"]:     as_f32c(point_labels),
        io["image_embed"]:      as_f32c(image_embed),
        io["high_res_feats_0"]: as_f32c(high_res_feats_0),
        io["high_res_feats_1"]: as_f32c(high_res_feats_1),
    }
    return sess_dec.run(None, feed)


# ──────────────────────────────────────────────────────────────────────────────
# Visualization helpers
# ──────────────────────────────────────────────────────────────────────────────

def green_overlay(bgr: np.ndarray,
                  mask255: np.ndarray,
                  alpha: float = 0.5) -> np.ndarray:
    """
    Alpha-blend a solid green overlay where mask > 0.
    Accepts masks in {0,1}, {0,255}, or any nonzero-valued mask.
    """
    # treat any positive value as foreground
    fg = (mask255 > 0)
    color = np.zeros_like(bgr)
    color[fg] = (0, 255, 0)
    return cv2.addWeighted(bgr, 1.0, color, alpha, 0)