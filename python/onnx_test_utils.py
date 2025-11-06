# sam2-onnx-cpp/python/onnx_test_utils.py
"""
Shared utilities for onnx_test_image.py and onnx_test_video.py.

Default behavior by platform:
- Windows/Linux: Prefer CUDA if available, else CPU.
- macOS: Default to CPU (Core ML has 16,384-per-axis limits that block big parts of SAM2).
  You can opt-in to Core ML for the encoder with SAM2_ORT_ACCEL=coreml.
  Decoder/memory remain on CPU by default to avoid 0-dim/dynamic shape issues.

Env toggles:
  SAM2_ORT_ACCEL = auto | cpu | cuda | coreml   (default: auto)
  SAM2_ORT_COREML_ALL = 0 | 1   (default: 0)  # if 1, try CoreML on every session (not recommended)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from onnxruntime import InferenceSession

# Make pip-provided CUDA/cuDNN/TensorRT DLLs discoverable for this process (Windows)
try:
    ort.preload_dlls()
except Exception:
    pass

ACCEL = os.getenv("SAM2_ORT_ACCEL", "auto").lower()            # 'auto' | 'cpu' | 'cuda' | 'coreml'
COREML_ALL = os.getenv("SAM2_ORT_COREML_ALL", "0").lower() in ("1", "true", "yes")

# ──────────────────────────────────────────────────────────────────────────────
# Info / environment
# ──────────────────────────────────────────────────────────────────────────────

def print_system_info() -> None:
    print("[INFO] OS :", sys.platform)
    print("[INFO] ONNX Runtime providers (available) :", ort.get_available_providers())


def set_cv2_threads(n: int = 1) -> None:
    try:
        cv2.setNumThreads(n)
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Provider presets
# ──────────────────────────────────────────────────────────────────────────────

def _cuda_providers(device_id: int = 0):
    return [("CUDAExecutionProvider", {
        "device_id": device_id,
        "arena_extend_strategy": "kNextPowerOfTwo",
        "cudnn_conv_algo_search": "HEURISTIC",
        "do_copy_in_default_stream": "1",
    }), "CPUExecutionProvider"]

def _coreml_mlprogram_opts(static: bool = True):
    return [("CoreMLExecutionProvider", {
        "ModelFormat": "MLProgram",
        "MLComputeUnits": "ALL",
        "RequireStaticInputShapes": "1" if static else "0",
        "EnableOnSubgraphs": "0",
        # "ModelCacheDirectory": "cache/coreml",  # Optional: enable if you want disk caching
    }), "CPUExecutionProvider"]

def _coreml_nn_opts(static: bool = True):
    return [("CoreMLExecutionProvider", {
        "ModelFormat": "NeuralNetwork",
        "MLComputeUnits": "ALL",
        "RequireStaticInputShapes": "1" if static else "0",
        "EnableOnSubgraphs": "0",
    }), "CPUExecutionProvider"]


# ──────────────────────────────────────────────────────────────────────────────
# ORT sessions
# ──────────────────────────────────────────────────────────────────────────────

def _create_session_with_fallback(path: str,
                                  so: ort.SessionOptions,
                                  primary_providers,
                                  fallback_providers=("CPUExecutionProvider",),
                                  tag: str = "") -> InferenceSession:
    try:
        return InferenceSession(path, sess_options=so, providers=list(primary_providers))
    except Exception as e:
        print("*************** EP Error ***************")
        print(f"EP Error {type(e).__name__} : {getattr(e, 'args', [''])[0]} "
              f"when using {list(primary_providers)}")
        print(f"Falling back to {list(fallback_providers)} and retrying.")
        print("****************************************")
        return InferenceSession(path, sess_options=so, providers=list(fallback_providers))


def make_encoder_session(path: str,
                         providers: Optional[Iterable[str]] = None) -> InferenceSession:
    path = str(Path(path).resolve())
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    so.intra_op_num_threads = max(1, (os.cpu_count() or 8) - 1)
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    av = ort.get_available_providers()

    # Decide provider attempts (ordered) based on env + availability
    if providers is not None:
        attempt_lists = [list(providers)]
    else:
        if ACCEL == "cpu":
            attempt_lists = [["CPUExecutionProvider"]]
        elif ACCEL == "cuda" and "CUDAExecutionProvider" in av:
            attempt_lists = [_cuda_providers()]
        elif ACCEL == "coreml" and "CoreMLExecutionProvider" in av:
            attempt_lists = [_coreml_mlprogram_opts(static=True),
                             _coreml_nn_opts(static=True),
                             ["CPUExecutionProvider"]]
        else:
            # auto:
            if "CUDAExecutionProvider" in av:
                attempt_lists = [_cuda_providers()]
            elif "CoreMLExecutionProvider" in av:
                # Default on macOS is CPU to avoid CoreML compile stalls; only try CoreML when explicitly asked
                attempt_lists = [["CPUExecutionProvider"]]
            else:
                attempt_lists = [["CPUExecutionProvider"]]

    last_err = None
    for primary in attempt_lists:
        print(f"[INFO] Loading {os.path.basename(path)} [encoder] with providers={primary}")
        try:
            sess = InferenceSession(path, sess_options=so, providers=primary)
            print("[INFO] Active providers:", sess.get_providers())
            print("[INFO] Inputs:", [(i.name, i.shape, i.type) for i in sess.get_inputs()])
            print("[INFO] Outputs:", [o.name for o in sess.get_outputs()])
            return sess
        except Exception as e:
            last_err = e
            print("*************** EP Error ***************")
            print(f"EP Error {type(e).__name__} : {getattr(e, 'args', [''])[0]} when using {primary}")
            print("****************************************")

    print("[WARN] Encoder: all preferred providers failed; using CPUExecutionProvider.")
    sess = InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
    print("[INFO] Active providers:", sess.get_providers())
    print("[INFO] Inputs:", [(i.name, i.shape, i.type) for i in sess.get_inputs()])
    print("[INFO] Outputs:", [o.name for o in sess.get_outputs()])
    return sess


def make_safe_session(path: str,
                      providers: Optional[Iterable[str]] = None,
                      tag: str = "safe") -> InferenceSession:
    path = str(Path(path).resolve())
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.add_session_config_entry("session.disable_gemm_fast_gelu_fusion", "1")

    av = ort.get_available_providers()

    if providers is not None:
        primary = list(providers)
    else:
        if ACCEL == "cpu":
            primary = ["CPUExecutionProvider"]
        elif ACCEL == "cuda" and "CUDAExecutionProvider" in av:
            primary = _cuda_providers()
        elif ACCEL == "coreml" and "CoreMLExecutionProvider" in av:
            # By default, keep decoder/memory on CPU due to 0-dim/dynamic issues.
            if COREML_ALL:
                # If you explicitly want CoreML everywhere (not recommended), try it; fallback will catch failures.
                primary = _coreml_mlprogram_opts(static=True)[0:1] + ["CPUExecutionProvider"]
            else:
                if tag in ("decoder", "memory_encoder", "memory_attention", "safe"):
                    primary = ["CPUExecutionProvider"]
                else:
                    primary = _coreml_mlprogram_opts(static=True)[0:1] + ["CPUExecutionProvider"]
        else:
            # auto:
            if "CUDAExecutionProvider" in av:
                primary = _cuda_providers()
            else:
                # On macOS auto: CPU only (fast, reliable)
                primary = ["CPUExecutionProvider"]

    print(f"[INFO] Loading {os.path.basename(path)} [{tag}] with providers={primary}")
    sess = _create_session_with_fallback(path, so, primary, ("CPUExecutionProvider",), tag=tag)
    print("[INFO] Active providers:", sess.get_providers())
    print("[INFO] Inputs:", [(i.name, i.shape, i.type) for i in sess.get_inputs()])
    print("[INFO] Outputs:", [o.name for o in sess.get_outputs()])
    return sess


# ──────────────────────────────────────────────────────────────────────────────
# Paths / small helpers
# ──────────────────────────────────────────────────────────────────────────────

def prefer_quantized_encoder(ckpt_dir: str | os.PathLike,
                             base_name: str = "image_encoder") -> Optional[str]:
    """
    Pick the best encoder artifact for the current acceleration mode.
    - If accelerating (CUDA or explicit CoreML), prefer float .onnx over int8.
    - If CPU, prefer int8 when available.
    """
    av = ort.get_available_providers()

    # Are we actually trying to accelerate?
    accel = False
    if ACCEL == "cuda" and "CUDAExecutionProvider" in av:
        accel = True
    elif ACCEL == "coreml" and "CoreMLExecutionProvider" in av:
        accel = True
    elif ACCEL == "auto" and "CUDAExecutionProvider" in av:
        accel = True  # auto only accelerates by default for CUDA; mac auto stays CPU

    if accel:
        order = [f"{base_name}.onnx", f"{base_name}.int8.onnx"]
    else:
        order = [f"{base_name}.int8.onnx", f"{base_name}.onnx"]
        
    ckpt_path = Path(ckpt_dir)
    for fname in order:
        p = ckpt_path / fname
        if p.exists():
            return str(p.resolve())
    return None


def as_f32c(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32, copy=False)
    return np.ascontiguousarray(a)


# ──────────────────────────────────────────────────────────────────────────────
# Image preprocessing
# ──────────────────────────────────────────────────────────────────────────────

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD  = np.array([0.229, 0.224, 0.225], np.float32)

def bgr_to_input_tensor(img_bgr: np.ndarray,
                        enc_hw: Tuple[int, int]) -> np.ndarray:
    h_enc, w_enc = enc_hw
    img_resized = cv2.resize(img_bgr, (w_enc, h_enc))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_rgb = (img_rgb - _MEAN) / _STD
    tensor = np.transpose(img_rgb, (2, 0, 1))[np.newaxis, :]
    return as_f32c(tensor)


def prepare_image(img_bgr: np.ndarray,
                  enc_hw: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[int, int]]:
    H_org, W_org = img_bgr.shape[:2]
    return bgr_to_input_tensor(img_bgr, enc_hw), (H_org, W_org)


def compute_display_base(img_bgr: np.ndarray,
                         max_side: int = 1200
                         ) -> Tuple[np.ndarray, float]:
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


def prepare_rectangle(rect, img_size, enc_size):
    return prepare_box_prompt(rect, img_size, enc_size)


# ──────────────────────────────────────────────────────────────────────────────
# Encoder / Decoder runners
# ──────────────────────────────────────────────────────────────────────────────

def run_encoder(sess_enc: InferenceSession,
                input_tensor: np.ndarray) -> Dict[str, np.ndarray]:
    enc_input_name = sess_enc.get_inputs()[0].name
    out_names = [o.name for o in sess_enc.get_outputs()]
    values = sess_enc.run(None, {enc_input_name: as_f32c(input_tensor)})
    return dict(zip(out_names, values))


def _decoder_io_names(sess_dec: InferenceSession) -> Dict[str, str]:
    inps = [i.name for i in sess_dec.get_inputs()]

    def find(key: str) -> str:
        for nm in inps:
            if key in nm:
                return nm
        return key

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
    fg = (mask255 > 0)
    color = np.zeros_like(bgr)
    color[fg] = (0, 255, 0)
    return cv2.addWeighted(bgr, 1.0, color, alpha, 0)
