"""
Shared utilities for Python ONNX demos and benchmarks.

Default behavior by platform:
- Windows/Linux: Prefer CUDA if available, else CPU.
- macOS: Default to CPU (Core ML has 16,384-per-axis limits that block big parts of SAM2).
  You can opt-in to Core ML for the encoder with SAM2_ORT_ACCEL=coreml.
  Decoder/memory remain on CPU by default to avoid 0-dim/dynamic shape issues.

Env toggles:
  SAM2_ORT_ACCEL = auto | cpu | cuda | coreml   (default: auto)
  SAM2_ORT_COREML_ALL = 0 | 1   (default: 0)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from onnxruntime import InferenceSession

try:
    ort.preload_dlls()
except Exception:
    pass

RUNTIME_PROFILE = os.getenv("SAM2_ORT_RUNTIME_PROFILE", "").lower()
CPU_LOW_COST_PROFILE = RUNTIME_PROFILE in ("cpu_lowcost", "lowcost_cpu", "cpu-lowcost", "low-cost-cpu")
_ACCEL_ENV = os.getenv("SAM2_ORT_ACCEL")
ACCEL = (_ACCEL_ENV.lower() if _ACCEL_ENV is not None else ("cpu" if CPU_LOW_COST_PROFILE else "auto"))
COREML_ALL = os.getenv("SAM2_ORT_COREML_ALL", "0").lower() in ("1", "true", "yes")
STATIC_DECODER_OPT = os.getenv("SAM2_ORT_STATIC_DECODER_OPT", "1").lower() in ("1", "true", "yes")
EXPERIMENTAL_1FRAME_ATTN = os.getenv("SAM2_ORT_EXPERIMENTAL_1FRAME_ATTN", "0").lower() in ("1", "true", "yes")
EXPERIMENTAL_IMAGE_POINT_DECODER = os.getenv("SAM2_ORT_EXPERIMENTAL_IMAGE_POINT_DECODER", "0").lower() in ("1", "true", "yes")
EXPERIMENTAL_VIDEO_INIT_DECODER = os.getenv("SAM2_ORT_EXPERIMENTAL_VIDEO_INIT_DECODER", "0").lower() in ("1", "true", "yes")
VIDEO_AUTO_POLICY = os.getenv("SAM2_ORT_VIDEO_AUTO_POLICY", "speed" if CPU_LOW_COST_PROFILE else "correctness").lower()
ENCODER_VARIANT = os.getenv("SAM2_ORT_ENCODER_VARIANT", "auto").lower()


def _env_int(name: str, fallback: int, minimum: int = 0) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return fallback
    try:
        return max(int(value), minimum)
    except Exception:
        return fallback


_STATIC_SPECIALIZED_DECODERS = {
    "image_decoder_points.onnx",
    "image_decoder_box.onnx",
    "video_decoder_init.onnx",
    "video_decoder_propagate.onnx",
    "memory_attention_no_objptr_1frame.onnx",
}
DEFAULT_VIDEO_MEMORY_SLOTS = _env_int(
    "SAM2_ORT_VIDEO_MAX_MEMORY_FRAMES",
    3 if CPU_LOW_COST_PROFILE else (4 if VIDEO_AUTO_POLICY == "speed" else 7),
    minimum=1,
)
DEFAULT_VIDEO_OBJECT_POINTER_SLOTS = _env_int(
    "SAM2_ORT_VIDEO_MAX_OBJECT_POINTERS",
    4 if CPU_LOW_COST_PROFILE else (8 if VIDEO_AUTO_POLICY == "speed" else 16),
    minimum=1,
)
DEFAULT_CPU_THREADS = _env_int(
    "SAM2_ORT_CPU_THREADS",
    min(4, max(1, os.cpu_count() or 4)) if CPU_LOW_COST_PROFILE else max(1, (os.cpu_count() or 8) - 1),
    minimum=1,
)


def print_system_info() -> None:
    print("[INFO] OS :", sys.platform)
    print("[INFO] ONNX Runtime providers (available) :", ort.get_available_providers())
    print("[INFO] Runtime profile :", RUNTIME_PROFILE or "default")


def set_cv2_threads(n: int = 1) -> None:
    try:
        cv2.setNumThreads(n)
    except Exception:
        pass


def _cuda_providers(device_id: int = 0):
    return [
        (
            "CUDAExecutionProvider",
            {
                "device_id": device_id,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "cudnn_conv_algo_search": "HEURISTIC",
                "do_copy_in_default_stream": "1",
            },
        ),
        "CPUExecutionProvider",
    ]


def _coreml_mlprogram_opts(static: bool = True):
    return [
        (
            "CoreMLExecutionProvider",
            {
                "ModelFormat": "MLProgram",
                "MLComputeUnits": "ALL",
                "RequireStaticInputShapes": "1" if static else "0",
                "EnableOnSubgraphs": "0",
            },
        ),
        "CPUExecutionProvider",
    ]


def _coreml_nn_opts(static: bool = True):
    return [
        (
            "CoreMLExecutionProvider",
            {
                "ModelFormat": "NeuralNetwork",
                "MLComputeUnits": "ALL",
                "RequireStaticInputShapes": "1" if static else "0",
                "EnableOnSubgraphs": "0",
            },
        ),
        "CPUExecutionProvider",
    ]


def _create_session_with_fallback(
    path: str,
    so: ort.SessionOptions,
    primary_providers,
    fallback_providers=("CPUExecutionProvider",),
    tag: str = "",
) -> InferenceSession:
    try:
        return InferenceSession(path, sess_options=so, providers=list(primary_providers))
    except Exception as e:
        print("*************** EP Error ***************")
        print(
            f"EP Error {type(e).__name__} : {getattr(e, 'args', [''])[0]} "
            f"when using {list(primary_providers)}"
        )
        print(f"Falling back to {list(fallback_providers)} and retrying.")
        print("****************************************")
        return InferenceSession(path, sess_options=so, providers=list(fallback_providers))


def make_encoder_session(
    path: str,
    providers: Optional[Iterable[str]] = None,
) -> InferenceSession:
    path = str(Path(path).resolve())
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    so.intra_op_num_threads = DEFAULT_CPU_THREADS
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    available = ort.get_available_providers()

    if providers is not None:
        attempt_lists = [list(providers)]
    else:
        if CPU_LOW_COST_PROFILE or ACCEL == "cpu":
            attempt_lists = [["CPUExecutionProvider"]]
        elif ACCEL == "cuda" and "CUDAExecutionProvider" in available:
            attempt_lists = [_cuda_providers()]
        elif ACCEL == "coreml" and "CoreMLExecutionProvider" in available:
            attempt_lists = [
                _coreml_mlprogram_opts(static=True),
                _coreml_nn_opts(static=True),
                ["CPUExecutionProvider"],
            ]
        else:
            if "CUDAExecutionProvider" in available:
                attempt_lists = [_cuda_providers()]
            elif "CoreMLExecutionProvider" in available:
                attempt_lists = [["CPUExecutionProvider"]]
            else:
                attempt_lists = [["CPUExecutionProvider"]]

    for primary in attempt_lists:
        print(f"[INFO] Loading {os.path.basename(path)} [encoder] with providers={primary}")
        try:
            sess = InferenceSession(path, sess_options=so, providers=primary)
            print("[INFO] Active providers:", sess.get_providers())
            print("[INFO] Inputs:", [(i.name, i.shape, i.type) for i in sess.get_inputs()])
            print("[INFO] Outputs:", [o.name for o in sess.get_outputs()])
            return sess
        except Exception as e:
            print("*************** EP Error ***************")
            print(f"EP Error {type(e).__name__} : {getattr(e, 'args', [''])[0]} when using {primary}")
            print("****************************************")

    print("[WARN] Encoder: all preferred providers failed; using CPUExecutionProvider.")
    sess = InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
    print("[INFO] Active providers:", sess.get_providers())
    print("[INFO] Inputs:", [(i.name, i.shape, i.type) for i in sess.get_inputs()])
    print("[INFO] Outputs:", [o.name for o in sess.get_outputs()])
    return sess


def make_safe_session(
    path: str,
    providers: Optional[Iterable[str]] = None,
    tag: str = "safe",
) -> InferenceSession:
    path = str(Path(path).resolve())
    basename = os.path.basename(path).lower()
    so = ort.SessionOptions()
    use_static_decoder_opt = (
        STATIC_DECODER_OPT
        and ACCEL != "coreml"
        and basename in _STATIC_SPECIALIZED_DECODERS
    )
    if use_static_decoder_opt:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    else:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.intra_op_num_threads = DEFAULT_CPU_THREADS
    so.inter_op_num_threads = 1
    so.add_session_config_entry("session.disable_gemm_fast_gelu_fusion", "1")

    available = ort.get_available_providers()

    if providers is not None:
        primary = list(providers)
    else:
        if CPU_LOW_COST_PROFILE or ACCEL == "cpu":
            primary = ["CPUExecutionProvider"]
        elif ACCEL == "cuda" and "CUDAExecutionProvider" in available:
            primary = _cuda_providers()
        elif ACCEL == "coreml" and "CoreMLExecutionProvider" in available:
            if COREML_ALL:
                primary = _coreml_mlprogram_opts(static=True)[0:1] + ["CPUExecutionProvider"]
            else:
                if tag in ("decoder", "memory_encoder", "memory_attention", "safe"):
                    primary = ["CPUExecutionProvider"]
                else:
                    primary = _coreml_mlprogram_opts(static=True)[0:1] + ["CPUExecutionProvider"]
        else:
            if "CUDAExecutionProvider" in available:
                primary = _cuda_providers()
            else:
                primary = ["CPUExecutionProvider"]

    opt_mode = "extended(static-decoder)" if use_static_decoder_opt else "disable_all"
    print(f"[INFO] Loading {os.path.basename(path)} [{tag}] with providers={primary} graph_opt={opt_mode}")
    sess = _create_session_with_fallback(path, so, primary, ("CPUExecutionProvider",), tag=tag)
    print("[INFO] Active providers:", sess.get_providers())
    print("[INFO] Inputs:", [(i.name, i.shape, i.type) for i in sess.get_inputs()])
    print("[INFO] Outputs:", [o.name for o in sess.get_outputs()])
    return sess


def _resolve_existing_paths(
    ckpt_dir: str | os.PathLike,
    mapping: Dict[str, str],
) -> Dict[str, Path]:
    ckpt_path = Path(ckpt_dir)
    return {key: (ckpt_path / value) for key, value in mapping.items()}


def _ensure_paths_exist(paths: Dict[str, Path], label: str) -> None:
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {label} artifacts:\n  " + "\n  ".join(missing))


def resolve_image_decoder_path(
    ckpt_dir: str | os.PathLike,
    prompt: str,
    artifacts: str = "auto",
) -> tuple[str, str]:
    if prompt not in ("seed_points", "bounding_box"):
        raise ValueError(f"Unsupported prompt mode: {prompt}")
    if artifacts not in ("auto", "legacy", "specialized"):
        raise ValueError(f"Unsupported artifacts mode: {artifacts}")

    specialized_name = "image_decoder_box.onnx" if prompt == "bounding_box" else "image_decoder_points.onnx"
    legacy_path = Path(ckpt_dir) / "image_decoder.onnx"
    specialized_path = Path(ckpt_dir) / specialized_name
    allow_specialized_points = prompt != "seed_points" or EXPERIMENTAL_IMAGE_POINT_DECODER

    if artifacts == "legacy":
        if not legacy_path.exists():
            raise FileNotFoundError(f"Missing legacy decoder: {legacy_path}")
        return str(legacy_path.resolve()), "legacy"

    if artifacts == "specialized":
        if prompt == "seed_points" and not allow_specialized_points:
            if not legacy_path.exists():
                raise FileNotFoundError(
                    "Specialized image point decoder is currently experimental and disabled by default, "
                    f"and the legacy decoder is missing: {legacy_path}"
                )
            return str(legacy_path.resolve()), "legacy-safe-seed-points"
        if not specialized_path.exists():
            raise FileNotFoundError(f"Missing specialized decoder: {specialized_path}")
        return str(specialized_path.resolve()), "specialized"

    if specialized_path.exists() and allow_specialized_points:
        return str(specialized_path.resolve()), "specialized"
    if legacy_path.exists():
        return str(legacy_path.resolve()), "legacy"
    raise FileNotFoundError(f"Missing decoder artifacts in {ckpt_dir}")


def resolve_video_runtime_paths(
    ckpt_dir: str | os.PathLike,
    artifacts: str = "auto",
) -> Dict[str, str]:
    if artifacts not in ("auto", "legacy", "specialized"):
        raise ValueError(f"Unsupported artifacts mode: {artifacts}")

    legacy = _resolve_existing_paths(
        ckpt_dir,
        {
            "decoder_init": "image_decoder.onnx",
            "decoder_propagate": "image_decoder.onnx",
            "memory_attention": "memory_attention.onnx",
            "memory_encoder": "memory_encoder.onnx",
        },
    )
    specialized_base = _resolve_existing_paths(
        ckpt_dir,
        {
            "decoder_init": "video_decoder_init.onnx",
            "decoder_propagate": "video_decoder_propagate.onnx",
        },
    )
    specialized_attn_objptr = Path(ckpt_dir) / "memory_attention_objptr.onnx"
    specialized_attn_1frame = Path(ckpt_dir) / "memory_attention_no_objptr_1frame.onnx"
    specialized_attn_dynamic = Path(ckpt_dir) / "memory_attention_no_objptr.onnx"
    legacy_memenc = Path(ckpt_dir) / "memory_encoder.onnx"
    specialized_lite_memenc = Path(ckpt_dir) / "memory_encoder_lite.onnx"

    def _hybrid_result() -> Dict[str, str]:
        hybrid_paths = {
            "decoder_init": legacy["decoder_init"],
            "decoder_propagate": specialized_base["decoder_propagate"],
            "memory_attention": legacy["memory_attention"],
            "memory_encoder": legacy["memory_encoder"],
        }
        _ensure_paths_exist(hybrid_paths, "hybrid video")
        return {"mode": "hybrid-propagate", **{k: str(v.resolve()) for k, v in hybrid_paths.items()}}

    def _specialized_result() -> Dict[str, str]:
        _ensure_paths_exist(specialized_base, "specialized core")
        if specialized_attn_objptr.exists():
            memory_attention = specialized_attn_objptr
            mode_suffix = "objptr"
        elif EXPERIMENTAL_1FRAME_ATTN and specialized_attn_1frame.exists():
            memory_attention = specialized_attn_1frame
            mode_suffix = "1frame-attn"
        elif specialized_attn_dynamic.exists():
            memory_attention = specialized_attn_dynamic
            mode_suffix = "dynamic-attn"
        elif specialized_attn_1frame.exists():
            memory_attention = specialized_attn_1frame
            mode_suffix = "1frame-attn-fallback"
        else:
            raise FileNotFoundError(
                "Missing specialized memory attention artifacts:\n"
                f"  {specialized_attn_objptr}\n"
                f"  {specialized_attn_1frame}\n"
                f"  {specialized_attn_dynamic}"
            )
        # Prefer the legacy memory encoder when present because it exports
        # temporal_code, which lets the runtime rebuild SAM2's temporal memory positions.
        memenc_path = legacy_memenc if legacy_memenc.exists() else specialized_lite_memenc
        if not memenc_path.exists():
            raise FileNotFoundError(f"Missing specialized memory encoder fallback: {memenc_path}")
        mode = "specialized-temporal" if memenc_path == legacy_memenc else "specialized-lite"
        mode = f"{mode}-{mode_suffix}"
        return {
            "mode": mode,
            **{k: str(v.resolve()) for k, v in specialized_base.items()},
            "memory_attention": str(memory_attention.resolve()),
            "memory_encoder": str(memenc_path.resolve()),
        }

    legacy_available = all(path.exists() for path in legacy.values())
    hybrid_available = legacy_available and specialized_base["decoder_propagate"].exists()
    specialized_available = all(path.exists() for path in specialized_base.values()) and (
        specialized_attn_objptr.exists() or specialized_attn_1frame.exists() or specialized_attn_dynamic.exists()
    )

    def _preferred_optimized_result() -> Dict[str, str]:
        # Fixed-slot prompt init decoders change SAM prompt semantics and currently
        # poison the first-frame memory state, so the safe optimized path keeps the
        # legacy promptable init decoder and only specializes prompt-free propagation.
        if not EXPERIMENTAL_VIDEO_INIT_DECODER and hybrid_available:
            return _hybrid_result()
        if specialized_available:
            return _specialized_result()
        if hybrid_available:
            return _hybrid_result()
        raise FileNotFoundError("Missing optimized video artifacts")

    if artifacts == "legacy":
        _ensure_paths_exist(legacy, "legacy")
        return {"mode": "legacy", **{k: str(v.resolve()) for k, v in legacy.items()}}

    if artifacts == "specialized":
        return _preferred_optimized_result()

    prefer_specialized = VIDEO_AUTO_POLICY in ("speed", "specialized")

    if prefer_specialized and (hybrid_available or specialized_available):
        return _preferred_optimized_result()
    if legacy_available:
        return {"mode": "legacy", **{k: str(v.resolve()) for k, v in legacy.items()}}
    if hybrid_available or specialized_available:
        return _preferred_optimized_result()

    _ensure_paths_exist(legacy, "legacy")
    return {"mode": "legacy", **{k: str(v.resolve()) for k, v in legacy.items()}}


def prefer_quantized_encoder(
    ckpt_dir: str | os.PathLike,
    base_name: str = "image_encoder",
) -> Optional[str]:
    """
    Pick the best encoder artifact for the current acceleration mode.
    - If accelerating (CUDA or explicit CoreML), prefer float .onnx over int8.
    - If CPU, prefer int8 when available.
    """
    available = ort.get_available_providers()

    accel = False
    if CPU_LOW_COST_PROFILE:
        accel = False
    elif ACCEL == "cuda" and "CUDAExecutionProvider" in available:
        accel = True
    elif ACCEL == "coreml" and "CoreMLExecutionProvider" in available:
        accel = True
    elif ACCEL == "auto" and "CUDAExecutionProvider" in available:
        accel = True

    if ENCODER_VARIANT == "fp32":
        order = [f"{base_name}.onnx", f"{base_name}.int8.onnx"]
    elif ENCODER_VARIANT == "int8":
        order = [f"{base_name}.int8.onnx", f"{base_name}.onnx"]
    elif accel:
        order = [f"{base_name}.onnx", f"{base_name}.int8.onnx"]
    else:
        order = [f"{base_name}.int8.onnx", f"{base_name}.onnx"]

    ckpt_path = Path(ckpt_dir)
    for fname in order:
        candidate = ckpt_path / fname
        if candidate.exists():
            return str(candidate.resolve())
    return None


def as_f32c(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32, copy=False)
    return np.ascontiguousarray(a)


_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def bgr_to_input_tensor(img_bgr: np.ndarray, enc_hw: Tuple[int, int]) -> np.ndarray:
    h_enc, w_enc = enc_hw
    img_resized = cv2.resize(img_bgr, (w_enc, h_enc))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_rgb = (img_rgb - _MEAN) / _STD
    tensor = np.transpose(img_rgb, (2, 0, 1))[np.newaxis, :]
    return as_f32c(tensor)


def prepare_image(
    img_bgr: np.ndarray,
    enc_hw: Tuple[int, int],
) -> Tuple[np.ndarray, Tuple[int, int]]:
    h_org, w_org = img_bgr.shape[:2]
    return bgr_to_input_tensor(img_bgr, enc_hw), (h_org, w_org)


def compute_display_base(
    img_bgr: np.ndarray,
    max_side: int = 1200,
) -> Tuple[np.ndarray, float]:
    h, w = img_bgr.shape[:2]
    scale = min(1.0, max_side / max(w, h))
    disp = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    return disp, scale


def prepare_points(
    points: Iterable[Tuple[int, int]],
    labels: Iterable[int | float],
    img_size: Tuple[int, int],
    enc_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    points = list(points)
    if not points:
        return (np.zeros((1, 0, 2), np.float32), np.zeros((1, 0), np.float32))
    pts = np.asarray(points, dtype=np.float32)
    lbl = np.asarray(list(labels), dtype=np.float32)
    h_org, w_org = img_size
    h_enc, w_enc = enc_size
    pts[:, 0] = (pts[:, 0] / w_org) * w_enc
    pts[:, 1] = (pts[:, 1] / h_org) * h_enc
    return pts[np.newaxis, ...], lbl[np.newaxis, ...]


def prepare_box_prompt(
    rect: Optional[Tuple[int, int, int, int]],
    img_size: Tuple[int, int],
    enc_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    if rect is None:
        return (np.zeros((1, 0, 2), np.float32), np.zeros((1, 0), np.float32))
    x1, y1, x2, y2 = rect
    h_org, w_org = img_size
    h_enc, w_enc = enc_size
    pts = np.array([[x1, y1], [x2, y2]], np.float32)
    pts[:, 0] = (pts[:, 0] / w_org) * w_enc
    pts[:, 1] = (pts[:, 1] / h_org) * h_enc
    lbl = np.array([2.0, 3.0], np.float32)
    return pts[np.newaxis, ...], lbl[np.newaxis, ...]


def prepare_rectangle(rect, img_size, enc_size):
    return prepare_box_prompt(rect, img_size, enc_size)


def run_encoder(sess_enc: InferenceSession, input_tensor: np.ndarray) -> Dict[str, np.ndarray]:
    enc_input_name = sess_enc.get_inputs()[0].name
    out_names = [o.name for o in sess_enc.get_outputs()]
    values = sess_enc.run(None, {enc_input_name: as_f32c(input_tensor)})
    return dict(zip(out_names, values))


def _find_session_name(names: Iterable[str], key: str) -> Optional[str]:
    for name in names:
        if key in name:
            return name
    return None


def _normalize_prompt_inputs(
    sess_dec: InferenceSession,
    point_coords: Optional[np.ndarray],
    point_labels: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    if point_coords is None or point_labels is None:
        point_coords = np.zeros((1, 0, 2), np.float32)
        point_labels = np.zeros((1, 0), np.float32)

    point_coords = as_f32c(point_coords)
    point_labels = as_f32c(point_labels)

    point_input = _find_session_name((inp.name for inp in sess_dec.get_inputs()), "point_coords")
    if point_input is None:
        return point_coords, point_labels

    shape = next(inp.shape for inp in sess_dec.get_inputs() if inp.name == point_input)
    max_points = None
    if len(shape) >= 2 and isinstance(shape[1], int) and shape[1] > 0:
        max_points = int(shape[1])

    if max_points is None:
        return point_coords, point_labels

    current_points = int(point_coords.shape[1])
    if current_points > max_points:
        print(f"[WARN] Truncating prompts from {current_points} to {max_points} for fixed-shape decoder.")
        point_coords = point_coords[:, :max_points, :]
        point_labels = point_labels[:, :max_points]
        current_points = max_points

    if current_points < max_points:
        pad_points = np.zeros((point_coords.shape[0], max_points - current_points, 2), np.float32)
        pad_labels = -np.ones((point_labels.shape[0], max_points - current_points), np.float32)
        point_coords = np.concatenate([point_coords, pad_points], axis=1)
        point_labels = np.concatenate([point_labels, pad_labels], axis=1)

    return point_coords, point_labels


def run_decoder(
    sess_dec: InferenceSession,
    point_coords: Optional[np.ndarray],
    point_labels: Optional[np.ndarray],
    image_embed: np.ndarray,
    high_res_feats_0: np.ndarray,
    high_res_feats_1: np.ndarray,
):
    input_names = [inp.name for inp in sess_dec.get_inputs()]
    point_coords, point_labels = _normalize_prompt_inputs(sess_dec, point_coords, point_labels)

    feed = {}
    name = _find_session_name(input_names, "point_coords")
    if name is not None:
        feed[name] = point_coords
    name = _find_session_name(input_names, "point_labels")
    if name is not None:
        feed[name] = point_labels
    name = _find_session_name(input_names, "image_embed")
    if name is not None:
        feed[name] = as_f32c(image_embed)
    name = _find_session_name(input_names, "high_res_feats_0")
    if name is not None:
        feed[name] = as_f32c(high_res_feats_0)
    name = _find_session_name(input_names, "high_res_feats_1")
    if name is not None:
        feed[name] = as_f32c(high_res_feats_1)

    values = sess_dec.run(None, feed)
    output_names = [out.name for out in sess_dec.get_outputs()]
    outputs = dict(zip(output_names, values))

    obj_ptr = None
    mask_for_mem = None
    pred_mask = None

    name = _find_session_name(output_names, "obj_ptr")
    if name is not None:
        obj_ptr = outputs[name]
    name = _find_session_name(output_names, "mask_for_mem")
    if name is not None:
        mask_for_mem = outputs[name]
    name = _find_session_name(output_names, "pred_mask")
    if name is not None:
        pred_mask = outputs[name]

    if pred_mask is None:
        if len(values) == 1:
            pred_mask = values[0]
        elif len(values) == 2 and mask_for_mem is not None:
            pred_mask = values[1]
        elif len(values) >= 3:
            pred_mask = values[-1]

    if mask_for_mem is None and len(values) == 2 and pred_mask is values[1]:
        mask_for_mem = values[0]

    return obj_ptr, mask_for_mem, pred_mask


def run_memory_attention(
    sess_mat: InferenceSession,
    current_vision_feat: np.ndarray,
    current_vision_pos_embed: np.ndarray,
    memory_1: np.ndarray,
    memory_pos_embed: np.ndarray,
    memory_0: Optional[np.ndarray] = None,
    obj_ptr_offsets: Optional[np.ndarray] = None,
) -> np.ndarray:
    input_names = [inp.name for inp in sess_mat.get_inputs()]
    feed = {}

    name = _find_session_name(input_names, "current_vision_feat")
    if name is not None:
        feed[name] = as_f32c(current_vision_feat)
    name = _find_session_name(input_names, "current_vision_pos_embed")
    if name is not None:
        feed[name] = as_f32c(current_vision_pos_embed)
    name = _find_session_name(input_names, "memory_1")
    if name is not None:
        feed[name] = as_f32c(memory_1)
    name = _find_session_name(input_names, "memory_pos_embed")
    if name is not None:
        feed[name] = as_f32c(memory_pos_embed)
    name = _find_session_name(input_names, "memory_0")
    if name is not None:
        if memory_0 is None:
            memory_0 = np.zeros((0, 256), np.float32)
        feed[name] = as_f32c(memory_0)
    name = _find_session_name(input_names, "obj_ptr_offsets")
    if name is not None:
        if obj_ptr_offsets is None:
            obj_ptr_offsets = np.zeros((0,), np.float32)
        feed[name] = as_f32c(obj_ptr_offsets)

    return sess_mat.run(None, feed)[0]


def run_memory_encoder(
    sess_men: InferenceSession,
    mask_for_mem: np.ndarray,
    pix_feat: np.ndarray,
):
    input_names = [inp.name for inp in sess_men.get_inputs()]
    feed = {}

    name = _find_session_name(input_names, "mask_for_mem")
    if name is not None:
        feed[name] = as_f32c(mask_for_mem)
    name = _find_session_name(input_names, "pix_feat")
    if name is not None:
        feed[name] = as_f32c(pix_feat)

    values = sess_men.run(None, feed)
    output_names = [out.name for out in sess_men.get_outputs()]
    outputs = dict(zip(output_names, values))

    maskmem_features = outputs.get(_find_session_name(output_names, "maskmem_features"))
    maskmem_pos_enc = outputs.get(_find_session_name(output_names, "maskmem_pos_enc"))
    temporal_code = outputs.get(_find_session_name(output_names, "temporal_code"))

    if maskmem_features is None and len(values) >= 1:
        maskmem_features = values[0]
    if maskmem_pos_enc is None and len(values) >= 2:
        maskmem_pos_enc = values[1]
    if temporal_code is None and len(values) >= 3:
        temporal_code = values[2]

    return maskmem_features, maskmem_pos_enc, temporal_code


def _memory_attention_single_frame_only(sess_mat: InferenceSession) -> bool:
    for inp in sess_mat.get_inputs():
        if "memory_1" not in inp.name.lower():
            continue
        shape = inp.shape
        if not shape:
            return False
        return shape[0] == 1
    return False


def _memory_attention_uses_object_pointers(sess_mat: InferenceSession) -> bool:
    input_names = [inp.name.lower() for inp in sess_mat.get_inputs()]
    return any("obj_ptr_offsets" in name for name in input_names)


def _normalize_temporal_code(temporal_code: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if temporal_code is None:
        return None
    code = as_f32c(temporal_code)
    if code.ndim == 4:
        return code.reshape(code.shape[0], 1, code.shape[-1])
    if code.ndim == 3:
        return code.reshape(code.shape[0], 1, code.shape[-1])
    if code.ndim == 2:
        return code.reshape(code.shape[0], 1, code.shape[1])
    return None


@dataclass
class VideoMemoryFrame:
    features: np.ndarray
    pos: np.ndarray
    temporal_code: Optional[np.ndarray]
    obj_ptr: Optional[np.ndarray]
    frame_index: int


@dataclass
class VideoMemoryBank:
    max_slots: int = DEFAULT_VIDEO_MEMORY_SLOTS
    max_pointer_slots: int = DEFAULT_VIDEO_OBJECT_POINTER_SLOTS
    single_frame_only: bool = False
    use_object_pointers: bool = False
    conditioning: List[VideoMemoryFrame] = field(default_factory=list)
    recent: List[VideoMemoryFrame] = field(default_factory=list)

    @classmethod
    def from_session(cls, sess_mat: InferenceSession) -> "VideoMemoryBank":
        return cls(
            single_frame_only=_memory_attention_single_frame_only(sess_mat),
            use_object_pointers=_memory_attention_uses_object_pointers(sess_mat),
        )

    def _update_capacity(self, temporal_code: Optional[np.ndarray]) -> Optional[np.ndarray]:
        normalized = _normalize_temporal_code(temporal_code)
        if normalized is not None and normalized.shape[0] > 0:
            self.max_slots = min(self.max_slots, int(normalized.shape[0]))
        return normalized

    def _make_frame(
        self,
        features: np.ndarray,
        pos: np.ndarray,
        temporal_code: Optional[np.ndarray],
        obj_ptr: Optional[np.ndarray],
        frame_index: int,
    ) -> VideoMemoryFrame:
        normalized_code = self._update_capacity(temporal_code)
        normalized_obj_ptr = None
        if obj_ptr is not None:
            normalized_obj_ptr = as_f32c(obj_ptr)
            if normalized_obj_ptr.ndim == 1:
                normalized_obj_ptr = normalized_obj_ptr.reshape(1, -1)
        return VideoMemoryFrame(
            features=as_f32c(features),
            pos=as_f32c(pos),
            temporal_code=normalized_code,
            obj_ptr=normalized_obj_ptr,
            frame_index=frame_index,
        )

    def _trim_recent(self) -> None:
        if self.single_frame_only:
            max_recent = 1
        else:
            max_recent = self.max_slots - len(self.conditioning)
        max_recent = max(0, max_recent)
        if len(self.recent) > max_recent:
            del self.recent[max_recent:]

    def set_conditioning(
        self,
        features: np.ndarray,
        pos: np.ndarray,
        temporal_code: Optional[np.ndarray],
        obj_ptr: Optional[np.ndarray] = None,
        frame_index: int = 0,
    ) -> None:
        self.conditioning = [self._make_frame(features, pos, temporal_code, obj_ptr, frame_index)]
        self._trim_recent()

    def append_recent(
        self,
        features: np.ndarray,
        pos: np.ndarray,
        temporal_code: Optional[np.ndarray],
        obj_ptr: Optional[np.ndarray] = None,
        frame_index: int = 0,
    ) -> None:
        self.recent.insert(0, self._make_frame(features, pos, temporal_code, obj_ptr, frame_index))
        self._trim_recent()

    def build_attention_state(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        entries: list[tuple[int, VideoMemoryFrame]] = []
        if self.single_frame_only:
            if self.recent:
                entries.append((1, self.recent[0]))
            elif self.conditioning:
                entries.append((0, self.conditioning[0]))
        else:
            entries.extend((0, frame) for frame in self.conditioning)
            max_recent = max(0, self.max_slots - len(self.conditioning))
            entries.extend((idx + 1, frame) for idx, frame in enumerate(self.recent[:max_recent]))

        if not entries:
            return None, None

        memory_1 = np.concatenate([frame.features for _, frame in entries], axis=0).astype(np.float32, copy=False)
        pos_parts = []
        for t_pos, frame in entries:
            pos = frame.pos
            if frame.temporal_code is not None and frame.temporal_code.shape[0] > t_pos:
                temporal_index = frame.temporal_code.shape[0] - t_pos - 1
                pos = pos + frame.temporal_code[temporal_index : temporal_index + 1]
            pos_parts.append(pos.astype(np.float32, copy=False))

        memory_pos_embed = np.concatenate(pos_parts, axis=0).astype(np.float32, copy=False)
        return memory_1, memory_pos_embed

    def build_object_pointer_state(self, frame_index: int) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not self.use_object_pointers:
            return None, None

        entries: List[VideoMemoryFrame] = []
        for frame in self.conditioning:
            if frame.obj_ptr is not None:
                entries.append(frame)

        max_recent = max(0, self.max_pointer_slots - len(entries))
        for frame in self.recent[:max_recent]:
            if frame.obj_ptr is not None:
                entries.append(frame)

        if not entries:
            return None, None

        object_ptrs = np.concatenate([frame.obj_ptr for frame in entries if frame.obj_ptr is not None], axis=0)
        offsets = np.asarray(
            [max(0, frame_index - frame.frame_index) for frame in entries],
            dtype=np.float32,
        )
        return object_ptrs.astype(np.float32, copy=False), offsets


def warmup_video_runtime_sessions(
    sess_dec0: InferenceSession,
    sess_decn: InferenceSession,
    sess_mat: InferenceSession,
    sess_men: InferenceSession,
    first_record: Dict[str, np.ndarray],
    prompt_inputs: tuple[Optional[np.ndarray], Optional[np.ndarray]],
    repeats: int = 1,
    propagate_record: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    if repeats <= 0:
        return

    pts0, lbls0 = prompt_inputs
    propagate_record = first_record if propagate_record is None else propagate_record

    for _ in range(repeats):
        memory_bank = VideoMemoryBank.from_session(sess_mat)
        enc_embed0 = as_f32c(first_record["embed"])
        f0_0 = as_f32c(first_record["f0"])
        f1_0 = as_f32c(first_record["f1"])

        obj_ptr0, mask_for_mem0, _ = run_decoder(sess_dec0, pts0, lbls0, enc_embed0, f0_0, f1_0)
        if mask_for_mem0 is None:
            raise RuntimeError("Warmup failed: decoder init did not return mask_for_mem")

        mem_feats, mem_pos, temporal_code = run_memory_encoder(
            sess_men,
            mask_for_mem=mask_for_mem0[:, 0:1],
            pix_feat=enc_embed0,
        )
        memory_bank.set_conditioning(mem_feats, mem_pos, temporal_code, obj_ptr=obj_ptr0, frame_index=0)

        prop_embed = as_f32c(propagate_record["embed"])
        prop_curr_feat = as_f32c(propagate_record.get("curr_feat", propagate_record["embed"]))
        prop_f0 = as_f32c(propagate_record["f0"])
        prop_f1 = as_f32c(propagate_record["f1"])
        prop_vis_pos = propagate_record.get("vis_pos")
        if prop_vis_pos is not None:
            prop_vis_pos = as_f32c(prop_vis_pos)
        memory_1, memory_pos_embed = memory_bank.build_attention_state()
        object_ptrs, obj_ptr_offsets = memory_bank.build_object_pointer_state(frame_index=1)
        if memory_1 is None or memory_pos_embed is None:
            raise RuntimeError("Warmup failed: no memory bank available for propagation")

        fused_embed = run_memory_attention(
            sess_mat,
            current_vision_feat=prop_curr_feat,
            current_vision_pos_embed=prop_vis_pos,
            memory_1=memory_1,
            memory_pos_embed=memory_pos_embed,
            memory_0=object_ptrs,
            obj_ptr_offsets=obj_ptr_offsets,
        )
        fused_embed = as_f32c(fused_embed)

        obj_ptr_n, mask_for_mem_n, _ = run_decoder(sess_decn, None, None, fused_embed, prop_f0, prop_f1)
        if mask_for_mem_n is None:
            raise RuntimeError("Warmup failed: decoder propagate did not return mask_for_mem")

        mem_feats_n, mem_pos_n, temporal_code_n = run_memory_encoder(
            sess_men,
            mask_for_mem=mask_for_mem_n[:, 0:1],
            pix_feat=fused_embed,
        )
        memory_bank.append_recent(mem_feats_n, mem_pos_n, temporal_code_n, obj_ptr=obj_ptr_n, frame_index=1)


def green_overlay(
    bgr: np.ndarray,
    mask255: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    fg = mask255 > 0
    color = np.zeros_like(bgr)
    color[fg] = (0, 255, 0)
    return cv2.addWeighted(bgr, 1.0, color, alpha, 0)
