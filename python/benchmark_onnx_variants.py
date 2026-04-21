#!/usr/bin/env python3
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import statistics
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

make_encoder_session = None
make_safe_session = None
prefer_quantized_encoder = None
prepare_box_prompt = None
prepare_image = None
prepare_points = None
print_system_info = None
resolve_image_decoder_path = None
resolve_video_runtime_paths = None
run_decoder = None
run_encoder = None
run_memory_attention = None
run_memory_encoder = None
set_cv2_threads = None
VideoMemoryBank = None
warmup_video_runtime_sessions = None

_SAM2_MODEL_VARIANTS = {
    "base_plus": {
        "config": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "checkpoint": "checkpoints/sam2.1_hiera_base_plus.pt",
    },
    "large": {
        "config": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "checkpoint": "checkpoints/sam2.1_hiera_large.pt",
    },
    "small": {
        "config": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "checkpoint": "checkpoints/sam2.1_hiera_small.pt",
    },
    "tiny": {
        "config": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "checkpoint": "checkpoints/sam2.1_hiera_tiny.pt",
    },
}


def _load_runtime_helpers():
    global make_encoder_session
    global make_safe_session
    global prefer_quantized_encoder
    global prepare_box_prompt
    global prepare_image
    global prepare_points
    global print_system_info
    global resolve_image_decoder_path
    global resolve_video_runtime_paths
    global run_decoder
    global run_encoder
    global run_memory_attention
    global run_memory_encoder
    global set_cv2_threads
    global VideoMemoryBank
    global warmup_video_runtime_sessions

    from onnx_test_utils import (
        make_encoder_session as _make_encoder_session,
        make_safe_session as _make_safe_session,
        prefer_quantized_encoder as _prefer_quantized_encoder,
        prepare_box_prompt as _prepare_box_prompt,
        prepare_image as _prepare_image,
        prepare_points as _prepare_points,
        print_system_info as _print_system_info,
        resolve_image_decoder_path as _resolve_image_decoder_path,
        resolve_video_runtime_paths as _resolve_video_runtime_paths,
        run_decoder as _run_decoder,
        run_encoder as _run_encoder,
        run_memory_attention as _run_memory_attention,
        run_memory_encoder as _run_memory_encoder,
        set_cv2_threads as _set_cv2_threads,
        VideoMemoryBank as _VideoMemoryBank,
        warmup_video_runtime_sessions as _warmup_video_runtime_sessions,
    )

    make_encoder_session = _make_encoder_session
    make_safe_session = _make_safe_session
    prefer_quantized_encoder = _prefer_quantized_encoder
    prepare_box_prompt = _prepare_box_prompt
    prepare_image = _prepare_image
    prepare_points = _prepare_points
    print_system_info = _print_system_info
    resolve_image_decoder_path = _resolve_image_decoder_path
    resolve_video_runtime_paths = _resolve_video_runtime_paths
    run_decoder = _run_decoder
    run_encoder = _run_encoder
    run_memory_attention = _run_memory_attention
    run_memory_encoder = _run_memory_encoder
    set_cv2_threads = _set_cv2_threads
    VideoMemoryBank = _VideoMemoryBank
    warmup_video_runtime_sessions = _warmup_video_runtime_sessions


def _mean_ms(values):
    return statistics.mean(values) if values else 0.0


def _runtime_label(mode_key: str, resolved_mode: str | None = None) -> str:
    if resolved_mode:
        return resolved_mode
    if mode_key == "specialized":
        return "optimized"
    return mode_key


def _prompt_spec(prompt_mode: str, image_shape: tuple[int, int]):
    h_org, w_org = image_shape
    if prompt_mode == "bounding_box":
        return {
            "type": "box",
            "box": (
                int(w_org * 0.2),
                int(h_org * 0.2),
                int(w_org * 0.8),
                int(h_org * 0.8),
            ),
        }

    return {
        "type": "points",
        "points": [
            (int(w_org * 0.5), int(h_org * 0.5)),
            (int(w_org * 0.25), int(h_org * 0.25)),
        ],
        "labels": [1, 0],
    }


def _prompt_inputs(prompt_mode: str, image_shape: tuple[int, int], enc_shape: tuple[int, int]):
    prompt_spec = _prompt_spec(prompt_mode, image_shape)
    if prompt_spec["type"] == "box":
        return prepare_box_prompt(prompt_spec["box"], image_shape, enc_shape)
    return prepare_points(prompt_spec["points"], prompt_spec["labels"], image_shape, enc_shape)


def _binary_iou(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs_bin = lhs > 0
    rhs_bin = rhs > 0
    union = np.logical_or(lhs_bin, rhs_bin).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(lhs_bin, rhs_bin).sum()
    return float(intersection / union)


def _compare_logits(ref: np.ndarray, other: np.ndarray) -> dict[str, float]:
    diff = np.abs(ref - other)
    return {
        "mean_abs": float(diff.mean()),
        "max_abs": float(diff.max()),
        "binary_iou": _binary_iou(ref, other),
    }


def _load_video_frames(video_path: str, max_frames: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if max_frames > 0 and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        raise RuntimeError(f"Video is empty: {video_path}")
    return frames


def _benchmark_encoder(sess_enc, image_bgr: np.ndarray, repeats: int, warmup: int):
    enc_h, enc_w = sess_enc.get_inputs()[0].shape[2:]
    tensor, _ = prepare_image(image_bgr, (enc_h, enc_w))

    for _ in range(warmup):
        run_encoder(sess_enc, tensor)

    times = []
    last = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        last = run_encoder(sess_enc, tensor)
        times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "times_ms": times,
        "enc_dict": last,
        "enc_shape": (enc_h, enc_w),
    }


def _benchmark_image_decoder(sess_dec, enc_dict, prompt_inputs, repeats: int, warmup: int):
    pts, lbls = prompt_inputs
    image_embed = enc_dict["image_embeddings"].astype(np.float32, copy=False)
    feats0 = enc_dict["high_res_features1"].astype(np.float32, copy=False)
    feats1 = enc_dict["high_res_features2"].astype(np.float32, copy=False)

    for _ in range(warmup):
        run_decoder(sess_dec, pts, lbls, image_embed, feats0, feats1)

    times = []
    pred_mask = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        _, _, pred_mask = run_decoder(sess_dec, pts, lbls, image_embed, feats0, feats1)
        times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "times_ms": times,
        "pred_mask": pred_mask[0, 0].astype(np.float32, copy=False),
    }


def _precompute_video_encoder(sess_enc, frames: list[np.ndarray], warmup: int = 0):
    enc_input_name = sess_enc.get_inputs()[0].name
    enc_out_names = [o.name for o in sess_enc.get_outputs()]
    enc_h, enc_w = sess_enc.get_inputs()[0].shape[2:]

    if warmup > 0 and frames:
        tensor0, _ = prepare_image(frames[0], (enc_h, enc_w))
        for _ in range(warmup):
            sess_enc.run(None, {enc_input_name: tensor0})

    records = []
    enc_times = []
    for frame in frames:
        tensor, _ = prepare_image(frame, (enc_h, enc_w))
        t0 = time.perf_counter()
        enc_vals = sess_enc.run(None, {enc_input_name: tensor})
        enc_times.append((time.perf_counter() - t0) * 1000.0)
        enc = dict(zip(enc_out_names, enc_vals))
        curr_feat = enc.get("current_vision_feat", enc["image_embeddings"])
        records.append(
            {
                "embed": enc["image_embeddings"].astype(np.float32, copy=False),
                "curr_feat": curr_feat.astype(np.float32, copy=False),
                "f0": enc["high_res_features1"].astype(np.float32, copy=False),
                "f1": enc["high_res_features2"].astype(np.float32, copy=False),
                "vis_pos": enc.get("vision_pos_embed"),
            }
        )
        if records[-1]["vis_pos"] is not None:
            records[-1]["vis_pos"] = records[-1]["vis_pos"].astype(np.float32, copy=False)
    return {
        "times_ms": enc_times,
        "records": records,
        "enc_shape": (enc_h, enc_w),
    }


def _benchmark_video_variant(paths: dict[str, str], encoded_frames, prompt_inputs, session_warmup: int = 0):
    sess_dec0 = make_safe_session(paths["decoder_init"], tag="decoder")
    if paths["decoder_propagate"] == paths["decoder_init"]:
        sess_decn = sess_dec0
    else:
        sess_decn = make_safe_session(paths["decoder_propagate"], tag="decoder")
    sess_mat = make_safe_session(paths["memory_attention"], tag="memory_attention")
    sess_men = make_safe_session(paths["memory_encoder"], tag="memory_encoder")

    if encoded_frames and session_warmup > 0:
        warmup_video_runtime_sessions(
            sess_dec0=sess_dec0,
            sess_decn=sess_decn,
            sess_mat=sess_mat,
            sess_men=sess_men,
            first_record=encoded_frames[0],
            propagate_record=encoded_frames[1] if len(encoded_frames) > 1 else encoded_frames[0],
            prompt_inputs=prompt_inputs,
            repeats=session_warmup,
        )

    pts0, lbls0 = prompt_inputs
    memory_bank = VideoMemoryBank.from_session(sess_mat)
    print(
        f"[VIDEO] Memory caps frames={memory_bank.max_slots} "
        f"object_ptrs={memory_bank.max_pointer_slots}"
    )
    frame0 = {
        "attn_ms": [],
        "dec_ms": [],
        "men_ms": [],
        "total_ms": [],
    }
    propagate = {
        "attn_ms": [],
        "dec_ms": [],
        "men_ms": [],
        "total_ms": [],
    }
    pred_masks = []

    for idx, record in enumerate(encoded_frames):
        enc_embed = record["embed"]
        curr_feat = record.get("curr_feat", enc_embed)
        f0 = record["f0"]
        f1 = record["f1"]
        vis_pos = record["vis_pos"]

        if idx > 0:
            memory_1, memory_pos_embed = memory_bank.build_attention_state()
            object_ptrs, obj_ptr_offsets = memory_bank.build_object_pointer_state(frame_index=idx)
        else:
            memory_1, memory_pos_embed = (None, None)
            object_ptrs, obj_ptr_offsets = (None, None)

        if idx > 0 and memory_1 is not None and memory_pos_embed is not None:
            t0 = time.perf_counter()
            fused_embed = run_memory_attention(
                sess_mat,
                current_vision_feat=curr_feat,
                current_vision_pos_embed=vis_pos,
                memory_1=memory_1,
                memory_pos_embed=memory_pos_embed,
                memory_0=object_ptrs,
                obj_ptr_offsets=obj_ptr_offsets,
            ).astype(np.float32, copy=False)
            attn_elapsed_ms = (time.perf_counter() - t0) * 1000.0
        else:
            fused_embed = enc_embed
            attn_elapsed_ms = 0.0

        t0 = time.perf_counter()
        if idx == 0:
            obj_ptr, mask_for_mem, pred_mask = run_decoder(sess_dec0, pts0, lbls0, fused_embed, f0, f1)
        else:
            obj_ptr, mask_for_mem, pred_mask = run_decoder(sess_decn, None, None, fused_embed, f0, f1)
        dec_elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if mask_for_mem is None or pred_mask is None:
            raise RuntimeError("Decoder did not return mask_for_mem and pred_mask")

        t0 = time.perf_counter()
        mem_feats, mem_pos, temporal_code = run_memory_encoder(
            sess_men,
            mask_for_mem=mask_for_mem[:, 0:1],
            pix_feat=fused_embed,
        )
        men_elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if idx == 0:
            memory_bank.set_conditioning(mem_feats, mem_pos, temporal_code, obj_ptr=obj_ptr, frame_index=idx)
        else:
            memory_bank.append_recent(mem_feats, mem_pos, temporal_code, obj_ptr=obj_ptr, frame_index=idx)
        pred_masks.append(pred_mask[0, 0].astype(np.float32, copy=False))

        bucket = frame0 if idx == 0 else propagate
        bucket["attn_ms"].append(attn_elapsed_ms)
        bucket["dec_ms"].append(dec_elapsed_ms)
        bucket["men_ms"].append(men_elapsed_ms)
        bucket["total_ms"].append(attn_elapsed_ms + dec_elapsed_ms + men_elapsed_ms)

    attn_ms = frame0["attn_ms"] + propagate["attn_ms"]
    dec_ms = frame0["dec_ms"] + propagate["dec_ms"]
    men_ms = frame0["men_ms"] + propagate["men_ms"]

    return {
        "attn_ms": attn_ms,
        "dec_ms": dec_ms,
        "men_ms": men_ms,
        "frame0": frame0,
        "propagate": propagate,
        "pred_masks": pred_masks,
    }


def _compare_video_masks(ref_masks: list[np.ndarray], other_masks: list[np.ndarray]) -> dict[str, float]:
    mean_abs = []
    max_abs = []
    binary_iou = []
    for ref, other in zip(ref_masks, other_masks):
        delta = _compare_logits(ref, other)
        mean_abs.append(delta["mean_abs"])
        max_abs.append(delta["max_abs"])
        binary_iou.append(delta["binary_iou"])
    return {
        "mean_abs": statistics.mean(mean_abs) if mean_abs else 0.0,
        "max_abs": max(max_abs) if max_abs else 0.0,
        "binary_iou": statistics.mean(binary_iou) if binary_iou else 0.0,
    }


def _median_of_video_runs(runs, key: str) -> float:
    values = [_mean_ms(run[key]) for run in runs]
    return statistics.median(values) if values else 0.0


def _median_of_video_section(runs, section: str, key: str) -> float:
    values = [_mean_ms(run[section][key]) for run in runs if run[section][key]]
    return statistics.median(values) if values else 0.0


def _video_section_count(runs, section: str, key: str) -> int:
    for run in runs:
        if run[section][key]:
            return len(run[section][key])
    return 0


def _resolve_native_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _prepare_native_video_frames(repo_root: Path, frames: list[np.ndarray]) -> tempfile.TemporaryDirectory:
    temp_dir = tempfile.TemporaryDirectory(prefix="sam2_native_frames_", dir=str(repo_root))
    frame_dir = Path(temp_dir.name)
    for idx, frame in enumerate(frames):
        out_path = frame_dir / f"{idx:05d}.jpg"
        if not cv2.imwrite(str(out_path), frame):
            temp_dir.cleanup()
            raise RuntimeError(f"Failed to write temporary frame: {out_path}")
    return temp_dir


def _binary_iou_from_masks(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs_bin = lhs.astype(bool, copy=False)
    rhs_bin = rhs.astype(bool, copy=False)
    union = np.logical_or(lhs_bin, rhs_bin).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(lhs_bin, rhs_bin).sum()
    return float(intersection / union)


def _compare_native_masks(onnx_masks: list[np.ndarray], native_masks: list[np.ndarray]) -> dict[str, float]:
    binary_iou = []
    area_ratio = []
    zero_frames = 0
    for onnx_mask, native_mask in zip(onnx_masks, native_masks):
        onnx_bin = np.squeeze(onnx_mask) > 0
        native_mask_2d = np.squeeze(native_mask)
        if onnx_bin.ndim != 2 or native_mask_2d.ndim != 2:
            continue
        native_resized = cv2.resize(
            native_mask_2d.astype(np.uint8),
            (onnx_bin.shape[1], onnx_bin.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        binary_iou.append(_binary_iou_from_masks(onnx_bin, native_resized))
        native_area = float(native_resized.sum())
        onnx_area = float(onnx_bin.sum())
        if native_area > 0.0:
            area_ratio.append(onnx_area / native_area)
        if onnx_area == 0.0:
            zero_frames += 1
    return {
        "binary_iou": statistics.mean(binary_iou) if binary_iou else 0.0,
        "area_ratio": statistics.mean(area_ratio) if area_ratio else 0.0,
        "zero_frames": float(zero_frames),
    }


def _benchmark_native_video(
    repo_root: Path,
    model_size: str,
    prompt_mode: str,
    frames: list[np.ndarray],
    native_device: str,
):
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    try:
        import torch
        from sam2.build_sam import build_sam2_video_predictor
    except Exception as exc:
        raise RuntimeError(f"Native SAM2 import failed: {exc}") from exc

    variant = _SAM2_MODEL_VARIANTS[model_size]
    config_file = variant["config"]
    checkpoint = repo_root / variant["checkpoint"]
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing native checkpoint: {checkpoint}")

    prompt_spec = _prompt_spec(prompt_mode, frames[0].shape[:2])
    actual_device = _resolve_native_device(native_device)
    temp_dir = _prepare_native_video_frames(repo_root, frames)

    try:
        predictor = build_sam2_video_predictor(
            config_file=config_file,
            ckpt_path=str(checkpoint),
            device=actual_device,
            mode="eval",
            vos_optimized=False,
        )

        t0 = time.perf_counter()
        inference_state = predictor.init_state(
            video_path=temp_dir.name,
            offload_video_to_cpu=False,
            offload_state_to_cpu=(actual_device == "cpu"),
        )
        init_ms = (time.perf_counter() - t0) * 1000.0

        if prompt_spec["type"] == "box":
            t0 = time.perf_counter()
            predictor.add_new_points_or_box(
                inference_state,
                frame_idx=0,
                obj_id=1,
                box=np.array(prompt_spec["box"], dtype=np.float32),
            )
            prompt_ms = (time.perf_counter() - t0) * 1000.0
        else:
            t0 = time.perf_counter()
            predictor.add_new_points_or_box(
                inference_state,
                frame_idx=0,
                obj_id=1,
                points=np.asarray(prompt_spec["points"], dtype=np.float32),
                labels=np.asarray(prompt_spec["labels"], dtype=np.int32),
            )
            prompt_ms = (time.perf_counter() - t0) * 1000.0

        propagate_times = []
        pred_masks = []
        areas = []
        iterator = predictor.propagate_in_video(inference_state)
        while True:
            t0 = time.perf_counter()
            try:
                _, _, mask_logits = next(iterator)
            except StopIteration:
                break
            propagate_times.append((time.perf_counter() - t0) * 1000.0)
            mask = np.squeeze((mask_logits[0] > 0).to(torch.uint8).cpu().numpy())
            pred_masks.append(mask.astype(bool, copy=False))
            areas.append(int(mask.sum()))

        return {
            "device": actual_device,
            "init_ms": init_ms,
            "prompt_ms": prompt_ms,
            "propagate_ms": propagate_times,
            "pred_masks": pred_masks,
            "areas": areas,
        }
    finally:
        temp_dir.cleanup()


def run_image_benchmark(args, ckpt_dir: Path, enc_path: str):
    if not args.image:
        return

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise RuntimeError(f"Could not read image: {args.image}")

    sess_enc = make_encoder_session(enc_path)
    enc_bench = _benchmark_encoder(sess_enc, image_bgr, repeats=args.repeat, warmup=args.warmup)
    prompt_inputs = _prompt_inputs(args.prompt, image_bgr.shape[:2], enc_bench["enc_shape"])

    results = {}
    labels = {}
    for mode in ("legacy", "specialized"):
        try:
            dec_path, resolved = resolve_image_decoder_path(ckpt_dir, args.prompt, mode)
        except FileNotFoundError:
            continue
        if mode == "specialized" and resolved == "legacy-safe-seed-points":
            print(
                "[IMAGE] Specialized seed-point image decoder is disabled by default "
                "because its fixed prompt slots change SAM prompt semantics. "
                "Skipping specialized comparison."
            )
            continue
        print(f"[IMAGE] Benchmarking {resolved} decoder: {Path(dec_path).name}")
        sess_dec = make_safe_session(dec_path, tag="decoder")
        labels[mode] = _runtime_label(mode, resolved)
        results[mode] = _benchmark_image_decoder(
            sess_dec,
            enc_bench["enc_dict"],
            prompt_inputs,
            repeats=args.repeat,
            warmup=args.warmup,
        )

    print("")
    print("[IMAGE] Shared encoder")
    print(f"  avg={_mean_ms(enc_bench['times_ms']):.2f} ms min={min(enc_bench['times_ms']):.2f} ms")

    for mode in ("legacy", "specialized"):
        if mode not in results:
            continue
        times = results[mode]["times_ms"]
        print(f"[IMAGE] {labels.get(mode, _runtime_label(mode)).capitalize()} decoder")
        print(f"  avg={_mean_ms(times):.2f} ms min={min(times):.2f} ms")

    if "legacy" in results and "specialized" in results:
        delta = _compare_logits(results["legacy"]["pred_mask"], results["specialized"]["pred_mask"])
        speedup = _mean_ms(results["legacy"]["times_ms"]) / max(_mean_ms(results["specialized"]["times_ms"]), 1e-9)
        print("[IMAGE] Comparison")
        print(
            f"  baseline={labels.get('legacy', 'legacy')}"
            f" candidate={labels.get('specialized', _runtime_label('specialized'))}"
        )
        print(f"  speedup={speedup:.2f}x")
        print(f"  mean_abs={delta['mean_abs']:.6f} max_abs={delta['max_abs']:.6f} binary_iou={delta['binary_iou']:.6f}")


def run_video_benchmark(args, ckpt_dir: Path, enc_path: str):
    if not args.video:
        return

    frames = _load_video_frames(args.video, args.frames)
    sess_enc = make_encoder_session(enc_path)
    enc_bench = _precompute_video_encoder(sess_enc, frames, warmup=args.warmup)
    prompt_inputs = _prompt_inputs(args.prompt, frames[0].shape[:2], enc_bench["enc_shape"])

    runtime_paths_by_mode = {}
    labels = {}
    for mode in ("legacy", "specialized"):
        try:
            runtime_paths_by_mode[mode] = resolve_video_runtime_paths(ckpt_dir, mode)
        except FileNotFoundError:
            continue
        labels[mode] = _runtime_label(mode, runtime_paths_by_mode[mode].get("mode"))

    if args.video_order == "both" and "legacy" in runtime_paths_by_mode and "specialized" in runtime_paths_by_mode:
        pass_orders = [
            ("forward", ["legacy", "specialized"]),
            ("reverse", ["specialized", "legacy"]),
        ]
    else:
        pass_orders = [("single", list(runtime_paths_by_mode.keys()))]

    results = {mode: [] for mode in runtime_paths_by_mode}
    for pass_name, modes in pass_orders:
        if len(pass_orders) > 1:
            print(f"[VIDEO] Pass {pass_name}")
        for mode in modes:
            runtime_paths = runtime_paths_by_mode[mode]
            print(f"[VIDEO] Benchmarking {runtime_paths['mode']} runtime")
            results[mode].append(
                _benchmark_video_variant(
                    runtime_paths,
                    enc_bench["records"],
                    prompt_inputs,
                    session_warmup=args.session_warmup,
                )
            )

    print("")
    print("[VIDEO] Shared encoder")
    print(f"  avg/frame={_mean_ms(enc_bench['times_ms']):.2f} ms frames={len(enc_bench['times_ms'])}")

    for mode in ("legacy", "specialized"):
        if mode not in results or not results[mode]:
            continue
        attn_avg = _median_of_video_runs(results[mode], "attn_ms")
        dec_avg = _median_of_video_runs(results[mode], "dec_ms")
        men_avg = _median_of_video_runs(results[mode], "men_ms")
        total = attn_avg + dec_avg + men_avg
        print(f"[VIDEO] {labels.get(mode, _runtime_label(mode)).capitalize()} runtime")
        if len(results[mode]) > 1:
            print(f"  median_of_passes={len(results[mode])}")
        print(f"  attn={attn_avg:.2f} ms dec={dec_avg:.2f} ms memenc={men_avg:.2f} ms total_no_encoder={total:.2f} ms")
        frame0_total = _median_of_video_section(results[mode], "frame0", "total_ms")
        propagate_total = _median_of_video_section(results[mode], "propagate", "total_ms")
        if frame0_total > 0.0:
            print(
                "  frame0"
                f" dec={_median_of_video_section(results[mode], 'frame0', 'dec_ms'):.2f} ms"
                f" memenc={_median_of_video_section(results[mode], 'frame0', 'men_ms'):.2f} ms"
                f" total_no_encoder={frame0_total:.2f} ms"
            )
        if propagate_total > 0.0:
            print(
                "  propagate_avg"
                f" frames={_video_section_count(results[mode], 'propagate', 'total_ms')}"
                f" attn={_median_of_video_section(results[mode], 'propagate', 'attn_ms'):.2f} ms"
                f" dec={_median_of_video_section(results[mode], 'propagate', 'dec_ms'):.2f} ms"
                f" memenc={_median_of_video_section(results[mode], 'propagate', 'men_ms'):.2f} ms"
                f" total_no_encoder={propagate_total:.2f} ms"
            )

    if "legacy" in results and "specialized" in results and results["legacy"] and results["specialized"]:
        delta = _compare_video_masks(results["legacy"][0]["pred_masks"], results["specialized"][0]["pred_masks"])
        legacy_total = (
            _median_of_video_runs(results["legacy"], "attn_ms")
            + _median_of_video_runs(results["legacy"], "dec_ms")
            + _median_of_video_runs(results["legacy"], "men_ms")
        )
        specialized_total = (
            _median_of_video_runs(results["specialized"], "attn_ms")
            + _median_of_video_runs(results["specialized"], "dec_ms")
            + _median_of_video_runs(results["specialized"], "men_ms")
        )
        speedup = legacy_total / max(specialized_total, 1e-9)
        print("[VIDEO] Comparison")
        print(
            f"  baseline={labels.get('legacy', 'legacy')}"
            f" candidate={labels.get('specialized', _runtime_label('specialized'))}"
        )
        print(f"  speedup={speedup:.2f}x")
        legacy_frame0 = _median_of_video_section(results["legacy"], "frame0", "total_ms")
        specialized_frame0 = _median_of_video_section(results["specialized"], "frame0", "total_ms")
        if legacy_frame0 > 0.0 and specialized_frame0 > 0.0:
            print(f"  frame0_speedup={legacy_frame0 / max(specialized_frame0, 1e-9):.2f}x")
        legacy_propagate = _median_of_video_section(results["legacy"], "propagate", "total_ms")
        specialized_propagate = _median_of_video_section(results["specialized"], "propagate", "total_ms")
        if legacy_propagate > 0.0 and specialized_propagate > 0.0:
            print(f"  propagate_speedup={legacy_propagate / max(specialized_propagate, 1e-9):.2f}x")
        print(f"  mean_abs={delta['mean_abs']:.6f} max_abs={delta['max_abs']:.6f} binary_iou={delta['binary_iou']:.6f}")

    if args.native_compare:
        repo_root = Path(__file__).resolve().parent.parent
        try:
            native = _benchmark_native_video(
                repo_root=repo_root,
                model_size=args.model_size,
                prompt_mode=args.prompt,
                frames=frames,
                native_device=args.native_device,
            )
        except Exception as exc:
            print(f"[VIDEO] Native SAM2 comparison skipped: {exc}")
            return
        native_avg = _mean_ms(native["propagate_ms"])
        native_zero = sum(1 for area in native["areas"] if area == 0)
        print("[VIDEO] Native SAM2")
        print(f"  device={native['device']} init={native['init_ms']:.2f} ms prompt={native['prompt_ms']:.2f} ms")
        print(
            f"  propagate_avg={native_avg:.2f} ms frames={len(native['propagate_ms'])}"
            f" zero_frames={native_zero}"
        )

        for mode in ("legacy", "specialized"):
            if mode not in results or not results[mode]:
                continue
            delta = _compare_native_masks(results[mode][0]["pred_masks"], native["pred_masks"])
            print(f"[VIDEO] {labels.get(mode, _runtime_label(mode)).capitalize()} vs native")
            print(
                f"  binary_iou={delta['binary_iou']:.6f}"
                f" area_ratio={delta['area_ratio']:.6f}"
                f" zero_frames={int(delta['zero_frames'])}"
            )


def main():
    ap = argparse.ArgumentParser(description="Benchmark legacy vs optimized ONNX runtime variants")
    ap.add_argument("--model_size", default="tiny", choices=["base_plus", "large", "small", "tiny"])
    ap.add_argument("--prompt", default="seed_points", choices=["seed_points", "bounding_box"])
    ap.add_argument("--image", default="", help="Optional image path for decoder benchmarking")
    ap.add_argument("--video", default="", help="Optional video path for runtime benchmarking")
    ap.add_argument("--frames", type=int, default=16, help="Max video frames to benchmark")
    ap.add_argument("--video_order", default="both", choices=["both", "single"], help="Run video variants in both orders and report median results, or only once in legacy-then-optimized order")
    ap.add_argument("--session_warmup", type=int, default=1, help="Warmup passes for video runtime sessions before measured runs")
    ap.add_argument("--warmup", type=int, default=3, help="Warmup runs for image decoder benchmark")
    ap.add_argument("--repeat", type=int, default=10, help="Measured runs for image decoder benchmark")
    ap.add_argument("--native_compare", action="store_true", help="Also benchmark native SAM2 video propagation on the same prompt and frames")
    ap.add_argument("--native_device", default="auto", choices=["auto", "cpu", "cuda"], help="Device for --native_compare")
    args = ap.parse_args()

    if not args.image and not args.video:
        sys.exit("Please provide --image and/or --video.")

    _load_runtime_helpers()
    print_system_info()
    set_cv2_threads(1)

    repo_root = Path(__file__).resolve().parent.parent
    ckpt_dir = repo_root / "checkpoints" / args.model_size
    enc_path = prefer_quantized_encoder(str(ckpt_dir))
    if enc_path is None:
        sys.exit(f"ERROR: Encoder ONNX not found in {ckpt_dir}")

    run_image_benchmark(args, ckpt_dir, enc_path)
    run_video_benchmark(args, ckpt_dir, enc_path)


if __name__ == "__main__":
    main()
