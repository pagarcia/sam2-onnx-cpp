#!/usr/bin/env python3
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import statistics
import sys
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


def _mean_ms(values):
    return statistics.mean(values) if values else 0.0


def _prompt_inputs(prompt_mode: str, image_shape: tuple[int, int], enc_shape: tuple[int, int]):
    h_org, w_org = image_shape
    if prompt_mode == "bounding_box":
        rect = (
            int(w_org * 0.2),
            int(h_org * 0.2),
            int(w_org * 0.8),
            int(h_org * 0.8),
        )
        return prepare_box_prompt(rect, image_shape, enc_shape)

    points = [
        (int(w_org * 0.5), int(h_org * 0.5)),
        (int(w_org * 0.25), int(h_org * 0.25)),
    ]
    labels = [1, 0]
    return prepare_points(points, labels, image_shape, enc_shape)


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


def _precompute_video_encoder(sess_enc, frames: list[np.ndarray]):
    enc_input_name = sess_enc.get_inputs()[0].name
    enc_out_names = [o.name for o in sess_enc.get_outputs()]
    enc_h, enc_w = sess_enc.get_inputs()[0].shape[2:]

    records = []
    enc_times = []
    for frame in frames:
        tensor, _ = prepare_image(frame, (enc_h, enc_w))
        t0 = time.perf_counter()
        enc_vals = sess_enc.run(None, {enc_input_name: tensor})
        enc_times.append((time.perf_counter() - t0) * 1000.0)
        enc = dict(zip(enc_out_names, enc_vals))
        records.append(
            {
                "embed": enc["image_embeddings"].astype(np.float32, copy=False),
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


def _benchmark_video_variant(paths: dict[str, str], encoded_frames, prompt_inputs):
    sess_dec0 = make_safe_session(paths["decoder_init"], tag="decoder")
    if paths["decoder_propagate"] == paths["decoder_init"]:
        sess_decn = sess_dec0
    else:
        sess_decn = make_safe_session(paths["decoder_propagate"], tag="decoder")
    sess_mat = make_safe_session(paths["memory_attention"], tag="memory_attention")
    sess_men = make_safe_session(paths["memory_encoder"], tag="memory_encoder")

    pts0, lbls0 = prompt_inputs
    mem_feats = None
    mem_pos = None
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
        f0 = record["f0"]
        f1 = record["f1"]
        vis_pos = record["vis_pos"]

        if idx > 0 and mem_feats is not None:
            t0 = time.perf_counter()
            fused_embed = run_memory_attention(
                sess_mat,
                current_vision_feat=enc_embed,
                current_vision_pos_embed=vis_pos,
                memory_1=mem_feats,
                memory_pos_embed=mem_pos,
            ).astype(np.float32, copy=False)
            attn_elapsed_ms = (time.perf_counter() - t0) * 1000.0
        else:
            fused_embed = enc_embed
            attn_elapsed_ms = 0.0

        t0 = time.perf_counter()
        if idx == 0:
            _, mask_for_mem, pred_mask = run_decoder(sess_dec0, pts0, lbls0, fused_embed, f0, f1)
        else:
            _, mask_for_mem, pred_mask = run_decoder(sess_decn, None, None, fused_embed, f0, f1)
        dec_elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if mask_for_mem is None or pred_mask is None:
            raise RuntimeError("Decoder did not return mask_for_mem and pred_mask")

        t0 = time.perf_counter()
        mem_feats, mem_pos, _ = run_memory_encoder(
            sess_men,
            mask_for_mem=mask_for_mem[:, 0:1],
            pix_feat=fused_embed,
        )
        men_elapsed_ms = (time.perf_counter() - t0) * 1000.0

        mem_feats = mem_feats.astype(np.float32, copy=False)
        mem_pos = mem_pos.astype(np.float32, copy=False)
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
    for mode in ("legacy", "specialized"):
        try:
            dec_path, resolved = resolve_image_decoder_path(ckpt_dir, args.prompt, mode)
        except FileNotFoundError:
            continue
        print(f"[IMAGE] Benchmarking {resolved} decoder: {Path(dec_path).name}")
        sess_dec = make_safe_session(dec_path, tag="decoder")
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
        print(f"[IMAGE] {mode.capitalize()} decoder")
        print(f"  avg={_mean_ms(times):.2f} ms min={min(times):.2f} ms")

    if "legacy" in results and "specialized" in results:
        delta = _compare_logits(results["legacy"]["pred_mask"], results["specialized"]["pred_mask"])
        speedup = _mean_ms(results["legacy"]["times_ms"]) / max(_mean_ms(results["specialized"]["times_ms"]), 1e-9)
        print("[IMAGE] Comparison")
        print(f"  speedup={speedup:.2f}x")
        print(f"  mean_abs={delta['mean_abs']:.6f} max_abs={delta['max_abs']:.6f} binary_iou={delta['binary_iou']:.6f}")


def run_video_benchmark(args, ckpt_dir: Path, enc_path: str):
    if not args.video:
        return

    frames = _load_video_frames(args.video, args.frames)
    sess_enc = make_encoder_session(enc_path)
    enc_bench = _precompute_video_encoder(sess_enc, frames)
    prompt_inputs = _prompt_inputs(args.prompt, frames[0].shape[:2], enc_bench["enc_shape"])

    runtime_paths_by_mode = {}
    for mode in ("legacy", "specialized"):
        try:
            runtime_paths_by_mode[mode] = resolve_video_runtime_paths(ckpt_dir, mode)
        except FileNotFoundError:
            continue

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
            results[mode].append(_benchmark_video_variant(runtime_paths, enc_bench["records"], prompt_inputs))

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
        print(f"[VIDEO] {mode.capitalize()} runtime")
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


def main():
    ap = argparse.ArgumentParser(description="Benchmark legacy vs specialized ONNX runtime variants")
    ap.add_argument("--model_size", default="tiny", choices=["base_plus", "large", "small", "tiny"])
    ap.add_argument("--prompt", default="seed_points", choices=["seed_points", "bounding_box"])
    ap.add_argument("--image", default="", help="Optional image path for decoder benchmarking")
    ap.add_argument("--video", default="", help="Optional video path for runtime benchmarking")
    ap.add_argument("--frames", type=int, default=16, help="Max video frames to benchmark")
    ap.add_argument("--video_order", default="both", choices=["both", "single"], help="Run video variants in both orders and report median results, or only once in legacy-then-specialized order")
    ap.add_argument("--warmup", type=int, default=3, help="Warmup runs for image decoder benchmark")
    ap.add_argument("--repeat", type=int, default=10, help="Measured runs for image decoder benchmark")
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
