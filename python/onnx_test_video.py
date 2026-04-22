#!/usr/bin/env python3
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PyQt5 import QtWidgets

from onnx_test_utils import (
    compute_display_base,
    green_overlay,
    make_encoder_session,
    make_safe_session,
    prefer_quantized_encoder,
    prepare_box_prompt,
    prepare_image,
    prepare_points,
    print_system_info,
    resolve_video_runtime_paths,
    run_decoder,
    run_encoder,
    run_memory_attention,
    run_memory_encoder,
    set_cv2_threads,
    VideoMemoryBank,
    warmup_video_runtime_sessions,
)


WINDOW_NAME = "Video Anchors - SAM2"
JUMP_FRAMES = 10


def _binary_preview_mask(pred_mask: np.ndarray) -> np.ndarray:
    return (pred_mask[0, 0] > 0).astype(np.uint8)


def _load_frame(cap: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_index}")
    return frame


def _frame_count(cap: cv2.VideoCapture, max_frames: int) -> int:
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        if max_frames > 0:
            return max_frames
        raise RuntimeError("Could not determine video frame count. Use --max_frames.")
    if max_frames > 0:
        total = min(total, max_frames)
    return max(1, total)


def _encode_frame(
    sess_enc,
    frame_bgr: np.ndarray,
    enc_shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    tensor, _ = prepare_image(frame_bgr, enc_shape)
    enc = run_encoder(sess_enc, tensor)
    record = {
        "embed": enc["image_embeddings"].astype(np.float32, copy=False),
        "curr_feat": enc.get("current_vision_feat", enc["image_embeddings"]).astype(np.float32, copy=False),
        "f0": enc["high_res_features1"].astype(np.float32, copy=False),
        "f1": enc["high_res_features2"].astype(np.float32, copy=False),
        "vis_pos": None,
    }
    if "vision_pos_embed" in enc:
        record["vis_pos"] = enc["vision_pos_embed"].astype(np.float32, copy=False)
    return record


def _clamp_frame(frame_index: int, total_frames: int) -> int:
    return max(0, min(frame_index, total_frames - 1))


def _draw_hud(
    image: np.ndarray,
    frame_index: int,
    total_frames: int,
    anchor_count: int,
    prompt: str,
) -> None:
    lines = [
        f"Frame {frame_index + 1}/{total_frames} | Anchors: {anchor_count}",
        f"Prompt: {prompt}",
        "A/D: +/-1 frame | J/L: +/-10 frames",
        "Enter/Space: run video | Esc/Q: finish | C: clear frame",
    ]
    if prompt == "seed_points":
        lines.append("L-click: FG | R-click: BG | M-click: reset current frame")
    else:
        lines.append("L-drag: box | R/M-click: reset current frame")

    y = 28
    for line in lines:
        cv2.putText(image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (15, 15, 15), 4, cv2.LINE_AA)
        cv2.putText(image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        y += 28


def _points_prompt_inputs(
    annotation: dict[str, list[tuple[int, int]] | list[int]],
    image_size: tuple[int, int],
    enc_size: tuple[int, int],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    points = annotation.get("points", [])
    labels = annotation.get("labels", [])
    if not points:
        return None, None
    return prepare_points(points, labels, image_size, enc_size)


def _box_prompt_inputs(
    annotation: dict[str, tuple[int, int, int, int]],
    image_size: tuple[int, int],
    enc_size: tuple[int, int],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    return prepare_box_prompt(annotation.get("box"), image_size, enc_size)


def _collect_point_anchors(video_path: str, sess_enc, sess_dec0, enc_shape: tuple[int, int], max_frames: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("cannot open video")
    total_frames = _frame_count(cap, max_frames)

    annotations: dict[int, dict[str, list]] = {}
    encoded_cache: dict[int, dict[str, np.ndarray]] = {}
    current_index = 0
    current_frame = _load_frame(cap, current_index)
    current_points: list[tuple[int, int]] = []
    current_labels: list[int] = []
    current_scale = 1.0

    def sync_current_annotation() -> None:
        if current_points:
            annotations[current_index] = {
                "points": list(current_points),
                "labels": list(current_labels),
            }
        else:
            annotations.pop(current_index, None)

    def load_current_annotation() -> None:
        nonlocal current_points, current_labels
        ann = annotations.get(current_index)
        if ann is None:
            current_points = []
            current_labels = []
        else:
            current_points = list(ann["points"])
            current_labels = list(ann["labels"])

    def get_encoded_record() -> dict[str, np.ndarray]:
        if current_index not in encoded_cache:
            encoded_cache[current_index] = _encode_frame(sess_enc, current_frame, enc_shape)
        return encoded_cache[current_index]

    def render() -> None:
        nonlocal current_scale
        base, current_scale = compute_display_base(current_frame, max_side=1200)
        vis = base.copy()

        if current_points:
            record = get_encoded_record()
            image_size = (current_frame.shape[0], current_frame.shape[1])
            pts, lbls = prepare_points(current_points, current_labels, image_size, enc_shape)
            _, _, pred_low = run_decoder(sess_dec0, pts, lbls, record["embed"], record["f0"], record["f1"])
            if pred_low is not None:
                preview = cv2.resize(_binary_preview_mask(pred_low), (base.shape[1], base.shape[0]), cv2.INTER_NEAREST)
                vis = green_overlay(vis, preview, 0.5)

        for idx, (px, py) in enumerate(current_points):
            color = (0, 0, 255) if current_labels[idx] == 1 else (255, 0, 0)
            cv2.circle(vis, (int(px * current_scale), int(py * current_scale)), 6, color, -1)

        _draw_hud(vis, current_index, total_frames, len(annotations), "seed_points")
        cv2.imshow(WINDOW_NAME, vis)

    def goto_frame(frame_index: int) -> None:
        nonlocal current_index, current_frame
        sync_current_annotation()
        current_index = _clamp_frame(frame_index, total_frames)
        current_frame = _load_frame(cap, current_index)
        load_current_annotation()
        render()

    def on_mouse(event, x, y, _flags, _param) -> None:
        if event == cv2.EVENT_MBUTTONDOWN:
            current_points.clear()
            current_labels.clear()
            sync_current_annotation()
            render()
            return
        if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            return
        px = int(x / max(current_scale, 1e-6))
        py = int(y / max(current_scale, 1e-6))
        current_points.append((px, py))
        current_labels.append(1 if event == cv2.EVENT_LBUTTONDOWN else 0)
        sync_current_annotation()
        render()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    render()

    print("[INFO] Multi-anchor seed mode ready.")
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10, 32, 27, ord("q"), ord("Q")):
            sync_current_annotation()
            break
        if key in (ord("a"), ord("A")):
            goto_frame(current_index - 1)
        elif key in (ord("d"), ord("D")):
            goto_frame(current_index + 1)
        elif key in (ord("j"), ord("J")):
            goto_frame(current_index - JUMP_FRAMES)
        elif key in (ord("l"), ord("L")):
            goto_frame(current_index + JUMP_FRAMES)
        elif key in (ord("c"), ord("C")):
            current_points.clear()
            current_labels.clear()
            sync_current_annotation()
            render()

    cv2.destroyAllWindows()
    cap.release()
    return annotations, encoded_cache, total_frames


def _collect_box_anchors(video_path: str, sess_enc, sess_dec0, enc_shape: tuple[int, int], max_frames: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("cannot open video")
    total_frames = _frame_count(cap, max_frames)

    annotations: dict[int, dict[str, tuple[int, int, int, int]]] = {}
    encoded_cache: dict[int, dict[str, np.ndarray]] = {}
    current_index = 0
    current_frame = _load_frame(cap, current_index)
    current_box: tuple[int, int, int, int] | None = None
    drag_start: tuple[int, int] | None = None
    drag_end: tuple[int, int] | None = None
    drawing = False
    current_scale = 1.0

    def sync_current_annotation() -> None:
        if current_box is not None:
            annotations[current_index] = {"box": current_box}
        else:
            annotations.pop(current_index, None)

    def load_current_annotation() -> None:
        nonlocal current_box, drag_start, drag_end, drawing
        drawing = False
        ann = annotations.get(current_index)
        current_box = None if ann is None else ann["box"]
        drag_start = None
        drag_end = None

    def get_encoded_record() -> dict[str, np.ndarray]:
        if current_index not in encoded_cache:
            encoded_cache[current_index] = _encode_frame(sess_enc, current_frame, enc_shape)
        return encoded_cache[current_index]

    def render() -> None:
        nonlocal current_scale
        base, current_scale = compute_display_base(current_frame, max_side=1200)
        vis = base.copy()

        preview_box = current_box
        if drawing and drag_start is not None and drag_end is not None:
            x1 = int(drag_start[0] / max(current_scale, 1e-6))
            y1 = int(drag_start[1] / max(current_scale, 1e-6))
            x2 = int(drag_end[0] / max(current_scale, 1e-6))
            y2 = int(drag_end[1] / max(current_scale, 1e-6))
            preview_box = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

        if preview_box is not None:
            record = get_encoded_record()
            image_size = (current_frame.shape[0], current_frame.shape[1])
            pts, lbls = prepare_box_prompt(preview_box, image_size, enc_shape)
            _, _, pred_low = run_decoder(sess_dec0, pts, lbls, record["embed"], record["f0"], record["f1"])
            if pred_low is not None:
                preview = cv2.resize(_binary_preview_mask(pred_low), (base.shape[1], base.shape[0]), cv2.INTER_NEAREST)
                vis = green_overlay(vis, preview, 0.5)
            x1, y1, x2, y2 = preview_box
            cv2.rectangle(
                vis,
                (int(x1 * current_scale), int(y1 * current_scale)),
                (int(x2 * current_scale), int(y2 * current_scale)),
                (0, 255, 255),
                2,
            )

        _draw_hud(vis, current_index, total_frames, len(annotations), "bounding_box")
        cv2.imshow(WINDOW_NAME, vis)

    def goto_frame(frame_index: int) -> None:
        nonlocal current_index, current_frame
        sync_current_annotation()
        current_index = _clamp_frame(frame_index, total_frames)
        current_frame = _load_frame(cap, current_index)
        load_current_annotation()
        render()

    def on_mouse(event, x, y, _flags, _param) -> None:
        nonlocal current_box, drag_start, drag_end, drawing
        if event in (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MBUTTONDOWN):
            current_box = None
            drag_start = None
            drag_end = None
            drawing = False
            sync_current_annotation()
            render()
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            drag_start = (x, y)
            drag_end = (x, y)
            drawing = True
            render()
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            drag_end = (x, y)
            render()
        elif event == cv2.EVENT_LBUTTONUP and drawing:
            drag_end = (x, y)
            drawing = False
            x1 = int(drag_start[0] / max(current_scale, 1e-6))
            y1 = int(drag_start[1] / max(current_scale, 1e-6))
            x2 = int(drag_end[0] / max(current_scale, 1e-6))
            y2 = int(drag_end[1] / max(current_scale, 1e-6))
            current_box = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            sync_current_annotation()
            render()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    render()

    print("[INFO] Multi-anchor box mode ready.")
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10, 32, 27, ord("q"), ord("Q")):
            sync_current_annotation()
            break
        if key in (ord("a"), ord("A")):
            goto_frame(current_index - 1)
        elif key in (ord("d"), ord("D")):
            goto_frame(current_index + 1)
        elif key in (ord("j"), ord("J")):
            goto_frame(current_index - JUMP_FRAMES)
        elif key in (ord("l"), ord("L")):
            goto_frame(current_index + JUMP_FRAMES)
        elif key in (ord("c"), ord("C")):
            current_box = None
            drag_start = None
            drag_end = None
            drawing = False
            sync_current_annotation()
            render()

    cv2.destroyAllWindows()
    cap.release()
    return annotations, encoded_cache, total_frames


def _collect_video_anchors(video_path: str, prompt: str, sess_enc, sess_dec0, enc_shape: tuple[int, int], max_frames: int):
    if prompt == "bounding_box":
        return _collect_box_anchors(video_path, sess_enc, sess_dec0, enc_shape, max_frames)
    return _collect_point_anchors(video_path, sess_enc, sess_dec0, enc_shape, max_frames)


def process_video(args):
    repo_root = Path(__file__).resolve().parent.parent
    ckpt_dir = repo_root / "checkpoints" / args.model_size
    enc_path = prefer_quantized_encoder(str(ckpt_dir))
    if enc_path is None:
        sys.exit(f"ERROR: Encoder ONNX not found in {ckpt_dir}")

    try:
        runtime_paths = resolve_video_runtime_paths(ckpt_dir, args.artifacts)
    except FileNotFoundError as exc:
        sys.exit(f"ERROR: {exc}")

    print(f"[INFO] Video artifacts : {runtime_paths['mode']}")
    print(f"[INFO] Decoder init   : {Path(runtime_paths['decoder_init']).name}")
    print(f"[INFO] Decoder prop   : {Path(runtime_paths['decoder_propagate']).name}")
    print(f"[INFO] Mem attention  : {Path(runtime_paths['memory_attention']).name}")
    print(f"[INFO] Mem encoder    : {Path(runtime_paths['memory_encoder']).name}")

    sess_enc = make_encoder_session(enc_path)
    sess_dec0 = make_safe_session(runtime_paths["decoder_init"], tag="decoder")
    if runtime_paths["decoder_propagate"] == runtime_paths["decoder_init"]:
        sess_decn = sess_dec0
    else:
        sess_decn = make_safe_session(runtime_paths["decoder_propagate"], tag="decoder")
    sess_men = make_safe_session(runtime_paths["memory_encoder"], tag="memory_encoder")
    sess_mat = make_safe_session(runtime_paths["memory_attention"], tag="memory_attention")

    enc_h, enc_w = sess_enc.get_inputs()[0].shape[2:]
    enc_shape = (enc_h, enc_w)
    print(f"[INFO] Encoder input = {(enc_h, enc_w)}")

    try:
        annotations, encoded_cache, total_frames = _collect_video_anchors(
            args.video,
            args.prompt,
            sess_enc,
            sess_dec0,
            enc_shape,
            args.max_frames,
        )
    except RuntimeError as exc:
        sys.exit(f"ERROR: {exc}")

    if not annotations:
        sys.exit("ERROR: No anchor annotations were provided.")

    anchor_frames = sorted(annotations.keys())
    print(f"[INFO] Anchor frames : {anchor_frames}")

    first_anchor = anchor_frames[0]
    preview_cap = cv2.VideoCapture(args.video)
    if not preview_cap.isOpened():
        sys.exit("ERROR: cannot reopen video for warmup")
    first_anchor_frame = _load_frame(preview_cap, first_anchor)
    preview_cap.release()

    first_record = encoded_cache.get(first_anchor)
    if first_record is None:
        first_record = _encode_frame(sess_enc, first_anchor_frame, enc_shape)
        encoded_cache[first_anchor] = first_record

    image_size = (first_anchor_frame.shape[0], first_anchor_frame.shape[1])
    if args.prompt == "bounding_box":
        prompt_inputs = _box_prompt_inputs(annotations[first_anchor], image_size, enc_shape)
    else:
        prompt_inputs = _points_prompt_inputs(annotations[first_anchor], image_size, enc_shape)

    if args.session_warmup > 0:
        print(f"[INFO] Session warmup : {args.session_warmup} pass(es)")
        warmup_video_runtime_sessions(
            sess_dec0=sess_dec0,
            sess_decn=sess_decn,
            sess_mat=sess_mat,
            sess_men=sess_men,
            first_record=first_record,
            propagate_record=first_record,
            prompt_inputs=prompt_inputs,
            repeats=args.session_warmup,
        )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit("ERROR: cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w_org = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_org = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_output_frames = _frame_count(cap, args.max_frames)
    out_path = os.path.splitext(args.video)[0] + f"_{runtime_paths['mode']}_mask_overlay.mkv"
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"XVID"),
        fps if fps > 0 else 25.0,
        (w_org, h_org),
    )
    if not writer.isOpened():
        sys.exit("ERROR: cannot open VideoWriter")

    memory_bank = VideoMemoryBank.from_session(sess_mat)
    print(
        f"[INFO] Video memory caps : frames={memory_bank.max_slots} "
        f"object_ptrs={memory_bank.max_pointer_slots}"
    )

    active_segment = False
    fidx = 0
    while True:
        if fidx >= total_output_frames:
            break

        ok, frame = cap.read()
        if not ok or frame is None:
            break

        anchor = annotations.get(fidx)
        enc_start = time.time()
        record = encoded_cache.get(fidx)
        if record is None:
            record = _encode_frame(sess_enc, frame, enc_shape)
            if anchor is not None:
                encoded_cache[fidx] = record
        enc_ms = (time.time() - enc_start) * 1000

        if anchor is not None:
            memory_bank = VideoMemoryBank.from_session(sess_mat)
            active_segment = True
            if args.prompt == "bounding_box":
                pts0, lbls0 = _box_prompt_inputs(anchor, (frame.shape[0], frame.shape[1]), enc_shape)
            else:
                pts0, lbls0 = _points_prompt_inputs(anchor, (frame.shape[0], frame.shape[1]), enc_shape)

            t_dec = time.time()
            obj_ptr, mask_for_mem, pred_mask = run_decoder(
                sess_dec0,
                pts0,
                lbls0,
                record["embed"],
                record["f0"],
                record["f1"],
            )
            dec_ms = (time.time() - t_dec) * 1000
            mat_ms = 0.0

            if mask_for_mem is None or pred_mask is None:
                sys.exit(f"ERROR: Decoder init failed at anchor frame {fidx}.")

            t_men = time.time()
            mem_feats, mem_pos, temporal_code = run_memory_encoder(
                sess_men,
                mask_for_mem=mask_for_mem[:, 0:1],
                pix_feat=record["embed"],
            )
            memory_bank.set_conditioning(mem_feats, mem_pos, temporal_code, obj_ptr=obj_ptr, frame_index=0)
            men_ms = (time.time() - t_men) * 1000
            print(f"[INFO] Anchor frame {fidx:03d} reset interval")
        elif active_segment:
            memory_1, memory_pos_embed = memory_bank.build_attention_state()
            object_ptrs, obj_ptr_offsets = memory_bank.build_object_pointer_state(frame_index=fidx)

            if memory_1 is not None and memory_pos_embed is not None:
                t_mat = time.time()
                fused_embed = run_memory_attention(
                    sess_mat,
                    current_vision_feat=record["curr_feat"],
                    current_vision_pos_embed=record["vis_pos"],
                    memory_1=memory_1,
                    memory_pos_embed=memory_pos_embed,
                    memory_0=object_ptrs,
                    obj_ptr_offsets=obj_ptr_offsets,
                ).astype(np.float32, copy=False)
                mat_ms = (time.time() - t_mat) * 1000
            else:
                fused_embed = record["embed"]
                mat_ms = 0.0

            t_dec = time.time()
            obj_ptr, mask_for_mem, pred_mask = run_decoder(sess_decn, None, None, fused_embed, record["f0"], record["f1"])
            dec_ms = (time.time() - t_dec) * 1000
            if mask_for_mem is None or pred_mask is None:
                sys.exit(f"ERROR: Decoder propagation failed at frame {fidx}.")

            t_men = time.time()
            mem_feats, mem_pos, temporal_code = run_memory_encoder(
                sess_men,
                mask_for_mem=mask_for_mem[:, 0:1],
                pix_feat=fused_embed,
            )
            memory_bank.append_recent(mem_feats, mem_pos, temporal_code, obj_ptr=obj_ptr, frame_index=fidx)
            men_ms = (time.time() - t_men) * 1000
        else:
            logits = np.zeros((h_org, w_org), np.float32)
            mask = np.zeros((h_org, w_org), np.uint8)
            writer.write(green_overlay(frame, mask, 0.5))
            print(f"Frame {fidx:03d} - Enc:{enc_ms:.1f} | Inactive interval")
            fidx += 1
            continue

        logits = pred_mask[0, 0]
        mask_hi = cv2.resize(logits, (w_org, h_org), cv2.INTER_LINEAR)
        mask = (mask_hi > 0).astype(np.uint8)
        writer.write(green_overlay(frame, mask, 0.5))

        if anchor is not None:
            print(f"Frame {fidx:03d} - Enc:{enc_ms:.1f} | Dec:{dec_ms:.1f} | MemEnc:{men_ms:.1f}")
        else:
            print(f"Frame {fidx:03d} - Enc:{enc_ms:.1f} | Attn:{mat_ms:.1f} | Dec:{dec_ms:.1f} | MemEnc:{men_ms:.1f}")
        fidx += 1

    cap.release()
    writer.release()
    print(f"Done! Wrote {fidx} frames with overlays to {out_path}")


def main():
    print_system_info()
    set_cv2_threads(1)

    ap = argparse.ArgumentParser(description="Video segmentation demo for SAM2 ONNX")
    ap.add_argument("--model_size", default="base_plus", choices=["base_plus", "large", "small", "tiny"])
    ap.add_argument("--prompt", default="seed_points", choices=["seed_points", "bounding_box"])
    ap.add_argument("--artifacts", default="auto", choices=["auto", "legacy", "specialized"])
    ap.add_argument("--max_frames", type=int, default=0, help="Max frames to process (0 = all).")
    ap.add_argument("--session_warmup", type=int, default=1, help="Warmup passes for decoder/memory sessions before processing.")
    ap.add_argument("--video", default="", help="Optional video path. If omitted, a file dialog opens.")
    args = ap.parse_args()

    if not args.video:
        app = QtWidgets.QApplication(sys.argv)
        vid, _ = QtWidgets.QFileDialog.getOpenFileName(
            None,
            "Select Video",
            "",
            "Video files (*.mp4 *.mkv *.avi *.mov *.m4v);;All files (*.*)",
        )
        if not vid:
            sys.exit("No video selected, exiting.")
        args.video = vid
    print(f"[INFO] Selected video: {args.video}")

    process_video(args)


if __name__ == "__main__":
    main()
