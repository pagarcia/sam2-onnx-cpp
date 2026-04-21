#!/usr/bin/env python3
# CPU-optimized: encoder fast (BASIC/EXTENDED), decoder/memory safe.
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


def _binary_preview_mask(pred_mask: np.ndarray) -> np.ndarray:
    return (pred_mask[0, 0] > 0).astype(np.uint8)


def interactive_select_points(first_bgr, sess_enc, sess_dec, enc_shape):
    enc_h, enc_w = enc_shape
    tensor, (h_org, w_org) = prepare_image(first_bgr, (enc_h, enc_w))
    enc = run_encoder(sess_enc, tensor)
    embed = enc["image_embeddings"]
    curr_feat = enc.get("current_vision_feat", embed)
    f0 = enc["high_res_features1"]
    f1 = enc["high_res_features2"]

    base, scale = compute_display_base(first_bgr, max_side=1200)
    points, labels = [], []

    def show(mask=None):
        vis = base.copy()
        if mask is not None:
            m = cv2.resize(mask, (base.shape[1], base.shape[0]), cv2.INTER_NEAREST)
            vis = green_overlay(vis, m, 0.5)
        for i, (px, py) in enumerate(points):
            color = (0, 0, 255) if labels[i] == 1 else (255, 0, 0)
            cv2.circle(vis, (int(px * scale), int(py * scale)), 6, color, -1)
        cv2.imshow("First Frame - SAM-2", vis)

    def run():
        if not points:
            show()
            return
        pts, lbls = prepare_points(points, labels, (h_org, w_org), (enc_h, enc_w))
        _, _, pred_low = run_decoder(sess_dec, pts, lbls, embed, f0, f1)
        show(_binary_preview_mask(pred_low))

    def cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((int(x / scale), int(y / scale)))
            labels.append(1)
            run()
        elif event == cv2.EVENT_RBUTTONDOWN:
            points.append((int(x / scale), int(y / scale)))
            labels.append(0)
            run()
        elif event == cv2.EVENT_MBUTTONDOWN:
            points.clear()
            labels.clear()
            run()

    cv2.namedWindow("First Frame - SAM-2")
    cv2.setMouseCallback("First Frame - SAM-2", cb)
    run()
    print("[INFO] L-click=FG, R-click=BG, M-click=reset. ESC/Enter to continue.")
    while True:
        if cv2.waitKey(20) & 0xFF in (27, 13):
            break
    cv2.destroyAllWindows()
    vis_pos = enc.get("vision_pos_embed")
    if vis_pos is not None:
        vis_pos = vis_pos.astype(np.float32, copy=False)
    return points, labels, embed, curr_feat, f0, f1, vis_pos, (h_org, w_org)


def interactive_select_box(first_bgr, sess_enc, sess_dec, enc_shape):
    enc_h, enc_w = enc_shape
    tensor, (h_org, w_org) = prepare_image(first_bgr, (enc_h, enc_w))
    enc = run_encoder(sess_enc, tensor)
    embed = enc["image_embeddings"]
    curr_feat = enc.get("current_vision_feat", embed)
    f0 = enc["high_res_features1"]
    f1 = enc["high_res_features2"]

    base, scale = compute_display_base(first_bgr, max_side=1200)
    rect_s = rect_e = None
    drawing = False

    def show(mask=None):
        vis = base.copy()
        if mask is not None:
            m = cv2.resize(mask, (base.shape[1], base.shape[0]), cv2.INTER_NEAREST)
            vis = green_overlay(vis, m, 0.5)
        if rect_s and rect_e:
            cv2.rectangle(vis, rect_s, rect_e, (0, 255, 255), 2)
        cv2.imshow("First Frame - SAM-2", vis)

    def run():
        if not (rect_s and rect_e):
            show()
            return
        x1d, y1d = rect_s
        x2d, y2d = rect_e
        x1, x2 = sorted((int(x1d / scale), int(x2d / scale)))
        y1, y2 = sorted((int(y1d / scale), int(y2d / scale)))
        pts, lbls = prepare_box_prompt((x1, y1, x2, y2), (h_org, w_org), (enc_h, enc_w))
        _, _, pred_low = run_decoder(sess_dec, pts, lbls, embed, f0, f1)
        show(_binary_preview_mask(pred_low))

    def cb(event, x, y, flags, param):
        nonlocal rect_s, rect_e, drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            rect_s = rect_e = (x, y)
            show()
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            rect_e = (x, y)
            show()
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            rect_e = (x, y)
            run()
        elif event in (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_LBUTTONDBLCLK):
            rect_s = rect_e = None
            show()

    cv2.namedWindow("First Frame - SAM-2")
    cv2.setMouseCallback("First Frame - SAM-2", cb)
    show()
    print("[INFO] Draw rectangle, release to preview. ESC/Enter to continue.")
    while True:
        if cv2.waitKey(20) & 0xFF in (27, 13):
            break
    cv2.destroyAllWindows()

    if rect_s and rect_e:
        x1d, y1d = rect_s
        x2d, y2d = rect_e
        x1, x2 = sorted((int(x1d / scale), int(x2d / scale)))
        y1, y2 = sorted((int(y1d / scale), int(y2d / scale)))
        box = (x1, y1, x2, y2)
    else:
        box = None
    vis_pos = enc.get("vision_pos_embed")
    if vis_pos is not None:
        vis_pos = vis_pos.astype(np.float32, copy=False)
    return box, embed, curr_feat, f0, f1, vis_pos, (h_org, w_org)


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
    print(f"[INFO] Encoder input = {(enc_h, enc_w)}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit("ERROR: cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w_org = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_org = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = os.path.splitext(args.video)[0] + f"_{runtime_paths['mode']}_mask_overlay.mkv"
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"XVID"),
        fps if fps > 0 else 25.0,
        (w_org, h_org),
    )
    if not writer.isOpened():
        sys.exit("ERROR: cannot open VideoWriter")

    ret, first_bgr = cap.read()
    if not ret:
        sys.exit("ERROR: empty video")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if args.prompt == "bounding_box":
        box, embed0, curr_feat0, f0_0, f1_0, vis_pos0, (h_org, w_org) = interactive_select_box(
            first_bgr, sess_enc, sess_dec0, (enc_h, enc_w)
        )
        pts0, lbls0 = prepare_box_prompt(box, (h_org, w_org), (enc_h, enc_w)) if box else (None, None)
    else:
        pts, lbls, embed0, curr_feat0, f0_0, f1_0, vis_pos0, (h_org, w_org) = interactive_select_points(
            first_bgr, sess_enc, sess_dec0, (enc_h, enc_w)
        )
        pts0, lbls0 = prepare_points(pts, lbls, (h_org, w_org), (enc_h, enc_w))

    embed0 = embed0.astype(np.float32, copy=False)
    curr_feat0 = curr_feat0.astype(np.float32, copy=False)
    f0_0 = f0_0.astype(np.float32, copy=False)
    f1_0 = f1_0.astype(np.float32, copy=False)

    if args.session_warmup > 0:
        print(f"[INFO] Session warmup : {args.session_warmup} pass(es)")
        warmup_video_runtime_sessions(
            sess_dec0=sess_dec0,
            sess_decn=sess_decn,
            sess_mat=sess_mat,
            sess_men=sess_men,
            first_record={
                "embed": embed0,
                "curr_feat": curr_feat0,
                "f0": f0_0,
                "f1": f1_0,
                "vis_pos": vis_pos0,
            },
            propagate_record={
                "embed": embed0,
                "curr_feat": curr_feat0,
                "f0": f0_0,
                "f1": f1_0,
                "vis_pos": vis_pos0,
            },
            prompt_inputs=(pts0, lbls0),
            repeats=args.session_warmup,
        )

    fidx = 0
    memory_bank = VideoMemoryBank.from_session(sess_mat)
    print(
        f"[INFO] Video memory caps : frames={memory_bank.max_slots} "
        f"object_ptrs={memory_bank.max_pointer_slots}"
    )

    enc_input_name = sess_enc.get_inputs()[0].name
    enc_out_names = [o.name for o in sess_enc.get_outputs()]

    while True:
        ret, frame = cap.read()
        if not ret or (args.max_frames > 0 and fidx >= args.max_frames):
            break

        t_enc = time.time()
        if fidx == 0:
            enc_embed, enc_curr_feat, f0, f1 = embed0, curr_feat0, f0_0, f1_0
            vis_pos = None
            enc_ms = (time.time() - t_enc) * 1000
        else:
            tensor, _ = prepare_image(frame, (enc_h, enc_w))
            enc_vals = sess_enc.run(None, {enc_input_name: tensor})
            enc = dict(zip(enc_out_names, enc_vals))
            enc_embed = enc["image_embeddings"].astype(np.float32, copy=False)
            enc_curr_feat = enc.get("current_vision_feat", enc_embed).astype(np.float32, copy=False)
            f0 = enc["high_res_features1"].astype(np.float32, copy=False)
            f1 = enc["high_res_features2"].astype(np.float32, copy=False)
            vis_pos = enc["vision_pos_embed"].astype(np.float32, copy=False)
            enc_ms = (time.time() - t_enc) * 1000

        if fidx > 0:
            memory_1, memory_pos_embed = memory_bank.build_attention_state()
            object_ptrs, obj_ptr_offsets = memory_bank.build_object_pointer_state(frame_index=fidx)
        else:
            memory_1, memory_pos_embed = (None, None)
            object_ptrs, obj_ptr_offsets = (None, None)

        if fidx > 0 and memory_1 is not None and memory_pos_embed is not None:
            t_mat = time.time()
            fused_embed = run_memory_attention(
                sess_mat,
                current_vision_feat=enc_curr_feat,
                current_vision_pos_embed=vis_pos,
                memory_1=memory_1,
                memory_pos_embed=memory_pos_embed,
                memory_0=object_ptrs,
                obj_ptr_offsets=obj_ptr_offsets,
            ).astype(np.float32, copy=False)
            mat_ms = (time.time() - t_mat) * 1000
        else:
            fused_embed = enc_embed
            mat_ms = 0.0

        t_dec = time.time()
        if fidx == 0:
            obj_ptr, mask_for_mem, pred_mask = run_decoder(sess_dec0, pts0, lbls0, fused_embed, f0, f1)
        else:
            obj_ptr, mask_for_mem, pred_mask = run_decoder(sess_decn, None, None, fused_embed, f0, f1)
        dec_ms = (time.time() - t_dec) * 1000

        if mask_for_mem is None or pred_mask is None:
            sys.exit("ERROR: Decoder did not return the expected outputs.")

        t_men = time.time()
        mem_feats, mem_pos, temporal_code = run_memory_encoder(
            sess_men,
            mask_for_mem=mask_for_mem[:, 0:1],
            pix_feat=fused_embed,
        )
        if fidx == 0:
            memory_bank.set_conditioning(mem_feats, mem_pos, temporal_code, obj_ptr=obj_ptr, frame_index=fidx)
        else:
            memory_bank.append_recent(mem_feats, mem_pos, temporal_code, obj_ptr=obj_ptr, frame_index=fidx)
        men_ms = (time.time() - t_men) * 1000

        logits = pred_mask[0, 0]
        mask_hi = cv2.resize(logits, (w_org, h_org), cv2.INTER_LINEAR)
        mask = (mask_hi > 0).astype(np.uint8)
        writer.write(green_overlay(frame, mask, 0.5))

        if fidx == 0:
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

    ap = argparse.ArgumentParser(description="Video segmentation demo for SAM-2 ONNX")
    ap.add_argument("--model_size", default="tiny", choices=["base_plus", "large", "small", "tiny"])
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
