# sam2-onnx-cpp/export/onnx_test_video.py
#!/usr/bin/env python3
# CPU-optimized: encoder fast (BASIC/EXTENDED), decoder/memory safe.
import os
# Limit BLAS thread pools before NumPy loads.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import time
import argparse
import cv2
import numpy as np
from PyQt5 import QtWidgets

from onnx_test_utils import (
    print_system_info, set_cv2_threads,
    make_encoder_session, make_safe_session,
    prepare_image, prepare_points, prepare_box_prompt,
    run_encoder, run_decoder,
    prefer_quantized_encoder, green_overlay, compute_display_base,
)

def interactive_select_points(first_bgr, sess_enc, sess_dec, enc_shape):
    enc_h, enc_w = enc_shape
    tensor, (h_org, w_org) = prepare_image(first_bgr, (enc_h, enc_w))
    enc = run_encoder(sess_enc, tensor)
    embed = enc["image_embeddings"]; f0 = enc["high_res_features1"]; f1 = enc["high_res_features2"]

    base, scale = compute_display_base(first_bgr, max_side=1200)
    points, labels = [], []

    def show(mask=None):
        vis = base.copy()
        if mask is not None:
            m = cv2.resize(mask, (base.shape[1], base.shape[0]), cv2.INTER_NEAREST)
            vis = green_overlay(vis, m, 0.5)
        for i, (px, py) in enumerate(points):
            col = (0, 0, 255) if labels[i] == 1 else (255, 0, 0)
            cv2.circle(vis, (int(px*scale), int(py*scale)), 6, col, -1)
        cv2.imshow("First Frame – SAM-2", vis)

    def run():
        if not points:
            show(); return
        pts, lbls = prepare_points(points, labels, (h_org, w_org), (enc_h, enc_w))
        _, mask_hi, _ = run_decoder(sess_dec, pts, lbls, embed, f0, f1)
        show((mask_hi[0, 0] > 0).astype(np.uint8))

    def cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((int(x/scale), int(y/scale))); labels.append(1); run()
        elif event == cv2.EVENT_RBUTTONDOWN:
            points.append((int(x/scale), int(y/scale))); labels.append(0); run()
        elif event == cv2.EVENT_MBUTTONDOWN:
            points.clear(); labels.clear(); run()

    cv2.namedWindow("First Frame – SAM-2")
    cv2.setMouseCallback("First Frame – SAM-2", cb)
    run()
    print("[INFO] L-click=FG, R-click=BG, M-click=reset. ESC/Enter to continue.")
    while True:
        if cv2.waitKey(20) & 0xFF in (27, 13):
            break
    cv2.destroyAllWindows()
    return points, labels, embed, f0, f1, (h_org, w_org)


def interactive_select_box(first_bgr, sess_enc, sess_dec, enc_shape):
    enc_h, enc_w = enc_shape
    tensor, (h_org, w_org) = prepare_image(first_bgr, (enc_h, enc_w))
    enc = run_encoder(sess_enc, tensor)
    embed = enc["image_embeddings"]; f0 = enc["high_res_features1"]; f1 = enc["high_res_features2"]

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
        cv2.imshow("First Frame – SAM-2", vis)

    def run():
        if not(rect_s and rect_e):
            show(); return
        x1d, y1d = rect_s; x2d, y2d = rect_e
        x1, x2 = sorted((int(x1d/scale), int(x2d/scale)))
        y1, y2 = sorted((int(y1d/scale), int(y2d/scale)))
        pts, lbls = prepare_box_prompt((x1, y1, x2, y2), (h_org, w_org), (enc_h, enc_w))
        _, mask_hi, _ = run_decoder(sess_dec, pts, lbls, embed, f0, f1)
        show((mask_hi[0, 0] > 0).astype(np.uint8))

    def cb(event, x, y, flags, param):
        nonlocal rect_s, rect_e, drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True; rect_s = rect_e = (x, y); show()
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            rect_e = (x, y); show()
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False; rect_e = (x, y); run()
        elif event in (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_LBUTTONDBLCLK):
            rect_s = rect_e = None; show()

    cv2.namedWindow("First Frame – SAM-2")
    cv2.setMouseCallback("First Frame – SAM-2", cb)
    show()
    print("[INFO] Draw rectangle, release → preview. ESC/Enter to continue.")
    while True:
        if cv2.waitKey(20) & 0xFF in (27, 13):
            break
    cv2.destroyAllWindows()

    if rect_s and rect_e:
        x1d, y1d = rect_s; x2d, y2d = rect_e
        x1, x2 = sorted((int(x1d/scale), int(x2d/scale)))
        y1, y2 = sorted((int(y1d/scale), int(y2d/scale)))
        box = (x1, y1, x2, y2)
    else:
        box = None
    return box, embed, f0, f1, (h_org, w_org)


def process_video(args):
    ckpt_dir = os.path.join("checkpoints", args.model_size)
    enc_path = prefer_quantized_encoder(ckpt_dir)
    if enc_path is None:
        sys.exit(f"ERROR: Encoder ONNX not found in {ckpt_dir}")

    paths = lambda name: os.path.join(ckpt_dir, f"{name}.onnx")
    dec_path = paths("image_decoder")
    men_path = paths("memory_encoder")
    mat_path = paths("memory_attention")
    for p in [dec_path, men_path, mat_path]:
        if not os.path.exists(p):
            sys.exit(f"ERROR: ONNX file missing: {p}")

    sess_enc = make_encoder_session(enc_path)
    sess_dec = make_safe_session(dec_path, tag="decoder")
    sess_men = make_safe_session(men_path, tag="memory_encoder")
    sess_mat = make_safe_session(mat_path, tag="memory_attention")

    enc_h, enc_w = sess_enc.get_inputs()[0].shape[2:]
    print(f"[INFO] Encoder input = {(enc_h, enc_w)}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit("ERROR: cannot open video")
    fps   = cap.get(cv2.CAP_PROP_FPS)
    w_org = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_org = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = os.path.splitext(args.video)[0] + "_mask_overlay.mkv"
    writer = cv2.VideoWriter(out_path,
                             cv2.VideoWriter_fourcc(*'XVID'),
                             fps if fps > 0 else 25.0,
                             (w_org, h_org))
    if not writer.isOpened():
        sys.exit("ERROR: cannot open VideoWriter")

    ret, first_bgr = cap.read()
    if not ret:
        sys.exit("ERROR: empty video")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if args.prompt == "bounding_box":
        box, embed0, f0_0, f1_0, (h_org, w_org) = \
            interactive_select_box(first_bgr, sess_enc, sess_dec, (enc_h, enc_w))
        pts0, lbls0 = prepare_box_prompt(box, (h_org, w_org), (enc_h, enc_w)) if box else (None, None)
    else:
        pts, lbls, embed0, f0_0, f1_0, (h_org, w_org) = \
            interactive_select_points(first_bgr, sess_enc, sess_dec, (enc_h, enc_w))
        pts0, lbls0 = prepare_points(pts, lbls, (h_org, w_org), (enc_h, enc_w))

    embed0 = embed0.astype(np.float32, copy=False)
    f0_0   = f0_0.astype(np.float32, copy=False)
    f1_0   = f1_0.astype(np.float32, copy=False)

    fidx = 0
    mem_feats = mem_pos = None

    enc_input_name = sess_enc.get_inputs()[0].name
    enc_out_names = [o.name for o in sess_enc.get_outputs()]

    while True:
        ret, frame = cap.read()
        if not ret or (args.max_frames > 0 and fidx >= args.max_frames):
            break

        # Encoder
        t_enc = time.time()
        if fidx == 0:
            enc_embed, f0, f1 = embed0, f0_0, f1_0
            vis_pos = None
            enc_ms = (time.time() - t_enc) * 1000
        else:
            tensor, _ = prepare_image(frame, (enc_h, enc_w))
            enc_vals = sess_enc.run(None, {enc_input_name: tensor})
            enc = dict(zip(enc_out_names, enc_vals))
            enc_embed = enc["image_embeddings"].astype(np.float32, copy=False)
            f0        = enc["high_res_features1"].astype(np.float32, copy=False)
            f1        = enc["high_res_features2"].astype(np.float32, copy=False)
            vis_pos   = enc["vision_pos_embed"].astype(np.float32, copy=False)
            enc_ms = (time.time() - t_enc) * 1000

        # Memory attention (from 2nd frame)
        if fidx > 0 and mem_feats is not None:
            t_mat = time.time()
            attn_inputs = {
                "current_vision_feat":      enc_embed.astype(np.float32, copy=False),
                "current_vision_pos_embed": vis_pos.astype(np.float32, copy=False),
                "memory_0":                 np.zeros((0, 256), np.float32),  # no obj ptrs for now
                "memory_1":                 mem_feats.astype(np.float32, copy=False),
                "memory_pos_embed":         mem_pos.astype(np.float32, copy=False),
            }
            fused_embed = sess_mat.run(None, attn_inputs)[0].astype(np.float32, copy=False)
            mat_ms = (time.time() - t_mat) * 1000
        else:
            fused_embed = enc_embed
            mat_ms = 0.0

        # Decoder
        t_dec = time.time()
        if fidx == 0:
            _, mask_for_mem, _ = run_decoder(sess_dec, pts0, lbls0, fused_embed, f0, f1)
        else:
            _, mask_for_mem, _ = run_decoder(sess_dec, None, None, fused_embed, f0, f1)
        dec_ms = (time.time() - t_dec) * 1000

        # Memory encoder
        t_men = time.time()
        men_out = sess_men.run(None, {
            "mask_for_mem": mask_for_mem[:, 0:1].astype(np.float32, copy=False),
            "pix_feat":     fused_embed.astype(np.float32, copy=False),
        })
        mem_feats, mem_pos, _ = [x.astype(np.float32, copy=False) for x in men_out]
        men_ms = (time.time() - t_men) * 1000

        # Overlay & write
        logits  = mask_for_mem[0, 0]
        mask_hi = cv2.resize(logits, (w_org, h_org), cv2.INTER_LINEAR)
        mask    = (mask_hi > 0).astype(np.uint8)
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

    ap = argparse.ArgumentParser(description="Video segmentation demo for SAM-2 ONNX (CPU-optimized)")
    ap.add_argument("--model_size", default="tiny", choices=["base_plus", "large", "small", "tiny"])
    ap.add_argument("--prompt", default="seed_points", choices=["seed_points", "bounding_box"])
    ap.add_argument("--max_frames", type=int, default=0, help="Max frames to process (0 = all).")
    args = ap.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    vid, _ = QtWidgets.QFileDialog.getOpenFileName(
        None, "Select Video", "",
        "Video files (*.mp4 *.mkv *.avi *.mov *.m4v);;All files (*.*)")
    if not vid:
        sys.exit("No video selected – exiting.")
    args.video = vid
    print(f"[INFO] Selected video: {vid}")

    process_video(args)

if __name__ == "__main__":
    main()
