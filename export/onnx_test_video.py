#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────────────
#  onnx_test_video.py
#  Unified interactive SAM-2 ONNX demo on videos with either
#    • seed_points   (positive / negative clicks)
#    • bounding_box  (click-and-drag rectangle)
#  Choose with  --prompt  seed_points | bounding_box   (default = seed_points)
# ──────────────────────────────────────────────────────────────────────────────
"""
Controls on the first frame
───────────────────────────
seed_points mode
    L-click : add foreground point   (red)
    R-click : add background point   (blue)
    M-click : reset all points

bounding_box mode
    L-drag        : draw rectangle preview (yellow)
    L-release     : run segmentation once
    R-click / dbl : reset rectangle

Common
    ESC / Enter : accept the prompt and start processing
"""
import os, sys, time, argparse
import cv2
import numpy as np
import onnxruntime as ort
from onnxruntime import InferenceSession
from PyQt5 import QtWidgets

# ─────────────────────────── utilities ────────────────────────────
def print_system_info() -> None:
    print("[INFO] OS :", sys.platform)
    print("[INFO] ONNX Runtime providers :", ort.get_available_providers())

def prepare_image(frame_bgr: np.ndarray, enc_shape: tuple[int, int]
                  ) -> tuple[np.ndarray, tuple[int, int]]:
    """BGR → pre-processed tensor [1,3,H,W] + original (H,W)."""
    h_enc, w_enc = enc_shape
    rgb = cv2.cvtColor(cv2.resize(frame_bgr, (w_enc, h_enc)),
                       cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.array([0.485, 0.456, 0.406], np.float32)) / \
          np.array([0.229, 0.224, 0.225], np.float32)
    tensor = np.transpose(rgb, (2, 0, 1))[np.newaxis, :]
    return tensor, frame_bgr.shape[:2]      # original (H,W)

def prepare_points(points, labels, img_sz, enc_sz):
    if not points:           # None or empty list
        return None, None
    pts  = np.asarray(points, dtype=np.float32)
    lbls = np.asarray(labels, dtype=np.float32)
    h_org, w_org = img_sz
    h_enc, w_enc = enc_sz
    pts[:, 0] = (pts[:, 0] / w_org) * w_enc
    pts[:, 1] = (pts[:, 1] / h_org) * h_enc
    return pts[np.newaxis, ...], lbls[np.newaxis, ...]

def prepare_box_prompt(rect, img_sz, enc_sz):
    if rect is None:
        return None, None
    x1, y1, x2, y2 = rect
    h_org, w_org = img_sz
    h_enc, w_enc = enc_sz
    pts = np.array([[x1, y1], [x2, y2]], np.float32)
    pts[:, 0] = (pts[:, 0] / w_org) * w_enc
    pts[:, 1] = (pts[:, 1] / h_org) * h_enc
    lbls = np.array([2.0, 3.0], np.float32)          # SAM-2 bbox labels
    return pts[np.newaxis, ...], lbls[np.newaxis, ...]

def decode(sess_dec, coords, labels, embed, f0, f1):
    """Convenience wrapper that handles the empty-prompt case."""
    if coords is None:
        coords = np.zeros((1, 0, 2), np.float32)
        labels = np.zeros((1, 0),   np.float32)
    return sess_dec.run(None, {
        "point_coords":     coords,
        "point_labels":     labels,
        "image_embed":      embed,
        "high_res_feats_0": f0,
        "high_res_feats_1": f1})

def green_overlay(bgr, mask255, alpha=0.5):
    out = bgr.copy()
    green = np.zeros_like(out); green[mask255 > 0] = (0, 255, 0)
    return cv2.addWeighted(out, 1.0, green, alpha, 0)

# ─────────────── interactive prompt – seed points ────────────────
def interactive_select_points(first_bgr, sess_enc, sess_dec, enc_shape):
    tensor, (h_org, w_org) = prepare_image(first_bgr, enc_shape)
    embed, f0, f1 = sess_enc.run(None, {sess_enc.get_inputs()[0].name: tensor})[:3]

    disp_max = 1200
    scale = min(1.0, disp_max / max(w_org, h_org))
    disp_w, disp_h = int(w_org * scale), int(h_org * scale)
    base = cv2.resize(first_bgr, (disp_w, disp_h))

    points, labels = [], []

    def show(mask=None):
        vis = base.copy()
        if mask is not None:
            m = cv2.resize(mask, (disp_w, disp_h), cv2.INTER_NEAREST)
            vis = green_overlay(vis, m, 0.5)
        for i, (px, py) in enumerate(points):
            col = (0, 0, 255) if labels[i] == 1 else (255, 0, 0)
            cv2.circle(vis, (int(px * scale), int(py * scale)), 6, col, -1)
        cv2.imshow("First Frame – SAM-2", vis)

    def run():
        if not points: show(); return
        pts, lbls = prepare_points(points, labels, (h_org, w_org), enc_shape)
        _, _, pred = decode(sess_dec, pts, lbls, embed, f0, f1)
        show((pred[0, 0] > 0).astype(np.uint8))

    def cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((int(x / scale), int(y / scale))); labels.append(1); run()
        elif event == cv2.EVENT_RBUTTONDOWN:
            points.append((int(x / scale), int(y / scale))); labels.append(0); run()
        elif event == cv2.EVENT_MBUTTONDOWN:
            points.clear(); labels.clear(); run()

    cv2.namedWindow("First Frame – SAM-2"); cv2.setMouseCallback("First Frame – SAM-2", cb)
    run(); print("[INFO] L-click=FG, R-click=BG, M-click=reset. ESC/Enter to continue.")
    while True:
        if cv2.waitKey(20) & 0xFF in (27, 13): break
    cv2.destroyAllWindows()
    return points, labels, embed, f0, f1, (h_org, w_org)

# ─────────── interactive prompt – bounding box ────────────
def interactive_select_box(first_bgr, sess_enc, sess_dec, enc_shape):
    tensor, (h_org, w_org) = prepare_image(first_bgr, enc_shape)
    embed, f0, f1 = sess_enc.run(None, {sess_enc.get_inputs()[0].name: tensor})[:3]

    disp_max = 1200
    scale = min(1.0, disp_max / max(w_org, h_org))
    disp_w, disp_h = int(w_org * scale), int(h_org * scale)
    base = cv2.resize(first_bgr, (disp_w, disp_h))

    rect_s = rect_e = None
    drawing = False

    def show(mask=None):
        vis = base.copy()
        if mask is not None:
            m = cv2.resize(mask, (disp_w, disp_h), cv2.INTER_NEAREST)
            vis = green_overlay(vis, m, 0.5)
        if rect_s and rect_e:
            cv2.rectangle(vis, rect_s, rect_e, (0, 255, 255), 2)
        cv2.imshow("First Frame – SAM-2", vis)

    def run():
        if not(rect_s and rect_e): show(); return
        x1d, y1d = rect_s; x2d, y2d = rect_e
        x1, x2 = sorted((int(x1d / scale), int(x2d / scale)))
        y1, y2 = sorted((int(y1d / scale), int(y2d / scale)))
        pts, lbls = prepare_box_prompt((x1, y1, x2, y2), (h_org, w_org), enc_shape)
        _, _, pred = decode(sess_dec, pts, lbls, embed, f0, f1)
        show((pred[0, 0] > 0).astype(np.uint8))

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

    cv2.namedWindow("First Frame – SAM-2"); cv2.setMouseCallback("First Frame – SAM-2", cb)
    show(); print("[INFO] Draw rectangle, release → preview. ESC/Enter to continue.")
    while True:
        if cv2.waitKey(20) & 0xFF in (27, 13): break
    cv2.destroyAllWindows()

    if rect_s and rect_e:
        x1d, y1d = rect_s; x2d, y2d = rect_e
        x1, x2 = sorted((int(x1d / scale), int(x2d / scale)))
        y1, y2 = sorted((int(y1d / scale), int(y2d / scale)))
        box = (x1, y1, x2, y2)
    else:
        box = None
    return box, embed, f0, f1, (h_org, w_org)

# ─────────────────────── main processing loop ───────────────────────
def process_video(args):
    ckpt_dir = os.path.join("checkpoints", args.model_size)
    paths = lambda name: os.path.join(ckpt_dir, f"{name}_{args.model_size}.onnx")
    sess_enc = InferenceSession(paths("image_encoder"), providers=ort.get_available_providers())
    sess_dec = InferenceSession(paths("image_decoder"), providers=ort.get_available_providers())
    sess_men = InferenceSession(paths("memory_encoder"),  providers=ort.get_available_providers())
    sess_mat = InferenceSession(paths("memory_attention"),providers=ort.get_available_providers())

    enc_h, enc_w = sess_enc.get_inputs()[0].shape[2:]
    print(f"[INFO] Encoder input = {(enc_h, enc_w)}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened(): sys.exit("ERROR: cannot open video")
    fps   = cap.get(cv2.CAP_PROP_FPS)
    w_org = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_org = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = os.path.splitext(args.video)[0] + "_mask_overlay.mkv"
    writer = cv2.VideoWriter(out_path,
                             cv2.VideoWriter_fourcc(*'XVID'),
                             fps, (w_org, h_org))
    if not writer.isOpened(): sys.exit("ERROR: cannot open VideoWriter")

    # ── prompt on first frame ─────────────────────────────────────
    ret, first_bgr = cap.read()
    if not ret: sys.exit("ERROR: empty video")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if args.prompt == "bounding_box":
        box, embed0, f0_0, f1_0, (h_org, w_org) = \
            interactive_select_box(first_bgr, sess_enc, sess_dec, (enc_h, enc_w))
        pts0, lbls0 = prepare_box_prompt(box, (h_org, w_org), (enc_h, enc_w)) \
                      if box else (None, None)
    else:  # seed_points
        pts, lbls, embed0, f0_0, f1_0, (h_org, w_org) = \
            interactive_select_points(first_bgr, sess_enc, sess_dec, (enc_h, enc_w))
        pts0, lbls0 = prepare_points(pts, lbls, (h_org, w_org), (enc_h, enc_w))

    mem_feats = mem_pos = None
    fidx = 0
    while True:
        ret, frame = cap.read()
        if not ret or (args.max_frames > 0 and fidx >= args.max_frames): break

        # ── Image encoder ─────────────────────────────────────────
        t_enc = time.time()
        if fidx == 0:
            embed, f0, f1 = embed0, f0_0, f1_0
            enc_ms = (time.time() - t_enc)*1000
            vis_pos = None  # not used on frame 0
        else:
            tensor, _ = prepare_image(frame, (enc_h, enc_w))
            enc_out = sess_enc.run(None, {sess_enc.get_inputs()[0].name: tensor})
            embed, f0, f1, _, vis_pos = enc_out
            enc_ms = (time.time() - t_enc)*1000

        # ── Memory attention (from 2nd frame) ─────────────────────
        if fidx > 0 and mem_feats is not None:
            t_mat = time.time()
            attn_inputs = {
                "current_vision_feat":      embed,
                "current_vision_pos_embed": vis_pos,
                "memory_0":                np.zeros((0,256), np.float32),
                "memory_1":                mem_feats,
                "memory_pos_embed":        mem_pos}
            embed = sess_mat.run(None, attn_inputs)[0]
            mat_ms = (time.time() - t_mat)*1000
        else:
            mat_ms = 0.0

        # ── Decoder ───────────────────────────────────────────────
        t_dec = time.time()
        if fidx == 0:
            _, mask_for_mem, pred = decode(sess_dec, pts0, lbls0, embed, f0, f1)
        else:
            _, mask_for_mem, pred = decode(sess_dec, None, None, embed, f0, f1)
        dec_ms = (time.time() - t_dec)*1000

        # ── Memory encoder ────────────────────────────────────────
        t_men = time.time()
        men_out = sess_men.run(None, {
            "mask_for_mem": mask_for_mem[:, 0:1],  # only first mask
            "pix_feat":     embed})
        mem_feats, mem_pos, _ = men_out
        men_ms = (time.time() - t_men)*1000

        # ── Overlay & write ───────────────────────────────────────
        mask = cv2.resize((pred[0, 0] > 0).astype(np.uint8),
                          (w_org, h_org), cv2.INTER_LINEAR)
        writer.write(green_overlay(frame, mask, 0.5))

        # ── log timings ───────────────────────────────────────────
        if fidx == 0:
            print(f"Frame {fidx:03d} - Enc:{enc_ms:.1f} | Dec:{dec_ms:.1f} | MemEnc:{men_ms:.1f}")
        else:
            print(f"Frame {fidx:03d} - Enc:{enc_ms:.1f} | Attn:{mat_ms:.1f} | "
                  f"Dec:{dec_ms:.1f} | MemEnc:{men_ms:.1f}")
        fidx += 1

    cap.release(); writer.release()
    print(f"Done! Wrote {fidx} frames with overlays to {out_path}")

# ────────────────────────── CLI wrapper ───────────────────────────
def main():
    print_system_info()
    ap = argparse.ArgumentParser(
        description="Unified video segmentation demo for SAM-2 ONNX "
                    "using either seed points or a bounding box prompt.")
    ap.add_argument("--model_size", default="tiny",
                    choices=["base_plus", "large", "small", "tiny"])
    ap.add_argument("--prompt", default="seed_points",
                    choices=["seed_points", "bounding_box"],
                    help="Interaction mode for the first frame.")
    ap.add_argument("--max_frames", type=int, default=0,
                    help="Max frames to process (0 = all).")
    args = ap.parse_args()

    # Select video through a Qt file dialog every time.
    app = QtWidgets.QApplication(sys.argv)
    vid, _ = QtWidgets.QFileDialog.getOpenFileName(
        None, "Select Video", "",
        "Video files (*.mp4 *.mkv *.avi *.mov *.m4v);;All files (*.*)")
    if not vid: sys.exit("No video selected – exiting.")
    args.video = vid
    print(f"[INFO] Selected video: {vid}")

    process_video(args)

if __name__ == "__main__":
    main()
