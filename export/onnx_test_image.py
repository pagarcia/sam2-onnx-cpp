#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────────────
#  onnx_test_image.py
#  Unified interactive demo for SAM-2 ONNX using either
#    • seed-points       (positive / negative clicks)
#    • bounding-box      (click-and-drag rectangle)
#  Choose with  --prompt  seed_points | bounding_box   (default = seed_points)
# ──────────────────────────────────────────────────────────────────────────────
"""
Controls
────────
seed_points mode
    L-click  : add foreground point   (red)
    R-click  : add background point   (blue)
    M-click  : reset all points

bounding_box mode
    L-drag   : draw rectangle preview (yellow)
    L-release: run segmentation once
    R-click / double-click : reset rectangle

Common
    ESC      : quit
"""

import os, sys, time, argparse
import cv2, numpy as np
import onnxruntime as ort
from onnxruntime import InferenceSession
from PyQt5 import QtWidgets


# ─────────────────────────── utilities ────────────────────────────
def print_system_info():
    print("[INFO] OS :", sys.platform)
    print("[INFO] ONNX Runtime providers :", ort.get_available_providers())


def prepare_points(points, labels, img_size, enc_size):
    if not points:
        return None, None
    pts = np.asarray(points,  dtype=np.float32)        # [N,2]
    lbl = np.asarray(labels,  dtype=np.float32)        # [N]
    H_org, W_org = img_size
    H_enc, W_enc = enc_size
    pts[:, 0] = (pts[:, 0] / W_org) * W_enc   # x-coords
    pts[:, 1] = (pts[:, 1] / H_org) * H_enc   # y-coords
    return pts[np.newaxis, ...], lbl[np.newaxis, ...]  # add batch-dim


def prepare_rectangle(rect, img_size, enc_size):
    if rect is None:
        return None, None
    x1, y1, x2, y2 = rect
    H_org, W_org = img_size
    H_enc, W_enc = enc_size
    pts = np.array([[x1, y1], [x2, y2]], np.float32)
    pts[:, 0] = (pts[:, 0] / W_org) * W_enc
    pts[:, 1] = (pts[:, 1] / H_org) * H_enc
    lbl = np.array([2.0, 3.0], np.float32)             # SAM-2 bbox labels
    return pts[np.newaxis, ...], lbl[np.newaxis, ...]


def green_overlay(bgr, mask255, alpha=.5):
    """Return bright-green overlay version of bgr where mask255!=0."""
    color   = np.zeros_like(bgr);  color[mask255==255] = (0,255,0)
    return cv2.addWeighted(bgr, 1.0, color, alpha, 0)


# ───────────────────────── main entry ─────────────────────────────
def main():
    print_system_info()

    ap = argparse.ArgumentParser(
        description="Unified SAM-2 ONNX demo (seed-points / bounding-box)")
    ap.add_argument("--model_size",
                    default="tiny",
                    choices=["base_plus", "large", "small", "tiny"],
                    help="Which SAM-2 checkpoint size to use.")
    ap.add_argument("--prompt",
                    default="seed_points",
                    choices=["seed_points", "bounding_box"],
                    help="Interaction mode (default: seed_points)")
    args = ap.parse_args()
    mode_bbox = args.prompt == "bounding_box"
    print(f"[INFO] Prompt mode : {'bounding_box' if mode_bbox else 'seed_points'}")

    # ── choose an image via Qt dialog ──────────────────────────────
    app = QtWidgets.QApplication(sys.argv)
    img_path, _ = QtWidgets.QFileDialog.getOpenFileName(
        None, "Select an Image",
        "", "Images (*.jpg *.jpeg *.png *.bmp);;All files (*)")
    if not img_path:
        sys.exit("No image selected – exiting.")
    print(f"[INFO] Selected image : {img_path}")

    # ── locate models ──────────────────────────────────────────────
    ckpt_dir   = os.path.join("checkpoints", args.model_size)
    enc_path   = os.path.join(ckpt_dir, f"image_encoder_{args.model_size}.onnx")
    dec_path   = os.path.join(ckpt_dir, f"image_decoder_{args.model_size}.onnx")
    if not (os.path.exists(enc_path) and os.path.exists(dec_path)):
        sys.exit(f"ERROR: ONNX files not found in {ckpt_dir}")

    # ── create sessions ────────────────────────────────────────────
    sess_enc = InferenceSession(enc_path, providers=ort.get_available_providers())
    sess_dec = InferenceSession(dec_path, providers=ort.get_available_providers())
    enc_in   = sess_enc.get_inputs()[0]
    enc_h, enc_w = enc_in.shape[2:]
    dec_in_names  = [x.name for x in sess_dec.get_inputs()]
    dec_out_names = [x.name for x in sess_dec.get_outputs()]
    print(f"[INFO] Encoder input size : {(enc_h, enc_w)}")

    # ── load & preprocess image ────────────────────────────────────
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        sys.exit("ERROR: Could not read image.")
    H_org, W_org = img_bgr.shape[:2]

    img_resized = cv2.resize(img_bgr, (enc_w, enc_h))
    img_rgb     = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    mean = np.array([0.485,0.456,0.406], np.float32)
    std  = np.array([0.229,0.224,0.225], np.float32)
    img_rgb     = (img_rgb - mean) / std
    inp_tensor  = np.transpose(img_rgb, (2,0,1))[np.newaxis, :]

    # ── run encoder once ───────────────────────────────────────────
    t0 = time.time()
    enc_out = sess_enc.run(None, {enc_in.name: inp_tensor})
    print(f"[INFO] Encoder time : {(time.time()-t0)*1000:.1f} ms")
    img_embed, feats0, feats1 = enc_out[:3]

    # ── common display scaling ─────────────────────────────────────
    disp_max = 1200
    scale    = min(1.0, disp_max / max(W_org, H_org))
    disp_w, disp_h = int(W_org*scale), int(H_org*scale)
    disp_base = cv2.resize(img_bgr, (disp_w, disp_h))

    # ───────────────────────── seed-points mode ────────────────────
    points, labels = [], []

    def run_decoder_points():
        if not points:
            cv2.imshow("SAM-2 Demo", disp_base); return

        pts, lbl = prepare_points(points, labels, (H_org,W_org), (enc_h,enc_w))
        dec_inputs = {
            dec_in_names[0]: pts,   # point_coords
            dec_in_names[1]: lbl,   # point_labels
            dec_in_names[2]: img_embed,
            dec_in_names[3]: feats0,
            dec_in_names[4]: feats1}
        t = time.time()
        _, _, low_masks = sess_dec.run(dec_out_names, dec_inputs)
        print(f"[INFO] Decoder time : {(time.time()-t)*1000:.1f} ms")
        mask256 = low_masks[0,0]
        mask    = cv2.resize(mask256, (W_org,H_org))
        mask255 = (mask>0).astype(np.uint8)*255
        overlay = green_overlay(img_bgr, mask255)
        for i,(px,py) in enumerate(points):
            col = (0,0,255) if labels[i]==1 else (255,0,0)
            cv2.circle(overlay,(px,py),6,col,-1)
        cv2.imshow("SAM-2 Demo", cv2.resize(overlay,(disp_w,disp_h)))

    # ───────────────────────── bounding-box mode ───────────────────
    rect_start = rect_end = None
    drawing    = False

    def run_decoder_box():
        if rect_start is None or rect_end is None: return
        x1_d,y1_d = rect_start
        x2_d,y2_d = rect_end
        x1,y1 = int(x1_d/scale), int(y1_d/scale)
        x2,y2 = int(x2_d/scale), int(y2_d/scale)
        x1,x2 = sorted((x1,x2));  y1,y2 = sorted((y1,y2))
        pts,lbl= prepare_rectangle((x1,y1,x2,y2),(H_org,W_org),(enc_h,enc_w))
        dec_inputs = {
            dec_in_names[0]: pts,
            dec_in_names[1]: lbl,
            dec_in_names[2]: img_embed,
            dec_in_names[3]: feats0,
            dec_in_names[4]: feats1}
        t=time.time()
        _,_,low_masks = sess_dec.run(dec_out_names, dec_inputs)
        print(f"[INFO] Decoder time : {(time.time()-t)*1000:.1f} ms")
        mask256 = low_masks[0,0]
        mask    = cv2.resize(mask256,(W_org,H_org))
        mask255 = (mask>0).astype(np.uint8)*255
        overlay = green_overlay(img_bgr, mask255)
        disp    = cv2.resize(overlay,(disp_w,disp_h))
        cv2.rectangle(disp, rect_start, rect_end,(0,255,255),2)
        cv2.imshow("SAM-2 Demo", disp)

    # ───────────────────────── unified mouse callback ───────────────
    def mouse_cb(event,x,y,flags,param):
        nonlocal rect_start, rect_end, drawing
        # ------ seed points ------
        if not mode_bbox:
            if event==cv2.EVENT_MBUTTONDOWN:
                points.clear(); labels.clear(); cv2.imshow("SAM-2 Demo", disp_base)
            elif event==cv2.EVENT_LBUTTONDOWN:
                points.append((int(x/scale), int(y/scale))); labels.append(1); run_decoder_points()
            elif event==cv2.EVENT_RBUTTONDOWN:
                points.append((int(x/scale), int(y/scale))); labels.append(0); run_decoder_points()
            return

        # ------ bounding box -----
        if event in (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_LBUTTONDBLCLK):
            rect_start = rect_end = None; cv2.imshow("SAM-2 Demo", disp_base); return
        if event==cv2.EVENT_LBUTTONDOWN:
            drawing=True; rect_start=rect_end=(x,y); cv2.imshow("SAM-2 Demo", disp_base)
        elif event==cv2.EVENT_MOUSEMOVE and drawing:
            rect_end=(x,y); vis=disp_base.copy()
            cv2.rectangle(vis,rect_start,rect_end,(0,255,255),2); cv2.imshow("SAM-2 Demo",vis)
        elif event==cv2.EVENT_LBUTTONUP and drawing:
            drawing=False; rect_end=(x,y); run_decoder_box()

    # ───────────────────────── run UI loop ──────────────────────────
    cv2.namedWindow("SAM-2 Demo", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("SAM-2 Demo", mouse_cb)
    cv2.imshow("SAM-2 Demo", disp_base)

    print("[INFO] Interactive mode ready.  ESC to quit.")
    while True:
        if cv2.waitKey(20) & 0xFF == 27:   # ESC
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
