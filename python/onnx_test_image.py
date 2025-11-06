# sam2-onnx-cpp/python/onnx_test_image.py
#!/usr/bin/env python3
# CPU-optimized: encoder fast (EXTENDED + prepacking), decoder safe (no risky fusion).
import os
# Limit BLAS thread pools before NumPy loads (helps avoid oversubscription).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from pathlib import Path
import sys
import time
import argparse
import cv2
import numpy as np
from PyQt5 import QtWidgets

from onnx_test_utils import (
    print_system_info, set_cv2_threads,
    make_encoder_session, make_safe_session,
    prepare_image, prepare_points, prepare_rectangle,
    run_encoder, run_decoder,
    prefer_quantized_encoder, compute_display_base, green_overlay,
)

def main():
    print_system_info()
    set_cv2_threads(1)

    ap = argparse.ArgumentParser(description="SAM-2 ONNX (seed-points / bounding-box) – CPU optimized")
    ap.add_argument("--model_size", default="tiny", choices=["base_plus","large","small","tiny"])
    ap.add_argument("--prompt", default="seed_points", choices=["seed_points","bounding_box"])
    args = ap.parse_args()
    mode_bbox = args.prompt == "bounding_box"
    print(f"[INFO] Prompt mode : {'bounding_box' if mode_bbox else 'seed_points'}")

    app = QtWidgets.QApplication(sys.argv)
    img_path, _ = QtWidgets.QFileDialog.getOpenFileName(
        None, "Select an Image", "", "Images (*.jpg *.jpeg *.png *.bmp);;All files (*)")
    if not img_path:
        sys.exit("No image selected – exiting.")
    print(f"[INFO] Selected image : {img_path}")

    # Resolve repo root (one level up from this file)
    REPO_ROOT = Path(__file__).resolve().parent.parent
    ckpt_dir = REPO_ROOT / "checkpoints" / args.model_size
    enc_path = prefer_quantized_encoder(str(ckpt_dir))
    if enc_path is None:
        sys.exit(f"ERROR: Encoder ONNX not found in {ckpt_dir}")

    dec_path = str(ckpt_dir / "image_decoder.onnx")
    if not os.path.exists(dec_path):
        sys.exit(f"ERROR: Decoder ONNX not found in {ckpt_dir}")

    sess_enc = make_encoder_session(enc_path)
    sess_dec = make_safe_session(dec_path, tag="decoder")

    enc_h, enc_w = sess_enc.get_inputs()[0].shape[2:]
    print(f"[INFO] Encoder input size : {(enc_h, enc_w)}")

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        sys.exit("ERROR: Could not read image.")
    H_org, W_org = img_bgr.shape[:2]

    # Preprocess + encode
    inp_tensor, _ = prepare_image(img_bgr, (enc_h, enc_w))
    t0 = time.time()
    enc_dict = run_encoder(sess_enc, inp_tensor)
    print(f"[INFO] Encoder time : {(time.time()-t0)*1000:.1f} ms")

    img_embed = enc_dict["image_embeddings"].astype(np.float32, copy=False)
    feats0    = enc_dict["high_res_features1"].astype(np.float32, copy=False)
    feats1    = enc_dict["high_res_features2"].astype(np.float32, copy=False)

    # Display base
    disp_base, scale = compute_display_base(img_bgr, max_side=1200)

    points, labels = [], []
    rect_start = rect_end = None
    drawing = False

    def run_decoder_points():
        if not points:
            cv2.imshow("SAM-2 Demo", disp_base); return
        pts, lbl = prepare_points(points, labels, (H_org, W_org), (enc_h, enc_w))
        t = time.time()
        _, _, pred_low = run_decoder(sess_dec, pts, lbl, img_embed, feats0, feats1)
        print(f"[INFO] Decoder time : {(time.time()-t)*1000:.1f} ms")
        mask256 = pred_low[0, 0]
        mask = cv2.resize(mask256, (W_org, H_org))
        mask255 = (mask > 0).astype(np.uint8) * 255
        overlay = green_overlay(img_bgr, mask255)
        vis = overlay.copy()
        for i, (px, py) in enumerate(points):
            col = (0, 0, 255) if labels[i] == 1 else (255, 0, 0)
            cv2.circle(vis, (px, py), 6, col, -1)
        cv2.imshow("SAM-2 Demo", cv2.resize(vis, (disp_base.shape[1], disp_base.shape[0])))

    def run_decoder_box():
        if rect_start is None or rect_end is None:
            return
        x1d, y1d = rect_start; x2d, y2d = rect_end
        x1, y1 = int(x1d/scale), int(y1d/scale)
        x2, y2 = int(x2d/scale), int(y2d/scale)
        x1, x2 = sorted((x1, x2)); y1, y2 = sorted((y1, y2))
        pts, lbl = prepare_rectangle((x1, y1, x2, y2), (H_org, W_org), (enc_h, enc_w))
        t = time.time()
        _, _, pred_low = run_decoder(sess_dec, pts, lbl, img_embed, feats0, feats1)
        print(f"[INFO] Decoder time : {(time.time()-t)*1000:.1f} ms")
        mask256 = pred_low[0, 0]
        mask = cv2.resize(mask256, (W_org, H_org))
        mask255 = (mask > 0).astype(np.uint8) * 255
        overlay = green_overlay(img_bgr, mask255)
        disp = cv2.resize(overlay, (disp_base.shape[1], disp_base.shape[0]))
        cv2.rectangle(disp, rect_start, rect_end, (0, 255, 255), 2)
        cv2.imshow("SAM-2 Demo", disp)

    def mouse_cb(event, x, y, flags, param):
        nonlocal rect_start, rect_end, drawing
        if not mode_bbox:
            if event == cv2.EVENT_MBUTTONDOWN:
                points.clear(); labels.clear(); cv2.imshow("SAM-2 Demo", disp_base)
            elif event == cv2.EVENT_LBUTTONDOWN:
                points.append((int(x/scale), int(y/scale))); labels.append(1); run_decoder_points()
            elif event == cv2.EVENT_RBUTTONDOWN:
                points.append((int(x/scale), int(y/scale))); labels.append(0); run_decoder_points()
            return
        # bbox mode
        if event in (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_LBUTTONDBLCLK):
            rect_start = rect_end = None; cv2.imshow("SAM-2 Demo", disp_base); return
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True; rect_start = rect_end = (x, y); cv2.imshow("SAM-2 Demo", disp_base)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            rect_end = (x, y)
            vis = disp_base.copy()
            cv2.rectangle(vis, rect_start, rect_end, (0, 255, 255), 2)
            cv2.imshow("SAM-2 Demo", vis)
        elif event == cv2.EVENT_LBUTTONUP and drawing:
            drawing = False; rect_end = (x, y); run_decoder_box()

    cv2.namedWindow("SAM-2 Demo", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("SAM-2 Demo", mouse_cb)
    cv2.imshow("SAM-2 Demo", disp_base)
    print("[INFO] Interactive mode ready.  ESC to quit.")
    while True:
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
