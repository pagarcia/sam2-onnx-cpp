"""
onnx_test_image_bounding_box.py
────────────────────────────────────────────────────────────────────────────
Interactive demo for running SAM-2 ONNX on a single image with a bounding-box
prompt.

Left button down / drag : draw rectangle (yellow preview)
Left button up          : run segmentation once, show green overlay
Right button (or double-click) : clear rectangle
ESC                     : quit
"""

import os, sys, time, argparse
import cv2
import numpy as np
import onnxruntime as ort
from PyQt5 import QtWidgets
from onnxruntime import InferenceSession


# ───────────────────────── helper functions ──────────────────────────
def print_system_info() -> None:
    print("[INFO] OS :", sys.platform)
    print("[INFO] ONNX Runtime providers :", ort.get_available_providers())


def prepare_rectangle(rect, img_size, enc_size):
    """
    Convert (x1,y1,x2,y2) from original image scale → encoder-input scale.
    Produces two corner points with labels 2 and 3, batched as [1,2,*].
    """
    if rect is None:
        return None, None
    x1, y1, x2, y2 = rect
    h_org, w_org   = img_size
    h_enc, w_enc   = enc_size

    pts = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
    pts[:, 0] = (pts[:, 0] / w_org) * w_enc
    pts[:, 1] = (pts[:, 1] / h_org) * h_enc
    lbls = np.array([2.0, 3.0], dtype=np.float32)

    return pts[np.newaxis, ...], lbls[np.newaxis, ...]


# ───────────────────────────── main ──────────────────────────────────
def main() -> None:
    print_system_info()

    ap = argparse.ArgumentParser(description="SAM-2 ONNX bounding-box demo")
    ap.add_argument("--model_size",
                    default="tiny",
                    choices=["base_plus", "large", "small", "tiny"])
    args = ap.parse_args()

    # ── pick an image ────────────────────────────────────────────────
    app = QtWidgets.QApplication(sys.argv)
    img_path, _ = QtWidgets.QFileDialog.getOpenFileName(
        None, "Select an Image",
        "", "Images (*.jpg *.jpeg *.png *.bmp);;All files (*)"
    )
    if not img_path:
        sys.exit("No image selected – exiting.")
    print(f"[INFO] Selected image : {img_path}")

    # ── find ONNX models ─────────────────────────────────────────────
    ckpt_dir = os.path.join("checkpoints", args.model_size)
    enc_path = os.path.join(ckpt_dir, f"image_encoder_{args.model_size}.onnx")
    dec_path = os.path.join(ckpt_dir, f"image_decoder_{args.model_size}.onnx")
    if not (os.path.exists(enc_path) and os.path.exists(dec_path)):
        sys.exit(f"ERROR: ONNX files not found in {ckpt_dir}")

    # ── create sessions ──────────────────────────────────────────────
    sess_enc = InferenceSession(enc_path, providers=ort.get_available_providers())
    sess_dec = InferenceSession(dec_path, providers=ort.get_available_providers())
    enc_in_name   = sess_enc.get_inputs()[0].name
    enc_h, enc_w  = sess_enc.get_inputs()[0].shape[2:]
    dec_in_names  = [i.name for i in sess_dec.get_inputs()]
    dec_out_names = [o.name for o in sess_dec.get_outputs()]
    print(f"[INFO] Encoder input size : {(enc_h, enc_w)}")

    # ── load & preprocess image (float32!) ───────────────────────────
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        sys.exit("ERROR: Could not read image.")
    h_org, w_org = img_bgr.shape[:2]

    img_resized = cv2.resize(img_bgr, (enc_w, enc_h))
    img_rgb     = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    img_rgb     /= np.float32(255.0)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_rgb = (img_rgb - mean) / std
    input_tensor = np.transpose(img_rgb, (2,0,1))[np.newaxis, :].astype(np.float32)

    # ── run encoder once ─────────────────────────────────────────────
    t0 = time.time()
    enc_out = sess_enc.run(None, {enc_in_name: input_tensor})
    print(f"[INFO] Encoder time : {(time.time()-t0)*1000:.1f} ms")
    img_embed, feats0, feats1 = enc_out[:3]   # we only need first three outputs

    # ── display prep ────────────────────────────────────────────────
    disp_max = 1200
    scale    = min(1.0, disp_max / max(w_org, h_org))
    disp_w, disp_h = int(w_org*scale), int(h_org*scale)
    disp_base = cv2.resize(img_bgr, (disp_w, disp_h))

    rect_start = rect_end = None
    drawing    = False

    def show_preview():
        vis = disp_base.copy()
        if rect_start and rect_end:
            cv2.rectangle(vis, rect_start, rect_end, (0,255,255), 2)
        cv2.imshow("SAM2 Bounding-Box Demo", vis)

    def run_decoder():
        if rect_start is None or rect_end is None:
            show_preview()
            return

        # convert display coords → original coords
        x1_d,y1_d = rect_start
        x2_d,y2_d = rect_end
        x1,y1,x2,y2 = (int(x1_d/scale), int(y1_d/scale),
                       int(x2_d/scale), int(y2_d/scale))
        x1,x2 = sorted((x1,x2));  y1,y2 = sorted((y1,y2))

        pts,lbls = prepare_rectangle((x1,y1,x2,y2), (h_org,w_org), (enc_h,enc_w))

        dec_inputs = {
            dec_in_names[0]: pts,
            dec_in_names[1]: lbls,
            dec_in_names[2]: img_embed,
            dec_in_names[3]: feats0,
            dec_in_names[4]: feats1
        }

        t = time.time()
        _, _, low_masks = sess_dec.run(dec_out_names, dec_inputs)
        print(f"[INFO] Decoder time : {(time.time()-t)*1000:.1f} ms")

        mask256   = low_masks[0,0]
        mask_full = cv2.resize(mask256, (w_org,h_org), cv2.INTER_LINEAR)
        mask_bin  = (mask_full > 0).astype(np.uint8)*255

        color = np.zeros_like(img_bgr);  color[mask_bin==255] = (0,255,0)
        overlay = cv2.addWeighted(img_bgr, 1.0, color, 0.5, 0)

        disp = cv2.resize(overlay, (disp_w,disp_h))
        cv2.rectangle(disp, rect_start, rect_end, (0,255,255), 2)
        cv2.imshow("SAM2 Bounding-Box Demo", disp)

    # ── mouse callback ──────────────────────────────────────────────
    def mouse_cb(event,x,y,flags,param):
        nonlocal rect_start, rect_end, drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            rect_start = rect_end = (x,y)
            show_preview()
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            rect_end = (x,y)
            show_preview()                      # live preview, no model
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            rect_end = (x,y)
            run_decoder()                       # run model once
        elif event in (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_LBUTTONDBLCLK):
            rect_start = rect_end = None
            show_preview()                      # reset view

    cv2.namedWindow("SAM2 Bounding-Box Demo", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("SAM2 Bounding-Box Demo", mouse_cb)
    show_preview()

    print("[INFO] Drag to draw; right-click to reset; ESC to quit.")
    while True:
        if cv2.waitKey(20) & 0xFF == 27:  # ESC
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
