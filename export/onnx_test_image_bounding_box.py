"""
onnx_test_image_bounding_box.py

Interactive demo for running SAM2 ONNX models on a single image using a *bounding-box*
prompt instead of point prompts.

Left-mouse-button  : click-and-drag to draw a rectangle (top-left ➜ bottom-right)
Right-mouse-button : reset / remove the rectangle
ESC                : quit
"""

import os
import sys
import time
import cv2
import argparse
import numpy as np
import onnxruntime
from onnxruntime import InferenceSession
from PyQt5 import QtWidgets


# --------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------
def print_system_info() -> None:
    providers = onnxruntime.get_available_providers()
    print("[INFO] OS :", sys.platform)
    print("[INFO] ONNX Runtime providers :", providers)


def prepare_rectangle(rect_coords, image_size, input_size):
    """
    Convert rectangle coordinates from original image scale ➜ encoder input scale.

    Args
    ----
    rect_coords : tuple(int,int,int,int)
        (x1, y1, x2, y2) in original image coordinates
    image_size  : (H_orig, W_orig)
    input_size  : (H_enc,  W_enc)

    Returns
    -------
    pts  : np.ndarray of shape [1, 2, 2]  (float32)
    lbls : np.ndarray of shape [1, 2]     (float32)
           First corner gets label 2, second gets label 3.
    """
    if rect_coords is None:
        return None, None

    x1, y1, x2, y2 = rect_coords
    H_orig, W_orig = image_size
    H_enc,  W_enc  = input_size

    # Two corner points
    pts = np.array([[x1, y1], [x2, y2]], dtype=np.float32)

    # Scale to encoder space
    pts[:, 0] = (pts[:, 0] / float(W_orig)) * W_enc
    pts[:, 1] = (pts[:, 1] / float(H_orig)) * H_enc

    # Labels 2 and 3 as required by SAM2
    lbls = np.array([2.0, 3.0], dtype=np.float32)

    # Add batch dimension
    pts  = pts[np.newaxis, ...]   # ➜ [1,2,2]
    lbls = lbls[np.newaxis, ...]  # ➜ [1,2]
    return pts, lbls


# --------------------------------------------------------------------------------
# Main demo logic
# --------------------------------------------------------------------------------
def main() -> None:
    print_system_info()

    parser = argparse.ArgumentParser(
        description="Interactive SAM2 ONNX demo using a *bounding-box* prompt."
    )
    parser.add_argument(
        "--model_size",
        default="tiny",
        choices=["base_plus", "large", "small", "tiny"],
        help="Which SAM2 model size to use."
    )
    args = parser.parse_args()

    # Pick an image file via a QFileDialog
    app = QtWidgets.QApplication(sys.argv)
    image_path, _ = QtWidgets.QFileDialog.getOpenFileName(
        None,
        "Select an Image",
        "",
        "Image files (*.jpg *.jpeg *.png *.bmp);;All files (*)"
    )
    if not image_path:
        print("No image selected – exiting.")
        sys.exit(0)
    args.image = image_path
    print(f"[INFO] Selected image : {image_path}")

    # -------------------------------------------------------------------------
    # Locate .onnx files
    # -------------------------------------------------------------------------
    outdir = os.path.join("checkpoints", args.model_size)
    encoder_path = os.path.join(outdir, f"image_encoder_{args.model_size}.onnx")
    decoder_path = os.path.join(outdir, f"image_decoder_{args.model_size}.onnx")
    if not os.path.exists(encoder_path) or not os.path.exists(decoder_path):
        sys.exit(f"ERROR: Could not find encoder/decoder ONNX files in {outdir}")

    # -------------------------------------------------------------------------
    # Create ONNX Runtime sessions
    # -------------------------------------------------------------------------
    session_encoder = InferenceSession(
        encoder_path,
        providers=onnxruntime.get_available_providers()
    )
    session_decoder = InferenceSession(
        decoder_path,
        providers=onnxruntime.get_available_providers()
    )
    enc_input_name  = session_encoder.get_inputs()[0].name
    enc_input_size  = session_encoder.get_inputs()[0].shape[2:]  # (H,W)
    dec_input_names = [i.name for i in session_decoder.get_inputs()]
    dec_output_names = [o.name for o in session_decoder.get_outputs()]

    print(f"[INFO] Encoder input size  : {enc_input_size}")
    print(f"[INFO] Decoder input names : {dec_input_names}")
    print(f"[INFO] Decoder output names: {dec_output_names}")

    # -------------------------------------------------------------------------
    # Load and preprocess the image once (resize ⇒ encoder-input size)
    # -------------------------------------------------------------------------
    original_bgr = cv2.imread(args.image)
    if original_bgr is None:
        sys.exit("ERROR: Could not read image.")
    H_orig, W_orig = original_bgr.shape[:2]
    resized = cv2.resize(original_bgr, (enc_input_size[1], enc_input_size[0]))
    resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    resized_rgb = (resized_rgb - mean) / std
    input_tensor = np.transpose(resized_rgb, (2,0,1))[np.newaxis, :]  # [1,3,H,W]

    # -------------------------------------------------------------------------
    # Run encoder once – we will reuse its outputs for every rectangle update
    # -------------------------------------------------------------------------
    print("[INFO] Running image encoder …")
    t0 = time.time()
    enc_outputs = session_encoder.run(
        None,
        {enc_input_name: input_tensor}
    )
    print(f"[INFO] Encoder time: {(time.time() - t0)*1000:.1f} ms")
    image_embed, feats0, feats1 = enc_outputs[:3]  # The first three outputs

    # -------------------------------------------------------------------------
    # Prepare interactive OpenCV window
    # -------------------------------------------------------------------------
    disp_max = 1200
    scale = min(1.0, disp_max / float(max(W_orig, H_orig)))
    disp_w, disp_h = int(W_orig*scale), int(H_orig*scale)
    display_copy = cv2.resize(original_bgr, (disp_w, disp_h))

    # Global mutable state for the rectangle
    rect_start = None      # (x,y) on display image
    rect_end   = None      # (x,y) on display image
    drawing    = False     # True while the mouse button is held

    def run_decoder():
        """
        Run the decoder with the current rectangle (if any) and show the result.
        """
        vis = display_copy.copy()

        if rect_start is None or rect_end is None:
            cv2.imshow("SAM2 Bounding-Box Demo", vis)
            return

        # Draw the rectangle being used
        cv2.rectangle(vis, rect_start, rect_end, (0,255,255), 2)

        # Convert coords back to original scale
        x1_d, y1_d = rect_start
        x2_d, y2_d = rect_end
        x1 = int(x1_d / scale)
        y1 = int(y1_d / scale)
        x2 = int(x2_d / scale)
        y2 = int(y2_d / scale)
        # Ensure TL↘BR ordering
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))

        pts, lbls = prepare_rectangle((x1,y1,x2,y2),
                                      (H_orig, W_orig),
                                      enc_input_size)
        dec_inputs = {
            dec_input_names[0]: pts,
            dec_input_names[1]: lbls,
            dec_input_names[2]: image_embed,
            dec_input_names[3]: feats0,
            dec_input_names[4]: feats1
        }

        t1 = time.time()
        _, _, low_res_masks = session_decoder.run(dec_output_names, dec_inputs)
        print(f"[INFO] Decoder time: {(time.time() - t1)*1000:.1f} ms")

        mask256 = low_res_masks[0,0]                                # [256,256]
        mask_full = cv2.resize(mask256, (W_orig, H_orig), cv2.INTER_LINEAR)
        mask_bin = (mask_full > 0).astype(np.uint8) * 255

        color = np.zeros_like(original_bgr)
        color[mask_bin==255] = (0,255,0)
        alpha = 0.5
        overlay = cv2.addWeighted(original_bgr, 1.0, color, alpha, 0)
        overlay_disp = cv2.resize(overlay, (disp_w, disp_h))
        # Re-draw rectangle & show
        cv2.rectangle(overlay_disp, rect_start, rect_end, (0,255,255), 2)
        cv2.imshow("SAM2 Bounding-Box Demo", overlay_disp)

    def mouse_cb(event, x, y, flags, param):
        nonlocal rect_start, rect_end, drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            rect_start = (x, y)
            rect_end   = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            rect_end = (x, y)
            run_decoder()        # preview while dragging
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            rect_end = (x, y)
            run_decoder()        # final rectangle
        elif event == cv2.EVENT_RBUTTONDOWN:
            # reset
            rect_start = rect_end = None
            run_decoder()

    cv2.namedWindow("SAM2 Bounding-Box Demo", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("SAM2 Bounding-Box Demo", mouse_cb)
    run_decoder()

    print("[INFO] Instructions:")
    print("  • Drag with left mouse button to draw a bounding box")
    print("  • Right-click to clear")
    print("  • Press ESC to quit")

    while True:
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
