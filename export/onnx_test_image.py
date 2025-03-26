import os
import sys
import time
import cv2
import numpy as np
import onnxruntime
from onnxruntime import InferenceSession
import argparse

# For the file dialog
try:
    import tkinter as tk
    from tkinter import filedialog
    HAVE_TK = True
except ImportError:
    HAVE_TK = False


def prepare_points(points_list, labels_list, image_size, input_size):
    """
    Convert 'points_list' (in original image scale) into
    normalized coords for the model input (input_size).

    Args:
        points_list (list of tuples): e.g. [(x1,y1), (x2,y2), ...]
        labels_list (list): e.g. [1, 1, 0, ...], same length
        image_size: (H_orig, W_orig) of the original image
        input_size: (H_in, W_in) that the encoder resized to (e.g. 1024,1024)

    Returns:
        pts:  shape [1, N, 2] in float32
        lbls: shape [1, N] in float32
    """
    if len(points_list) == 0:
        return None, None

    pts_array = np.array(points_list, dtype=np.float32)  # shape [N,2]
    lbls_array = np.array(labels_list, dtype=np.float32) # shape [N]

    H_orig, W_orig = image_size
    H_in, W_in     = input_size
    pts_array[:, 0] = (pts_array[:, 0] / float(W_orig)) * W_in  # x
    pts_array[:, 1] = (pts_array[:, 1] / float(H_orig)) * H_in  # y

    pts_array  = pts_array[np.newaxis, ...]  # => [1, N, 2]
    lbls_array = lbls_array[np.newaxis, ...] # => [1, N]
    return pts_array, lbls_array


def main():
    parser = argparse.ArgumentParser(description="Interactive SAM2 ONNX test with points prompt.")
    parser.add_argument(
        "--size_name",
        type=str,
        default="small",
        choices=["base_plus", "large", "small", "tiny"],
        help="Which model size to use: base_plus, large, small, or tiny."
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to the input image. If not provided, will open a file dialog."
    )
    args = parser.parse_args()

    # If no image is given, open a file selection dialog (requires Tk)
    if args.image is None:
        if not HAVE_TK:
            print("Tkinter is not installed, and no --image was provided.\n"
                  "Please install tkinter or specify --image.")
            sys.exit(1)
        else:
            root = tk.Tk()
            root.withdraw()  # Hide the main Tk window
            selected = filedialog.askopenfilename(
                title="Select an Image File",
                filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp"), ("All Files", "*.*")]
            )
            if selected:
                args.image = selected
                print(f"[INFO] Selected image: {args.image}")
            else:
                print("No file selected. Exiting.")
                sys.exit(0)

    size_name = args.size_name
    outdir = os.path.join("checkpoints", size_name)

    # For example, your .onnx files are named:
    #  image_encoder_small.onnx
    #  image_decoder_small.onnx
    encoder_filename = f"image_encoder_{size_name}.onnx"
    decoder_filename = f"image_decoder_{size_name}.onnx"

    encoder_path = os.path.join(outdir, encoder_filename)
    decoder_path = os.path.join(outdir, decoder_filename)

    # ---------------------------------------------------------------------
    # 1) Load the Encoder ONNX session
    # ---------------------------------------------------------------------
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Could not find: {encoder_path}")

    print(f"[INFO] Loading encoder ONNX => {encoder_path}")
    session_encoder = InferenceSession(
        encoder_path,
        providers=onnxruntime.get_available_providers()
    )

    enc_inputs = session_encoder.get_inputs()
    enc_input_name = enc_inputs[0].name  # e.g. "input"
    enc_input_shape = enc_inputs[0].shape  # e.g. [1,3,1024,1024]
    input_size = enc_input_shape[2:]
    enc_outputs = session_encoder.get_outputs()
    enc_output_names = [out.name for out in enc_outputs]

    print(f"[INFO] Encoder input shape: {enc_input_shape}")
    print(f"[INFO] Encoder output names: {enc_output_names}")

    # ---------------------------------------------------------------------
    # 2) Load the Decoder ONNX session
    # ---------------------------------------------------------------------
    if not os.path.exists(decoder_path):
        raise FileNotFoundError(f"Could not find: {decoder_path}")

    print(f"[INFO] Loading decoder ONNX => {decoder_path}")
    session_decoder = InferenceSession(
        decoder_path,
        providers=onnxruntime.get_available_providers()
    )

    dec_inputs = session_decoder.get_inputs()
    dec_input_names = [inp.name for inp in dec_inputs]
    dec_outputs = session_decoder.get_outputs()
    dec_output_names = [out.name for out in dec_outputs]

    print(f"[INFO] Decoder input names: {dec_input_names}")
    print(f"[INFO] Decoder output names: {dec_output_names}")

    # ---------------------------------------------------------------------
    # 3) Read and Preprocess the Image
    # ---------------------------------------------------------------------
    print(f"[INFO] Reading image from: {args.image}")
    original_bgr = cv2.imread(args.image)
    if original_bgr is None:
        raise ValueError(f"Cannot read image at {args.image}")

    H_orig, W_orig = original_bgr.shape[:2]
    print(f"[INFO] Original image size: {W_orig}x{H_orig}")

    # We'll keep a copy for display
    original_display = original_bgr.copy()

    # Resize image to the encoder's input resolution
    print(f"[INFO] Resizing image to: {input_size[1]}x{input_size[0]}")
    resized = cv2.resize(original_bgr, (input_size[1], input_size[0]))
    resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    resized_rgb = (resized_rgb - mean) / std

    # CHW layout and add batch dim => [1,3,H,W]
    resized_rgb = np.transpose(resized_rgb, (2, 0, 1))
    input_tensor = np.expand_dims(resized_rgb, axis=0)

    # ---------------------------------------------------------------------
    # 4) Run the Encoder once
    # ---------------------------------------------------------------------
    print("[INFO] Running encoder session...")
    t_enc_start = time.time()
    outputs_enc = session_encoder.run(
        enc_output_names,
        {enc_input_name: input_tensor}
    )
    t_enc_end = time.time()
    print(f"[INFO] Encoder inference time: {(t_enc_end - t_enc_start)*1000:.2f} ms")

    # The encoder outputs 5 arrays: 
    #   [0] image_embeddings:   [1,256,64,64]
    #   [1] high_res_features1: [1,32,256,256]
    #   [2] high_res_features2: [1,64,128,128]
    #   [3] current_vision_feat [4096,1,256] (not used in simple scenario)
    #   [4] vision_pos_embed    [4096,1,256] (not used in simple scenario)
    image_embeddings = outputs_enc[0]
    feats_0 = outputs_enc[1]
    feats_1 = outputs_enc[2]
    # ignoring outputs_enc[3] and outputs_enc[4] in this simple script

    # ---------------------------------------------------------------------
    # 5) Interactive Prompting
    # ---------------------------------------------------------------------
    points = []  # list of (x, y)
    labels = []  # list of 1 or 0
    disp_max_size = 1200
    scale_factor = 1.0
    if max(W_orig, H_orig) > disp_max_size:
        scale_factor = disp_max_size / float(max(W_orig, H_orig))

    disp_w = int(W_orig * scale_factor)
    disp_h = int(H_orig * scale_factor)
    display_image = cv2.resize(original_display, (disp_w, disp_h))

    def reset_points():
        points.clear()
        labels.clear()

    def mouse_callback(event, x, y, flags, param):
        """
        Left-click => positive point
        Right-click => negative point
        Middle-click => reset
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            real_x = int(x / scale_factor)
            real_y = int(y / scale_factor)
            points.append((real_x, real_y))
            labels.append(1)
            print(f"[INFO] Added POSITIVE point: ({real_x}, {real_y})")
            run_decoder_inference()

        elif event == cv2.EVENT_RBUTTONDOWN:
            real_x = int(x / scale_factor)
            real_y = int(y / scale_factor)
            points.append((real_x, real_y))
            labels.append(0)
            print(f"[INFO] Added NEGATIVE point: ({real_x}, {real_y})")
            run_decoder_inference()

        elif event == cv2.EVENT_MBUTTONDOWN:
            print("[INFO] Resetting all points.")
            reset_points()
            run_decoder_inference()

    # Register callback
    cv2.namedWindow("SAM2 Demo", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("SAM2 Demo", mouse_callback)

    def run_decoder_inference():
        out_vis = display_image.copy()
        if len(points) == 0:
            cv2.imshow("SAM2 Demo", out_vis)
            return

        # scale points to the encoder input
        pts, lbls = prepare_points(points, labels, (H_orig, W_orig), (input_size[0], input_size[1]))
        if pts is None:
            cv2.imshow("SAM2 Demo", out_vis)
            return

        # build inputs for decoder
        decoder_inputs = {
            dec_input_names[0]: pts,     # "point_coords"
            dec_input_names[1]: lbls,    # "point_labels"
            dec_input_names[2]: image_embeddings,
            dec_input_names[3]: feats_0,
            dec_input_names[4]: feats_1,
        }

        t_dec_start = time.time()
        # outputs: obj_ptr, mask_for_mem, low_res_masks
        obj_ptr, mask_for_mem, low_res_masks = session_decoder.run(dec_output_names, decoder_inputs)
        t_dec_end = time.time()
        print(f"[INFO] Decoder inference time: {(t_dec_end - t_dec_start)*1000:.2f} ms")

        # pick the first mask
        final_mask_lowres = low_res_masks[0, 0]  # shape [256,256]

        # upsample to original image size
        final_mask = cv2.resize(
            final_mask_lowres,
            (W_orig, H_orig),
            interpolation=cv2.INTER_LINEAR
        )
        final_mask = (final_mask > 0).astype(np.uint8) * 255  # binarize => 0 or 255

        # Prepare alpha-blended overlay in bright green
        overlay = original_display.copy()
        color_mask = np.zeros_like(overlay, dtype=np.uint8)
        color_mask[final_mask == 255] = (0, 255, 0)  # bright green
        alpha = 0.5
        overlay = cv2.addWeighted(overlay, 1.0, color_mask, alpha, 0)

        # draw points
        for (i, (px, py)) in enumerate(points):
            color = (0, 0, 255) if labels[i] == 1 else (255, 0, 0)  # positive=red, negative=blue
            cv2.circle(overlay, (px, py), 6, color, -1)

        # scale overlay to display window
        overlay_disp = cv2.resize(overlay, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("SAM2 Demo", overlay_disp)

    # Show initial window (with no points)
    run_decoder_inference()

    print("[INFO] Interactive mode. Use L/R/M clicks. Press ESC to exit.")
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
