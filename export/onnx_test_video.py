import os
import sys
import time
import cv2
import numpy as np
import argparse
import onnxruntime
from onnxruntime import InferenceSession

# For file dialog (optional)
try:
    import tkinter as tk
    from tkinter import filedialog
    HAVE_TK = True
except ImportError:
    HAVE_TK = False

def prepare_image(frame_bgr, input_size):
    """
    Convert an OpenCV BGR frame into a preprocessed tensor 
    for the SAM2 image encoder of shape [1,3,H,W].
    """
    H, W = frame_bgr.shape[:2]

    # Resize to the encoder input size (e.g. 1024x1024)
    resized = cv2.resize(frame_bgr, (input_size[1], input_size[0]))
    # Convert BGR -> RGB, scale [0..1], float32
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Normalize
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    resized = (resized - mean) / std

    # [H,W,3] -> [3,H,W] -> [1,3,H,W]
    resized = np.transpose(resized, (2, 0, 1))
    resized = np.expand_dims(resized, axis=0)
    return resized, (H, W)

def prepare_points(point_coords, point_labels, orig_im_size, input_size):
    """
    Scale user-specified points from the original frame size
    into the model's input resolution. Returns [1,N,2] and [1,N].
    """
    pts = np.array(point_coords, dtype=np.float32)[np.newaxis, ...]   # [1, N, 2]
    lbls = np.array(point_labels, dtype=np.float32)[np.newaxis, ...]  # [1, N]

    oh, ow = orig_im_size
    ih, iw = input_size
    pts[..., 0] = (pts[..., 0] / float(ow)) * iw  # x-scale
    pts[..., 1] = (pts[..., 1] / float(oh)) * ih  # y-scale
    return pts, lbls

def decode_with_points(session_decoder, point_coords, point_labels, image_embed, feats_0, feats_1):
    """
    Calls the new ImageDecoder with user-provided points.
    Returns (obj_ptr, mask_for_mem, pred_mask).
    """
    inputs = {
        "point_coords":     point_coords,  # [1, N, 2]
        "point_labels":     point_labels,  # [1, N]
        "image_embed":      image_embed,   # [1,256,64,64]
        "high_res_feats_0": feats_0,       # [1,32,256,256]
        "high_res_feats_1": feats_1,       # [1,64,128,128]
    }
    obj_ptr, mask_for_mem, pred_mask = session_decoder.run(None, inputs)
    return obj_ptr, mask_for_mem, pred_mask

def decode_no_points(session_decoder, image_embed, feats_0, feats_1):
    """
    Calls the new ImageDecoder with an empty (no prompts) scenario.
    """
    empty_pts  = np.zeros((1,0,2), dtype=np.float32)
    empty_lbls = np.zeros((1,0),   dtype=np.float32)

    inputs = {
        "point_coords":     empty_pts,
        "point_labels":     empty_lbls,
        "image_embed":      image_embed,
        "high_res_feats_0": feats_0,
        "high_res_feats_1": feats_1,
    }
    obj_ptr, mask_for_mem, pred_mask = session_decoder.run(None, inputs)
    return obj_ptr, mask_for_mem, pred_mask

def interactive_select_points(
    frame_bgr, session_encoder, session_decoder, encoder_input_size
):
    """
    1) Encode the given frame.
    2) Let the user interactively specify seed points (L-click=positive, R-click=negative, M-click=reset).
    3) Return the final chosen points, labels, plus the encoded outputs for subsequent usage.
    """
    # (A) Run the Encoder on the first frame
    frame_tensor, (orig_H, orig_W) = prepare_image(frame_bgr, encoder_input_size)
    enc_out = session_encoder.run(None, {session_encoder.get_inputs()[0].name: frame_tensor})
    # unpack
    image_embeddings = enc_out[0]  # [1,256,64,64]
    feats_0 = enc_out[1]          # [1,32,256,256]
    feats_1 = enc_out[2]          # [1,64,128,128]
    # ignoring enc_out[3] & enc_out[4] for interactive step

    # Prepare for interactive display
    disp_max_size = 1200
    scale_factor = 1.0
    h, w = frame_bgr.shape[:2]
    if max(w, h) > disp_max_size:
        scale_factor = disp_max_size / float(max(w, h))

    disp_w = int(w * scale_factor)
    disp_h = int(h * scale_factor)
    display_image = cv2.resize(frame_bgr, (disp_w, disp_h))

    # Lists for points and labels
    points = []
    labels = []

    # Decoder input names (read from session_decoder)
    dec_inputs = session_decoder.get_inputs()
    dec_input_names = [inp.name for inp in dec_inputs]
    dec_outputs = session_decoder.get_outputs()
    dec_output_names = [out.name for out in dec_outputs]

    def reset_points():
        points.clear()
        labels.clear()

    def run_decoder_inference():
        """
        Re-run the decoder each time we add a point or reset,
        and display the updated mask overlay.
        """
        out_vis = display_image.copy()
        if len(points) == 0:
            cv2.imshow("First Frame - Interactive SAM2", out_vis)
            return

        # Scale user-specified points to the encoder input
        pts, lbls = prepare_points(points, labels, (h, w), (encoder_input_size[0], encoder_input_size[1]))
        if pts is None:
            cv2.imshow("First Frame - Interactive SAM2", out_vis)
            return

        # Build inputs for the decoder
        decoder_inputs = {
            dec_input_names[0]: pts,           # point_coords
            dec_input_names[1]: lbls,          # point_labels
            dec_input_names[2]: image_embeddings,
            dec_input_names[3]: feats_0,
            dec_input_names[4]: feats_1,
        }

        # Run decoder
        obj_ptr, mask_for_mem, low_res_masks = session_decoder.run(dec_output_names, decoder_inputs)

        # The mask is in low_res_masks[0,0] => [256,256]
        final_mask_lowres = low_res_masks[0, 0]
        final_mask = cv2.resize(final_mask_lowres, (w, h), interpolation=cv2.INTER_LINEAR)
        final_mask = (final_mask > 0).astype(np.uint8) * 255

        # Create alpha-blended green overlay
        overlay = frame_bgr.copy()
        color_mask = np.zeros_like(overlay, dtype=np.uint8)
        color_mask[final_mask == 255] = (0, 255, 0)  # bright green
        alpha = 0.5
        overlay = cv2.addWeighted(overlay, 1.0, color_mask, alpha, 0)

        # Draw points
        for (i, (px, py)) in enumerate(points):
            color = (0, 0, 255) if labels[i] == 1 else (255, 0, 0)
            cv2.circle(overlay, (px, py), 6, color, -1)

        # Scale overlay to display window
        overlay_disp = cv2.resize(overlay, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("First Frame - Interactive SAM2", overlay_disp)

    def mouse_callback(event, x, y, flags, param):
        """
        L-click => positive point
        R-click => negative point
        M-click => reset all
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            real_x = int(x / scale_factor)
            real_y = int(y / scale_factor)
            points.append((real_x, real_y))
            labels.append(1)  # foreground
            run_decoder_inference()
        elif event == cv2.EVENT_RBUTTONDOWN:
            real_x = int(x / scale_factor)
            real_y = int(y / scale_factor)
            points.append((real_x, real_y))
            labels.append(0)  # background
            run_decoder_inference()
        elif event == cv2.EVENT_MBUTTONDOWN:
            reset_points()
            run_decoder_inference()

    # Create a window and set up the callback
    cv2.namedWindow("First Frame - Interactive SAM2", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("First Frame - Interactive SAM2", mouse_callback)

    # Initial display
    run_decoder_inference()

    print("[INFO] Interactive mode on first frame.")
    print("       L-click=foreground, R-click=background, M-click=reset.")
    print("       Press ESC (or Enter) when done selecting points.")

    while True:
        key = cv2.waitKey(20) & 0xFF
        # ESC or ENTER to finalize
        if key == 27 or key == 13:
            break

    cv2.destroyAllWindows()

    # Return final user-chosen points & labels, plus the encoder outputs
    return points, labels, image_embeddings, feats_0, feats_1, (orig_H, orig_W)

def onnx_test_video_mkv_full(args):
    """
    Reads a video file with OpenCV, uses all SAM2 ONNX modules:
      - image_encoder_<model_size>.onnx
      - image_decoder_<model_size>.onnx
      - memory_encoder_<model_size>.onnx
      - memory_attention_<model_size>.onnx

    to segment the entire video from a single user prompt on the first frame
    (interactively selected).
    We overlay the mask in semi-transparent green on each frame.
    Output is written to <video_basename>_mask_overlay.mkv .
    """

    # ---------------------------------------------------------------------
    # 1) Load all four ONNX sessions
    # ---------------------------------------------------------------------
    outdir = os.path.join("checkpoints", args.model_size)
    enc_path = os.path.join(outdir, f"image_encoder_{args.model_size}.onnx")
    dec_path = os.path.join(outdir, f"image_decoder_{args.model_size}.onnx")
    men_path = os.path.join(outdir, f"memory_encoder_{args.model_size}.onnx")
    mat_path = os.path.join(outdir, f"memory_attention_{args.model_size}.onnx")

    session_encoder = InferenceSession(enc_path, providers=onnxruntime.get_available_providers())
    session_decoder = InferenceSession(dec_path, providers=onnxruntime.get_available_providers())
    session_memenc  = InferenceSession(men_path,  providers=onnxruntime.get_available_providers())
    session_memattn = InferenceSession(mat_path,  providers=onnxruntime.get_available_providers())

    print("Encoder I/O:",
          [i.name for i in session_encoder.get_inputs()],
          [o.name for o in session_encoder.get_outputs()])
    print("Decoder I/O:",
          [i.name for i in session_decoder.get_inputs()],
          [o.name for o in session_decoder.get_outputs()])
    print("MemEnc  I/O:",
          [i.name for i in session_memenc.get_inputs()],
          [o.name for o in session_memenc.get_outputs()])
    print("MemAttn I/O:",
          [i.name for i in session_memattn.get_inputs()],
          [o.name for o in session_memattn.get_outputs()])

    # The encoder typically expects [1,3,1024,1024]
    enc_input_shape = session_encoder.get_inputs()[0].shape
    encoder_input_size = enc_input_shape[2:]  # e.g. (1024, 1024)
    print(f"[INFO] encoder_input_size => {encoder_input_size}")

    # ---------------------------------------------------------------------
    # 2) OpenCV read
    # ---------------------------------------------------------------------
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: could not open video {args.video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video {args.video} => {width}x{height}, fps={fps}, frames={num_frames}")

    # Create output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_filename = os.path.splitext(args.video)[0] + "_mask_overlay.mkv"
    writer = cv2.VideoWriter(out_filename, fourcc, fps, (width, height), True)
    if not writer.isOpened():
        print(f"ERROR: could not open VideoWriter for {out_filename}")
        cap.release()
        return
    print(f"Output overlay video => {out_filename}")

    # ---------------------------------------------------------------------
    # 3) Interactive user prompt on the FIRST frame
    # ---------------------------------------------------------------------
    ret, first_frame_bgr = cap.read()
    if not ret:
        print("ERROR: Could not read the first frame from the video.")
        cap.release()
        writer.release()
        return

    # Move back to frame 0 after reading it once:
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Let the user select points on this first frame
    (user_points, user_labels,
     first_image_embeddings,
     first_feats_0,
     first_feats_1,
     (orig_H, orig_W)) = interactive_select_points(
        first_frame_bgr,
        session_encoder,
        session_decoder,
        encoder_input_size
    )

    # If no points were chosen, proceed with no prompt
    if len(user_points) == 0:
        print("[WARNING] No points selected. Proceeding with no prompt on the first frame.")
        first_frame_prompt_coords = []
        first_frame_prompt_labels = []
        first_image_embeddings_pass = first_image_embeddings
        first_feats_0_pass = first_feats_0
        first_feats_1_pass = first_feats_1
    else:
        first_frame_prompt_coords = user_points
        first_frame_prompt_labels = user_labels
        first_image_embeddings_pass = first_image_embeddings
        first_feats_0_pass = first_feats_0
        first_feats_1_pass = first_feats_1

    # We'll keep memory from just the *last* frame
    mem_feats_accum   = None
    mem_pos_enc_accum = None

    max_frames = args.max_frames if args.max_frames > 0 else -1
    frame_index = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if max_frames > 0 and frame_index >= max_frames:
            print(f"Reached max_frames={max_frames}. Stopping early.")
            break

        # (A) Encoder
        t0 = time.time()
        if frame_index == 0:
            # We already have the encoder outputs for the first frame
            image_embeddings = first_image_embeddings_pass
            feats_0 = first_feats_0_pass
            feats_1 = first_feats_1_pass
            # just measure time roughly
            enc_time = (time.time() - t0)*1000
        else:
            frame_tensor, (orig_H, orig_W) = prepare_image(frame_bgr, encoder_input_size)
            enc_out = session_encoder.run(None, {session_encoder.get_inputs()[0].name: frame_tensor})
            (image_embeddings,
             feats_0,
             feats_1,
             flattened_feat,
             vision_pos_embed) = enc_out
            enc_time = (time.time() - t0)*1000

        # (B) Decoder logic
        if frame_index == 0:
            # First frame => user prompt
            pcoords, plabels = prepare_points(
                first_frame_prompt_coords,
                first_frame_prompt_labels,
                (orig_H, orig_W),
                encoder_input_size
            )
            t1 = time.time()
            if pcoords is None or plabels is None:
                # If user did not select any points, decode with no points
                obj_ptr, mask_for_mem, pred_mask = decode_no_points(
                    session_decoder,
                    image_embeddings,
                    feats_0,
                    feats_1
                )
            else:
                obj_ptr, mask_for_mem, pred_mask = decode_with_points(
                    session_decoder,
                    pcoords, plabels,
                    image_embeddings,
                    feats_0,
                    feats_1
                )
            dec_time = (time.time() - t1)*1000
            mat_time = 0.0
        else:
            # Subsequent frames => memory attention + no-points decode
            if mem_feats_accum is not None and mem_pos_enc_accum is not None:
                # memory attention
                t_mat = time.time()
                dummy_mem_0 = np.zeros((0,256), dtype=np.float32)
                mat_inputs = {
                    "current_vision_feat":      image_embeddings,
                    "current_vision_pos_embed": vision_pos_embed,
                    "memory_0":                dummy_mem_0,
                    "memory_1":                mem_feats_accum,
                    "memory_pos_embed":        mem_pos_enc_accum
                }
                fused_feat_list = session_memattn.run(None, mat_inputs)
                fused_feat = fused_feat_list[0]  # => [1,256,64,64]
                mat_time = (time.time() - t_mat)*1000

                # decode with no new points
                t2 = time.time()
                obj_ptr, mask_for_mem, pred_mask = decode_no_points(
                    session_decoder,
                    fused_feat,
                    feats_0,
                    feats_1
                )
                dec_time = (time.time() - t2)*1000
            else:
                # no memory yet => decode with no prompt
                mat_time = 0.0
                t2 = time.time()
                obj_ptr, mask_for_mem, pred_mask = decode_no_points(
                    session_decoder,
                    image_embeddings,
                    feats_0,
                    feats_1
                )
                dec_time = (time.time() - t2)*1000

        # (C) Upsample the predicted mask + overlay in green
        best_mask = pred_mask[0, 0]  # pick the first
        upsampled_logits = cv2.resize(best_mask, (orig_W, orig_H), interpolation=cv2.INTER_LINEAR)
        final_mask = (upsampled_logits > 0).astype(np.uint8) * 255

        alpha = 0.5
        color_mask = np.zeros_like(frame_bgr, dtype=np.uint8)
        color_mask[final_mask > 0] = (0, 255, 0)  # bright green
        overlay_frame = cv2.addWeighted(frame_bgr, 1.0, color_mask, alpha, 0)

        writer.write(overlay_frame)

        # (D) Update memory for next frame
        highres_mask = mask_for_mem[:, 0:1, ...]  # shape [1,1,1024,1024]

        t3 = time.time()
        memenc_inputs = {
            "mask_for_mem": highres_mask,
            "pix_feat":     image_embeddings  # or fused_feat
        }
        memenc_outputs = session_memenc.run(None, memenc_inputs)
        mem_feats, mem_pose, temporal_code = memenc_outputs
        memenc_time = (time.time() - t3)*1000

        mem_feats_accum   = mem_feats
        mem_pos_enc_accum = mem_pose

        # Print times
        if frame_index == 0:
            print(f"Frame {frame_index:03d} - Enc: {enc_time:.1f} ms | Dec: {dec_time:.1f} ms | MemEnc: {memenc_time:.1f} ms")
        else:
            if mat_time > 0:
                print(f"Frame {frame_index:03d} - Enc: {enc_time:.1f} ms | Attn: {mat_time:.1f} ms | Dec: {dec_time:.1f} ms | MemEnc: {memenc_time:.1f} ms")
            else:
                print(f"Frame {frame_index:03d} - Enc: {enc_time:.1f} ms | Dec: {dec_time:.1f} ms | MemEnc: {memenc_time:.1f} ms")

        frame_index += 1

    # Cleanup
    cap.release()
    writer.release()
    print(f"Done! Wrote {frame_index} frames with overlays to {out_filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Use all SAM2 ONNX modules on a video with a single (interactive) user prompt on the first frame. Overlays the predicted mask in bright-green with transparency."
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="tiny",
        choices=["base_plus", "large", "small", "tiny"],
        help="Which model size to use. (base_plus, large, small, or tiny)"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to input MKV/MP4 video file. If not given, will open a file dialog."
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Max frames to process; 0 => all."
    )
    args = parser.parse_args()

    # If no --video, open file dialog
    if args.video is None:
        if not HAVE_TK:
            print("Tkinter not available; please specify --video.")
            sys.exit(1)
        else:
            root = tk.Tk()
            root.withdraw()
            selected = filedialog.askopenfilename(
                title="Select a Video File",
                filetypes=[("Video Files", "*.mp4 *.mkv *.avi *.mov *.m4v"), ("All Files","*.*")]
            )
            if selected:
                args.video = selected
                print(f"[INFO] Selected video: {args.video}")
            else:
                print("No file selected. Exiting.")
                sys.exit(0)

    onnx_test_video_mkv_full(args)


if __name__ == "__main__":
    main()
