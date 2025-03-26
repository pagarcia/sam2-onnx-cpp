import os
import sys
import time
import cv2
import numpy as np
import argparse
import onnxruntime
from onnxruntime import InferenceSession

# For the file dialog (optional)
try:
    import tkinter as tk
    from tkinter import filedialog
    HAVE_TK = True
except ImportError:
    HAVE_TK = False

def prepare_image(frame_bgr, input_size):
    """
    Convert an OpenCV BGR frame into a preprocessed tensor
    for the SAM2 image encoder: shape [1,3,H,W].
    """
    H, W = frame_bgr.shape[:2]
    image_size = (H, W)

    # Resize to the encoder input size (e.g. [1024, 1024]).
    resized = cv2.resize(frame_bgr, (input_size[1], input_size[0]))
    resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    resized_rgb = (resized_rgb - mean) / std

    # [H,W,3] -> [3,H,W] -> [1,3,H,W]
    resized_rgb = np.transpose(resized_rgb, (2, 0, 1))
    resized_rgb = np.expand_dims(resized_rgb, axis=0)
    return resized_rgb, image_size

def prepare_points(points_list, labels_list, orig_im_size, input_size):
    """
    Scale user-specified points (in original frame coords)
    into the model's input resolution. Returns shape [1,N,2] and [1,N].
    """
    if len(points_list) == 0:
        # No prompts
        pts = np.zeros((1,0,2), dtype=np.float32)
        lbls = np.zeros((1,0), dtype=np.float32)
        return pts, lbls

    pts_array  = np.array(points_list, dtype=np.float32)  # shape [N,2]
    lbls_array = np.array(labels_list, dtype=np.float32)  # shape [N]

    oh, ow = orig_im_size
    ih, iw = input_size
    # x-scale
    pts_array[:, 0] = (pts_array[:, 0] / float(ow)) * iw
    # y-scale
    pts_array[:, 1] = (pts_array[:, 1] / float(oh)) * ih

    # Insert batch dimension: => [1,N,2] and [1,N]
    pts_array  = pts_array[np.newaxis, ...]
    lbls_array = lbls_array[np.newaxis, ...]
    return pts_array, lbls_array

def decode_points(
    session_decoder,
    point_coords,
    point_labels,
    image_embed,
    feats_0,
    feats_1
):
    """
    Calls the image_decoder with user-provided points. 
    Returns (obj_ptr, mask_for_mem, pred_mask).
    """
    decoder_inputs = {
        "point_coords":     point_coords,  # [1,N,2]
        "point_labels":     point_labels,  # [1,N]
        "image_embed":      image_embed,   # [1,256,64,64]
        "high_res_feats_0": feats_0,       # [1,32,256,256]
        "high_res_feats_1": feats_1,       # [1,64,128,128]
    }
    return session_decoder.run(None, decoder_inputs)

def decode_no_points(
    session_decoder,
    image_embed,
    feats_0,
    feats_1
):
    """
    Calls the image_decoder with an empty prompt (i.e. no points).
    """
    empty_pts  = np.zeros((1,0,2), dtype=np.float32)
    empty_lbls = np.zeros((1,0),   dtype=np.float32)
    decoder_inputs = {
        "point_coords":     empty_pts,
        "point_labels":     empty_lbls,
        "image_embed":      image_embed,
        "high_res_feats_0": feats_0,
        "high_res_feats_1": feats_1,
    }
    return session_decoder.run(None, decoder_inputs)

def main():
    parser = argparse.ArgumentParser(description="Interactive first-frame annotation, then full-video segmentation with SAM2 ONNX.")
    parser.add_argument(
        "--size_name", 
        type=str, 
        default="small",
        choices=["base_plus", "large", "small", "tiny"],
        help="Which model size to use: base_plus, large, small, or tiny."
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to a video file. If not provided, a file-dialog will open."
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="If > 0, limit the number of frames processed (for debugging)."
    )
    args = parser.parse_args()

    # --------------------------------------------------------------
    # 1) Possibly open a file dialog if --video is not given
    # --------------------------------------------------------------
    if args.video is None:
        if not HAVE_TK:
            print("Tkinter not available; please specify --video.")
            sys.exit(1)
        else:
            root = tk.Tk()
            root.withdraw()
            selected = filedialog.askopenfilename(
                title="Select a Video File",
                filetypes=[("Video Files", "*.mp4 *.mov *.avi *.mkv *.m4v"), ("All Files","*.*")]
            )
            if selected:
                args.video = selected
                print(f"[INFO] Selected video: {args.video}")
            else:
                print("No file selected. Exiting.")
                sys.exit(0)

    # --------------------------------------------------------------
    # 2) Set up paths and load ONNX models
    # --------------------------------------------------------------
    outdir = os.path.join("checkpoints", args.size_name)
    enc_path = os.path.join(outdir, f"image_encoder_{args.size_name}.onnx")
    dec_path = os.path.join(outdir, f"image_decoder_{args.size_name}.onnx")
    memenc_path = os.path.join(outdir, f"memory_encoder_{args.size_name}.onnx")
    memattn_path= os.path.join(outdir, f"memory_attention_{args.size_name}.onnx")

    if not os.path.exists(enc_path):
        raise FileNotFoundError(f"Could not find encoder: {enc_path}")
    if not os.path.exists(dec_path):
        raise FileNotFoundError(f"Could not find decoder: {dec_path}")
    if not os.path.exists(memenc_path):
        raise FileNotFoundError(f"Could not find memory_encoder: {memenc_path}")
    if not os.path.exists(memattn_path):
        raise FileNotFoundError(f"Could not find memory_attention: {memattn_path}")

    print(f"[INFO] Loading ONNX models from {outdir} ...")
    session_encoder = InferenceSession(enc_path, providers=onnxruntime.get_available_providers())
    session_decoder = InferenceSession(dec_path, providers=onnxruntime.get_available_providers())
    session_memenc  = InferenceSession(memenc_path, providers=onnxruntime.get_available_providers())
    session_memattn = InferenceSession(memattn_path, providers=onnxruntime.get_available_providers())

    enc_input_shape = session_encoder.get_inputs()[0].shape  # e.g. [1,3,1024,1024]
    encoder_input_size = enc_input_shape[2:]                 # (1024, 1024)
    print(f"[INFO] encoder_input_size = {encoder_input_size}")

    # --------------------------------------------------------------
    # 3) Open the Video (CV2)
    # --------------------------------------------------------------
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: cannot open video: {args.video}")
        sys.exit(1)

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video: {args.video} => {width}x{height}, fps={fps:.1f}, frames={nframes}")

    # Prepare output writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_name = os.path.splitext(args.video)[0] + "_mask_overlay.avi"
    writer = cv2.VideoWriter(out_name, fourcc, fps, (width,height))
    if not writer.isOpened():
        print(f"ERROR: cannot open VideoWriter for {out_name}")
        cap.release()
        sys.exit(1)
    print(f"[INFO] Output overlay video => {out_name}")

    # --------------------------------------------------------------
    # 4) Read the first frame, do interactive annotation
    # --------------------------------------------------------------
    ret, first_frame_bgr = cap.read()
    if not ret:
        print("ERROR: no frames in video.")
        cap.release()
        writer.release()
        sys.exit(1)

    # We'll keep the user-drawn points in these lists:
    first_frame_points = []
    first_frame_labels = []

    # For display, let's scale down if the video is large
    disp_max_size = 1200
    scale_factor = 1.0
    if max(width, height) > disp_max_size:
        scale_factor = disp_max_size / float(max(width, height))
    disp_w = int(width * scale_factor)
    disp_h = int(height * scale_factor)
    first_frame_disp = cv2.resize(first_frame_bgr, (disp_w, disp_h))

    # We'll store a flag to know if user is done
    done_annotating = False

    def mouse_callback(event, x, y, flags, param):
        """
        Left-click => positive point
        Right-click => negative point
        Middle-click => reset
        """
        nonlocal first_frame_points, first_frame_labels

        if done_annotating:
            # If user is done, ignore further clicks
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            real_x = int(x / scale_factor)
            real_y = int(y / scale_factor)
            first_frame_points.append((real_x, real_y))
            first_frame_labels.append(1)
            draw_overlay()

        elif event == cv2.EVENT_RBUTTONDOWN:
            real_x = int(x / scale_factor)
            real_y = int(y / scale_factor)
            first_frame_points.append((real_x, real_y))
            first_frame_labels.append(0)
            draw_overlay()

        elif event == cv2.EVENT_MBUTTONDOWN:
            # Reset
            first_frame_points = []
            first_frame_labels = []
            draw_overlay()

    # Prepare window
    cv2.namedWindow("First Frame Annotation", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("First Frame Annotation", mouse_callback)

    def draw_overlay():
        """
        Draw the current points on top of the first-frame 
        display image, and show in the window.
        """
        overlay = first_frame_disp.copy()
        for i, (px, py) in enumerate(first_frame_points):
            color = (0, 0, 255) if first_frame_labels[i] == 1 else (255, 0, 0)
            cv2.circle(overlay, (int(px*scale_factor), int(py*scale_factor)), 5, color, -1)
        cv2.imshow("First Frame Annotation", overlay)

    draw_overlay()
    print("[INFO] Place points (L=FG, R=BG, M=reset). Press Enter/Space to finish.")
    
    # Event loop for annotation
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in [13, 32]:  # Enter(13) or Space(32)
            # user says "done"
            done_annotating = True
            break
        elif key == 27:  # Esc
            print("[INFO] User canceled annotation. Exiting.")
            cap.release()
            writer.release()
            cv2.destroyAllWindows()
            sys.exit(0)

    cv2.destroyWindow("First Frame Annotation")

    # --------------------------------------------------------------
    # 5) Encode & Decode the first frame with the user points
    # --------------------------------------------------------------
    #  a) Encode
    first_tensor, (oh, ow) = prepare_image(first_frame_bgr, encoder_input_size)
    enc_out = session_encoder.run(None, {session_encoder.get_inputs()[0].name: first_tensor})
    # We expect 5 outputs: image_embeddings, feats_0, feats_1, flattened_feat, vision_pos_embed
    image_embeddings = enc_out[0]  # shape [1,256,64,64]
    feats_0          = enc_out[1]  # shape [1,32,256,256]
    feats_1          = enc_out[2]  # shape [1,64,128,128]
    flattened_feat   = enc_out[3]  # shape [4096,1,256]
    vision_pos_embed = enc_out[4]  # shape [4096,1,256]

    #  b) Decode with points
    pcoords, plabels = prepare_points(first_frame_points, first_frame_labels, (oh, ow), encoder_input_size)
    dec_out = decode_points(session_decoder, pcoords, plabels, image_embeddings, feats_0, feats_1)
    obj_ptr, mask_for_mem, low_res_masks = dec_out
    # low_res_masks => [1, num_masks, 256, 256]
    best_mask = low_res_masks[0, 0]  # shape [256, 256]

    #  c) Upsample & overlay for the first frame
    upsampled = cv2.resize(best_mask, (ow, oh), interpolation=cv2.INTER_LINEAR)
    final_mask = (upsampled > 0).astype(np.uint8) * 255

    alpha = 0.5
    color_mask = np.zeros_like(first_frame_bgr, dtype=np.uint8)
    color_mask[final_mask > 0] = (0, 0, 255)  # red
    first_frame_overlay = cv2.addWeighted(first_frame_bgr, 1.0, color_mask, alpha, 0)

    # Write it out
    writer.write(first_frame_overlay)
    print("[INFO] Wrote first frame with segmentation overlay to video output.")

    # --------------------------------------------------------------
    # 6) Memory encode from the first frame
    # --------------------------------------------------------------
    # mask_for_mem => shape [1, num_masks, 1024, 1024], pick the first mask
    mask_for_mem_0 = mask_for_mem[:, 0:1, ...]  # shape [1,1,1024,1024]
    memenc_inputs = {
        "mask_for_mem": mask_for_mem_0,      # [1,1,1024,1024]
        "pix_feat":     image_embeddings,    # [1,256,64,64]
    }
    memenc_out = session_memenc.run(None, memenc_inputs)
    mem_feats, mem_pos_enc, temporal_code = memenc_out
    # We'll store these to use for subsequent frames:
    #   mem_feats => e.g. shape [?,1,64]
    #   mem_pos_enc => e.g. shape [?,1,64]
    #   temporal_code => typically a small embedding for time, if used

    # --------------------------------------------------------------
    # 7) Process the rest of the frames with memory 
    #    => "memory_attention + decode_no_points"
    # --------------------------------------------------------------
    frame_index = 1
    max_frames = args.max_frames if args.max_frames > 0 else nframes

    while True:
        if frame_index >= max_frames:
            print(f"[INFO] Reached max_frames={max_frames}. Stopping.")
            break

        ret, frame_bgr = cap.read()
        if not ret:
            break  # end of video

        # Encode
        frame_tensor, (oh, ow) = prepare_image(frame_bgr, encoder_input_size)
        enc_out = session_encoder.run(None, {session_encoder.get_inputs()[0].name: frame_tensor})
        (image_embeddings,
         feats_0,
         feats_1,
         flattened_feat,
         vision_pos_embed) = enc_out

        # Memory attention
        # We'll pass the memory feats as "memory_1" and skip "memory_0" (object ptr tokens).
        # Adjust shapes to match your SAM2 memory_attention signature.
        # For example, memory_1 => shape [N,64,64,64]? 
        # This depends on how your model expects them. 
        # We'll do a minimal example here:
        memory_0 = np.zeros((0,256), dtype=np.float32)  # no object-pointer tokens

        memattn_inputs = {
            "current_vision_feat":      image_embeddings,   # [1,256,64,64]
            "current_vision_pos_embed": vision_pos_embed,   # [4096,1,256]
            "memory_0":                memory_0,
            "memory_1":                mem_feats,          # from first frame
            "memory_pos_embed":        mem_pos_enc,
        }
        fused_feat_list = session_memattn.run(None, memattn_inputs)
        fused_feat = fused_feat_list[0]  # shape [1,256,64,64]

        # Decode with no points
        dec_out = decode_no_points(session_decoder, fused_feat, feats_0, feats_1)
        obj_ptr, mask_for_mem_batch, low_res_masks = dec_out
        best_mask = low_res_masks[0, 0]

        # Upsample & overlay
        upsampled = cv2.resize(best_mask, (ow, oh), interpolation=cv2.INTER_LINEAR)
        final_mask = (upsampled > 0).astype(np.uint8) * 255

        color_mask = np.zeros_like(frame_bgr, dtype=np.uint8)
        color_mask[final_mask > 0] = (0, 0, 255)
        overlay = cv2.addWeighted(frame_bgr, 1.0, color_mask, 0.5, 0)

        # Write overlay
        writer.write(overlay)

        # Update memory (optional, or you can keep the same memory from frame 0)
        # We'll do memory encoder each frame to refine the memory
        mask_for_mem_0 = mask_for_mem_batch[:, 0:1, ...]  # pick first mask
        memenc_inputs = {
            "mask_for_mem": mask_for_mem_0,
            "pix_feat":     fused_feat,
        }
        mem_feats_out, mem_pos_enc_out, temp_code_out = session_memenc.run(None, memenc_inputs)

        # Now we can replace the memory for the next iteration
        mem_feats   = mem_feats_out
        mem_pos_enc = mem_pos_enc_out
        # We might not do anything special with the updated temporal_code 
        # unless your model needs it.

        frame_index += 1
        if frame_index % 20 == 0:
            print(f"[INFO] Processed frame {frame_index}/{nframes} ...")

    # Cleanup
    cap.release()
    writer.release()
    print(f"[INFO] Done! Processed {frame_index} frames. Output => {out_name}")

if __name__ == "__main__":
    main()
