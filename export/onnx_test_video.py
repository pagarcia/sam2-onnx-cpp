import os
import sys
import time
import cv2
import numpy as np
import argparse
import onnxruntime
from onnxruntime import InferenceSession

# For file dialog (if you want to pick a video via GUI)
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
        pts = np.zeros((1,0,2), dtype=np.float32)
        lbls= np.zeros((1,0),   dtype=np.float32)
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

def decode_points(session_decoder, point_coords, point_labels, image_embed, feats_0, feats_1):
    """
    Calls the image_decoder with user-provided points.
    Returns (obj_ptr, mask_for_mem, low_res_masks).
    """
    decoder_inputs = {
        "point_coords":     point_coords,  # [1,N,2]
        "point_labels":     point_labels,  # [1,N]
        "image_embed":      image_embed,   # [1,256,64,64]
        "high_res_feats_0": feats_0,       # [1,32,256,256]
        "high_res_feats_1": feats_1,       # [1,64,128,128]
    }
    return session_decoder.run(None, decoder_inputs)

def decode_no_points(session_decoder, image_embed, feats_0, feats_1):
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
    parser = argparse.ArgumentParser(description="Interactive first-frame annotation with green mask, then full-video segmentation with SAM2 ONNX.")
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

    # ------------------------------------------------------------------
    # 1) Possibly open a file dialog if --video is not given
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 2) Set up paths and load ONNX models
    # ------------------------------------------------------------------
    outdir = os.path.join("checkpoints", args.size_name)
    enc_path     = os.path.join(outdir, f"image_encoder_{args.size_name}.onnx")
    dec_path     = os.path.join(outdir, f"image_decoder_{args.size_name}.onnx")
    memenc_path  = os.path.join(outdir, f"memory_encoder_{args.size_name}.onnx")
    memattn_path = os.path.join(outdir, f"memory_attention_{args.size_name}.onnx")

    if not os.path.exists(enc_path):
        raise FileNotFoundError(f"Could not find: {enc_path}")
    if not os.path.exists(dec_path):
        raise FileNotFoundError(f"Could not find: {dec_path}")
    if not os.path.exists(memenc_path):
        raise FileNotFoundError(f"Could not find: {memenc_path}")
    if not os.path.exists(memattn_path):
        raise FileNotFoundError(f"Could not find: {memattn_path}")

    session_encoder = InferenceSession(enc_path,     providers=onnxruntime.get_available_providers())
    session_decoder = InferenceSession(dec_path,     providers=onnxruntime.get_available_providers())
    session_memenc  = InferenceSession(memenc_path,  providers=onnxruntime.get_available_providers())
    session_memattn = InferenceSession(memattn_path, providers=onnxruntime.get_available_providers())

    enc_input_shape = session_encoder.get_inputs()[0].shape  # e.g. [1,3,1024,1024]
    encoder_input_size = enc_input_shape[2:]                 # (1024, 1024)
    print(f"[INFO] Using encoder input size = {encoder_input_size}")

    # ------------------------------------------------------------------
    # 3) Open video
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: cannot open video: {args.video}")
        sys.exit(1)

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video: {args.video} => {width}x{height}, fps={fps:.1f}, frames={nframes}")

    # Prepare output
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_name = os.path.splitext(args.video)[0] + "_mask_overlay.avi"
    writer = cv2.VideoWriter(out_name, fourcc, fps, (width,height))
    if not writer.isOpened():
        print(f"ERROR: cannot open VideoWriter for {out_name}")
        cap.release()
        sys.exit(1)

    print(f"[INFO] Output overlay => {out_name}")

    # ------------------------------------------------------------------
    # 4) Read the first frame and let user annotate in real-time
    # ------------------------------------------------------------------
    ret, first_frame_bgr = cap.read()
    if not ret:
        print("ERROR: no frames in video.")
        cap.release()
        writer.release()
        sys.exit(1)

    # (A) Encode the first frame once
    first_frame_tensor, (oh, ow) = prepare_image(first_frame_bgr, encoder_input_size)
    enc_out = session_encoder.run(None, {session_encoder.get_inputs()[0].name: first_frame_tensor})
    (image_embeddings, feats_0, feats_1, _, _) = enc_out  # ignoring flattened_feat/vision_pos

    # We'll store interactive points here:
    first_frame_points = []
    first_frame_labels = []

    # We will display a scaled-down version if the video is large
    disp_max_size = 1200
    scale_factor = 1.0
    if max(width, height) > disp_max_size:
        scale_factor = disp_max_size / float(max(width, height))
    disp_w = int(width * scale_factor)
    disp_h = int(height* scale_factor)
    first_disp = cv2.resize(first_frame_bgr, (disp_w, disp_h))

    done_annotating = False
    def draw_overlay():
        """
        Re-run the decoder with the current points, upsample the mask,
        overlay in *green* on the first frame, and show it.
        """
        overlay_img = first_disp.copy()
        if len(first_frame_points) == 0:
            # just show points? actually no points => no mask
            for i, (px, py) in enumerate(first_frame_points):
                color = (0,0,255) if first_frame_labels[i] == 1 else (255,0,0)
                cv2.circle(overlay_img, (int(px*scale_factor), int(py*scale_factor)), 5, color, -1)
            cv2.imshow("First Frame (Green Mask)", overlay_img)
            return

        # 1) decode with current points
        pcoords, plabels = prepare_points(first_frame_points, first_frame_labels, (oh, ow), encoder_input_size)
        dec_out = decode_points(session_decoder, pcoords, plabels, image_embeddings, feats_0, feats_1)
        _, mask_for_mem, low_res_masks = dec_out
        best_mask_lowres = low_res_masks[0,0]  # pick the first
        # 2) upsample
        upsampled = cv2.resize(best_mask_lowres, (ow, oh), interpolation=cv2.INTER_LINEAR)
        final_mask = (upsampled > 0).astype(np.uint8)*255

        # 3) overlay in green
        color_mask = np.zeros_like(first_frame_bgr, dtype=np.uint8)
        color_mask[final_mask > 0] = (0,255,0)
        overlay = cv2.addWeighted(first_frame_bgr, 1.0, color_mask, 0.5, 0)

        # 4) draw points
        for i, (px, py) in enumerate(first_frame_points):
            color = (0,0,255) if first_frame_labels[i] == 1 else (255,0,0)
            cv2.circle(overlay, (px, py), 5, color, -1)

        # 5) scale overlay for display
        overlay_disp = cv2.resize(overlay, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("First Frame (Green Mask)", overlay_disp)

    def mouse_callback(event, x, y, flags, param):
        """
        Left-click => positive point
        Right-click => negative point
        Middle-click => reset
        """
        nonlocal first_frame_points, first_frame_labels
        if done_annotating:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            rx = int(x / scale_factor)
            ry = int(y / scale_factor)
            first_frame_points.append((rx, ry))
            first_frame_labels.append(1)
            draw_overlay()
        elif event == cv2.EVENT_RBUTTONDOWN:
            rx = int(x / scale_factor)
            ry = int(y / scale_factor)
            first_frame_points.append((rx, ry))
            first_frame_labels.append(0)
            draw_overlay()
        elif event == cv2.EVENT_MBUTTONDOWN:
            # reset
            first_frame_points = []
            first_frame_labels = []
            draw_overlay()

    cv2.namedWindow("First Frame (Green Mask)", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("First Frame (Green Mask)", mouse_callback)
    draw_overlay()

    print("[INFO] Interactively place points. (L=FG, R=BG, M=reset). Press Enter/Space to finalize.")
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in [13, 32]:  # Enter or Space => done
            done_annotating = True
            break
        elif key == 27:  # Esc => user canceled
            print("User canceled annotation. Exiting.")
            cap.release()
            writer.release()
            cv2.destroyAllWindows()
            sys.exit(0)

    cv2.destroyWindow("First Frame (Green Mask)")

    # ------------------------------------------------------------------
    # 5) Final decode on first frame with your chosen points + memory encode
    # ------------------------------------------------------------------
    if len(first_frame_points) > 0:
        pcoords, plabels = prepare_points(first_frame_points, first_frame_labels, (oh, ow), encoder_input_size)
        dec_out = decode_points(session_decoder, pcoords, plabels, image_embeddings, feats_0, feats_1)
    else:
        dec_out = decode_no_points(session_decoder, image_embeddings, feats_0, feats_1)

    obj_ptr, mask_for_mem, low_res_masks = dec_out
    best_mask_lowres = low_res_masks[0,0]
    upsampled = cv2.resize(best_mask_lowres, (ow, oh), interpolation=cv2.INTER_LINEAR)
    final_mask = (upsampled>0).astype(np.uint8)*255

    # green overlay for consistency
    c_mask = np.zeros_like(first_frame_bgr, dtype=np.uint8)
    c_mask[final_mask>0] = (0,255,0)
    first_overlay = cv2.addWeighted(first_frame_bgr, 1.0, c_mask, 0.5, 0)
    writer.write(first_overlay)  # write to output
    print("[INFO] Wrote first frame to output video with green overlay.")

    # memory encode => use mask_for_mem of the 1st mask
    # shape [1, numMasks, 1024,1024], pick the first mask
    first_mask_for_mem = mask_for_mem[:,0:1,...]
    memenc_inputs = {
        "mask_for_mem": first_mask_for_mem,  # [1,1,1024,1024]
        "pix_feat":     image_embeddings,    # [1,256,64,64]
    }
    memenc_out = session_memenc.run(None, memenc_inputs)
    mem_feats, mem_pos_enc, temporal_code = memenc_out

    # ------------------------------------------------------------------
    # 6) Process subsequent frames with memory
    #    => memory_attention + decode_no_points
    # ------------------------------------------------------------------
    frame_index = 1
    max_frames = args.max_frames if args.max_frames>0 else nframes

    while True:
        if frame_index >= max_frames:
            print(f"[INFO] Reached max_frames={max_frames}. Stopping.")
            break

        ret, frame_bgr = cap.read()
        if not ret:
            break  # end of video

        # 6a) encode this frame
        frame_tensor, (oh, ow) = prepare_image(frame_bgr, encoder_input_size)
        enc_out = session_encoder.run(None, {session_encoder.get_inputs()[0].name: frame_tensor})
        image_embeddings, feats_0, feats_1, flattened_feat, vision_pos_embed = enc_out

        # 6b) memory_attention => fuse current features with stored memory
        # We'll pass memory_0 = empty array (no obj ptr tokens) 
        # and memory_1 = mem_feats from first frame
        memory_0 = np.zeros((0,256), dtype=np.float32)
        mat_inputs = {
            "current_vision_feat":      image_embeddings,  
            "current_vision_pos_embed": vision_pos_embed,
            "memory_0":                memory_0,
            "memory_1":                mem_feats,         # from previous step
            "memory_pos_embed":        mem_pos_enc,
        }
        fused_out = session_memattn.run(None, mat_inputs)
        fused_feat = fused_out[0]  # shape [1,256,64,64]

        # 6c) decode with no points
        dec_out = decode_no_points(session_decoder, fused_feat, feats_0, feats_1)
        obj_ptr, mask_for_mem_batch, low_res_masks = dec_out
        best_mask_lowres = low_res_masks[0,0]

        # 6d) overlay in green
        upsampled = cv2.resize(best_mask_lowres, (ow, oh), interpolation=cv2.INTER_LINEAR)
        final_mask = (upsampled>0).astype(np.uint8)*255

        color_mask = np.zeros_like(frame_bgr, dtype=np.uint8)
        color_mask[final_mask>0] = (0,255,0)
        overlay = cv2.addWeighted(frame_bgr, 1.0, color_mask, 0.5, 0)

        writer.write(overlay)

        # 6e) optional: re-encode memory for the next frame
        # Here we pick the first mask from mask_for_mem_batch 
        # and pass fused_feat as pix_feat
        next_mask_for_mem = mask_for_mem_batch[:, 0:1, ...]
        memenc_inputs = {
            "mask_for_mem": next_mask_for_mem,
            "pix_feat":     fused_feat,
        }
        memenc_out = session_memenc.run(None, memenc_inputs)
        mem_feats, mem_pos_enc, temporal_code = memenc_out

        frame_index += 1
        if frame_index % 20 == 0:
            print(f"[INFO] Processed frame {frame_index}/{nframes} ...")

    # ------------------------------------------------------------------
    # 7) Cleanup
    # ------------------------------------------------------------------
    cap.release()
    writer.release()
    print(f"[INFO] Done! Processed {frame_index} frames. Output => {out_name}")

if __name__ == "__main__":
    main()
