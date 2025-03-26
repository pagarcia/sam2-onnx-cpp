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

def onnx_test_video_mkv_full(args):
    """
    Reads a video file with OpenCV, uses all SAM2 ONNX modules:
      - image_encoder_<size_name>.onnx
      - image_decoder_<size_name>.onnx
      - memory_encoder_<size_name>.onnx
      - memory_attention_<size_name>.onnx

    to segment the entire video from a single user prompt on the first frame
    (hardcoded at coords=(510,375)).

    We overlay the mask in semi-transparent red on each frame.
    Output is written to <video_basename>_mask_overlay.mkv .
    """

    # ---------------------------------------------------------------------
    # 1) Load all four ONNX sessions
    # ---------------------------------------------------------------------
    outdir = os.path.join("checkpoints", args.size_name)
    enc_path = os.path.join(outdir, f"image_encoder_{args.size_name}.onnx")
    dec_path = os.path.join(outdir, f"image_decoder_{args.size_name}.onnx")
    men_path = os.path.join(outdir, f"memory_encoder_{args.size_name}.onnx")
    mat_path = os.path.join(outdir, f"memory_attention_{args.size_name}.onnx")

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
    # 3) A single user prompt on the first frame (hardcoded)
    # ---------------------------------------------------------------------
    first_frame_prompt_coords = [[510, 375]]  # example
    first_frame_prompt_labels = [1]           # 1 => foreground

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
        frame_tensor, (orig_H, orig_W) = prepare_image(frame_bgr, encoder_input_size)
        # The encoder returns 5 outputs:
        #   image_embeddings, feats_0, feats_1, current_vision_feat, vision_pos_embed
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
                # 1) Memory Attention
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

                # 2) decode with no new points
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

        # (C) Upsample the predicted mask + overlay in red
        best_mask = pred_mask[0, 0]  # pick the first
        upsampled_logits = cv2.resize(best_mask, (orig_W, orig_H), interpolation=cv2.INTER_LINEAR)
        final_mask = (upsampled_logits > 0).astype(np.uint8) * 255

        alpha = 0.5
        color_mask = np.zeros_like(frame_bgr, dtype=np.uint8)
        color_mask[final_mask > 0] = (0, 0, 255)  # Red
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
        description="Use all SAM2 ONNX modules on a video with a single user prompt (hardcoded). Overlays the predicted mask in red."
    )
    parser.add_argument(
        "--size_name",
        type=str,
        default="small",
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
