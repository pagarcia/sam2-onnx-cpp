# sam2-onnx-cpp/export/onnx_test_video.py
#!/usr/bin/env python3
import os, sys, time, argparse
import cv2
import numpy as np
import onnxruntime as ort
from onnxruntime import InferenceSession
from PyQt5 import QtWidgets

# ─────────────────────────── utilities ────────────────────────────
def print_system_info() -> None:
    print("[INFO] OS :", sys.platform)
    print("[INFO] ONNX Runtime providers (available) :", ort.get_available_providers())

def _make_session(path: str, kind: str = "safe") -> InferenceSession:
    """
    kind:
      - "encoder": allow basic graph opts (faster, tends to be safe)
      - "safe":    disable all opts (decoder, memory_*), avoids FusedGemm issues
    """
    so = ort.SessionOptions()
    if kind == "encoder":
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    else:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        so.add_session_config_entry("session.disable_gemm_fast_gelu_fusion", "1")
        so.add_session_config_entry("session.disable_prepacking", "1")

    providers = ["CPUExecutionProvider"]
    print(f"[INFO] Loading {os.path.basename(path)} [{kind}] with providers={providers}")
    sess = InferenceSession(path, sess_options=so, providers=providers)
    print("[INFO] Inputs:", [(i.name, i.shape, i.type) for i in sess.get_inputs()])
    print("[INFO] Outputs:", [o.name for o in sess.get_outputs()])
    return sess

def prepare_image(frame_bgr: np.ndarray, enc_shape: tuple[int, int]
                  ) -> tuple[np.ndarray, tuple[int, int]]:
    h_enc, w_enc = enc_shape
    rgb = cv2.cvtColor(cv2.resize(frame_bgr, (w_enc, h_enc)),
                       cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.array([0.485, 0.456, 0.406], np.float32)) / \
          np.array([0.229, 0.224, 0.225], np.float32)
    tensor = np.transpose(rgb, (2, 0, 1))[np.newaxis, :].astype(np.float32)
    return np.ascontiguousarray(tensor), frame_bgr.shape[:2]

def prepare_points(points, labels, img_sz, enc_sz):
    if not points:
        return None, None
    pts  = np.asarray(points, dtype=np.float32)
    lbls = np.asarray(labels, dtype=np.float32)
    h_org, w_org = img_sz
    h_enc, w_enc = enc_sz
    pts[:, 0] = (pts[:, 0] / w_org) * w_enc
    pts[:, 1] = (pts[:, 1] / h_org) * h_enc
    return pts[np.newaxis, ...].astype(np.float32), lbls[np.newaxis, ...].astype(np.float32)

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
    """Bind by NAME; ensure float32 + contiguous."""
    if coords is None:
        coords = np.zeros((1, 0, 2), np.float32)
        labels = np.zeros((1, 0),   np.float32)
    feed = {
        "point_coords":     np.ascontiguousarray(coords.astype(np.float32)),
        "point_labels":     np.ascontiguousarray(labels.astype(np.float32)),
        "image_embed":      np.ascontiguousarray(embed.astype(np.float32)),
        "high_res_feats_0": np.ascontiguousarray(f0.astype(np.float32)),
        "high_res_feats_1": np.ascontiguousarray(f1.astype(np.float32)),
    }
    return sess_dec.run(None, feed)

def green_overlay(bgr, mask255, alpha=0.5):
    out = bgr.copy()
    green = np.zeros_like(out); green[mask255 > 0] = (0, 255, 0)
    return cv2.addWeighted(out, 1.0, green, alpha, 0)

# ─────────────── interactive prompt – seed points ────────────────
def interactive_select_points(first_bgr, sess_enc, sess_dec, enc_shape):
    tensor, (h_org, w_org) = prepare_image(first_bgr, enc_shape)
    # Run encoder and read outputs by NAME
    enc_input_name = sess_enc.get_inputs()[0].name
    enc_out_names = [o.name for o in sess_enc.get_outputs()]
    enc_vals = sess_enc.run(None, {enc_input_name: tensor})
    enc = dict(zip(enc_out_names, enc_vals))
    embed = enc["image_embeddings"]; f0 = enc["high_res_features1"]; f1 = enc["high_res_features2"]

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
        _, mask_hi, _ = decode(sess_dec, pts, lbls, embed, f0, f1)
        show((mask_hi[0, 0] > 0).astype(np.uint8))

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
    enc_input_name = sess_enc.get_inputs()[0].name
    enc_out_names = [o.name for o in sess_enc.get_outputs()]
    enc_vals = sess_enc.run(None, {enc_input_name: tensor})
    enc = dict(zip(enc_out_names, enc_vals))
    embed = enc["image_embeddings"]; f0 = enc["high_res_features1"]; f1 = enc["high_res_features2"]

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
        _, mask_hi, _ = decode(sess_dec, pts, lbls, embed, f0, f1)
        show((mask_hi[0, 0] > 0).astype(np.uint8))

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
    paths = lambda name: os.path.join(ckpt_dir, f"{name}.onnx")

    # Encoder: allow basic opts; others: safest settings
    sess_enc = _make_session(paths("image_encoder"), kind="encoder")
    sess_dec = _make_session(paths("image_decoder"), kind="safe")
    sess_men = _make_session(paths("memory_encoder"),  kind="safe")
    sess_mat = _make_session(paths("memory_attention"),kind="safe")

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
                             fps if fps > 0 else 25.0,
                             (w_org, h_org))
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

    # ensure contiguity/dtype
    embed0 = np.ascontiguousarray(embed0.astype(np.float32))
    f0_0   = np.ascontiguousarray(f0_0.astype(np.float32))
    f1_0   = np.ascontiguousarray(f1_0.astype(np.float32))

    mem_feats = mem_pos = None
    fidx = 0
    while True:
        ret, frame = cap.read()
        if not ret or (args.max_frames > 0 and fidx >= args.max_frames): break

        # ── Image encoder ─────────────────────────────────────────
        t_enc = time.time()
        if fidx == 0:
            enc_embed, f0, f1 = embed0, f0_0, f1_0
            vis_pos = None
            enc_ms = (time.time() - t_enc)*1000
        else:
            tensor, _ = prepare_image(frame, (enc_h, enc_w))
            enc_input_name = sess_enc.get_inputs()[0].name
            enc_out_names = [o.name for o in sess_enc.get_outputs()]
            enc_vals = sess_enc.run(None, {enc_input_name: tensor})
            enc = dict(zip(enc_out_names, enc_vals))
            enc_embed = enc["image_embeddings"]
            f0        = enc["high_res_features1"]
            f1        = enc["high_res_features2"]
            vis_pos   = enc["vision_pos_embed"]
            enc_embed = np.ascontiguousarray(enc_embed.astype(np.float32))
            f0        = np.ascontiguousarray(f0.astype(np.float32))
            f1        = np.ascontiguousarray(f1.astype(np.float32))
            vis_pos   = np.ascontiguousarray(vis_pos.astype(np.float32))
            enc_ms = (time.time() - t_enc)*1000

        # ── Memory attention (from 2nd frame) ─────────────────────
        if fidx > 0 and mem_feats is not None:
            t_mat = time.time()
            attn_inputs = {
                "current_vision_feat":      enc_embed,
                "current_vision_pos_embed": vis_pos,
                "memory_0":                 np.zeros((0,256), np.float32),
                "memory_1":                 mem_feats,
                "memory_pos_embed":         mem_pos
            }
            fused_embed = sess_mat.run(None, attn_inputs)[0]
            fused_embed = np.ascontiguousarray(fused_embed.astype(np.float32))
            mat_ms = (time.time() - t_mat)*1000
        else:
            mat_ms = 0.0
            fused_embed = enc_embed

        # ── Decoder ───────────────────────────────────────────────
        t_dec = time.time()
        if fidx == 0:
            _, mask_for_mem, pred = decode(sess_dec, pts0, lbls0, enc_embed, f0, f1)
        else:
            _, mask_for_mem, pred = decode(sess_dec, None, None, fused_embed, f0, f1)
        dec_ms = (time.time() - t_dec)*1000

        # ── Memory encoder ────────────────────────────────────────
        t_men = time.time()
        men_out = sess_men.run(None, {
            "mask_for_mem": np.ascontiguousarray(mask_for_mem[:, 0:1].astype(np.float32)),
            "pix_feat":     np.ascontiguousarray(enc_embed.astype(np.float32)),
        })
        mem_feats, mem_pos, _ = men_out
        mem_feats = np.ascontiguousarray(mem_feats.astype(np.float32))
        mem_pos   = np.ascontiguousarray(mem_pos.astype(np.float32))
        men_ms = (time.time() - t_men)*1000

        # ── Overlay & write ───────────────────────────────────────
        logits  = mask_for_mem[0, 0]                          # 1024×1024 float
        mask_hi = cv2.resize(logits, (w_org, h_org), cv2.INTER_LINEAR)
        mask    = (mask_hi > 0).astype(np.uint8)              # 0/1 full-res
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
