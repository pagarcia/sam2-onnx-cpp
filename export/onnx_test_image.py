# sam2-onnx-cpp/export/onnx_test_image.py
#!/usr/bin/env python3
import os, sys, time, argparse
import cv2, numpy as np
import onnxruntime as ort
from onnxruntime import InferenceSession
from PyQt5 import QtWidgets

def print_system_info():
    print("[INFO] OS :", sys.platform)
    print("[INFO] ONNX Runtime providers (available) :", ort.get_available_providers())

def prepare_points(points, labels, img_size, enc_size):
    if not points:
        return None, None
    pts = np.asarray(points, dtype=np.float32)   # [N,2]
    lbl = np.asarray(labels, dtype=np.float32)   # [N]
    H_org, W_org = img_size
    H_enc, W_enc = enc_size
    pts[:, 0] = (pts[:, 0] / W_org) * W_enc
    pts[:, 1] = (pts[:, 1] / H_org) * H_enc
    return pts[np.newaxis, ...], lbl[np.newaxis, ...]

def prepare_rectangle(rect, img_size, enc_size):
    if rect is None:
        return None, None
    x1, y1, x2, y2 = rect
    H_org, W_org = img_size
    H_enc, W_enc = enc_size
    pts = np.array([[x1, y1], [x2, y2]], np.float32)
    pts[:, 0] = (pts[:, 0] / W_org) * W_enc
    pts[:, 1] = (pts[:, 1] / H_org) * H_enc
    lbl = np.array([2.0, 3.0], np.float32)      # SAM-2 bbox labels
    return pts[np.newaxis, ...], lbl[np.newaxis, ...]

def green_overlay(bgr, mask255, alpha=.5):
    color = np.zeros_like(bgr); color[mask255==255] = (0,255,0)
    return cv2.addWeighted(bgr, 1.0, color, alpha, 0)

def _make_session(path: str) -> InferenceSession:
    # Force CPU EP and turn off fusions that sometimes break shapes (FusedGemm).
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    # Optional: also disable some specific fusions (harmless if unknown)
    so.add_session_config_entry("session.disable_gemm_fast_gelu_fusion", "1")
    so.add_session_config_entry("session.disable_prepacking", "1")
    providers = ["CPUExecutionProvider"]  # <= key change
    print(f"[INFO] Loading {os.path.basename(path)} with providers={providers}")
    sess = InferenceSession(path, sess_options=so, providers=providers)
    print("[INFO] Inputs:", [(i.name, i.shape, i.type) for i in sess.get_inputs()])
    print("[INFO] Outputs:", [o.name for o in sess.get_outputs()])
    return sess

def main():
    print_system_info()
    ap = argparse.ArgumentParser(description="Unified SAM-2 ONNX demo (seed-points / bounding-box)")
    ap.add_argument("--model_size", default="tiny", choices=["base_plus","large","small","tiny"])
    ap.add_argument("--prompt", default="seed_points", choices=["seed_points","bounding_box"])
    args = ap.parse_args()
    mode_bbox = args.prompt == "bounding_box"
    print(f"[INFO] Prompt mode : {'bounding_box' if mode_bbox else 'seed_points'}")

    app = QtWidgets.QApplication(sys.argv)
    img_path, _ = QtWidgets.QFileDialog.getOpenFileName(
        None, "Select an Image", "", "Images (*.jpg *.jpeg *.png *.bmp);;All files (*)")
    if not img_path: sys.exit("No image selected – exiting.")
    print(f"[INFO] Selected image : {img_path}")

    ckpt_dir = os.path.join("checkpoints", args.model_size)
    enc_path = os.path.join(ckpt_dir, "image_encoder.onnx")
    dec_path = os.path.join(ckpt_dir, "image_decoder.onnx")
    if not (os.path.exists(enc_path) and os.path.exists(dec_path)):
        sys.exit(f"ERROR: ONNX files not found in {ckpt_dir}")

    sess_enc = _make_session(enc_path)
    sess_dec = _make_session(dec_path)

    enc_input_name = sess_enc.get_inputs()[0].name
    enc_h, enc_w = sess_enc.get_inputs()[0].shape[2:]
    enc_out_names = [o.name for o in sess_enc.get_outputs()]
    print(f"[INFO] Encoder input size : {(enc_h, enc_w)}")

    img_bgr = cv2.imread(img_path)
    if img_bgr is None: sys.exit("ERROR: Could not read image.")
    H_org, W_org = img_bgr.shape[:2]

    img_resized = cv2.resize(img_bgr, (enc_w, enc_h))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485,0.456,0.406], np.float32)
    std  = np.array([0.229,0.224,0.225], np.float32)
    img_rgb = (img_rgb - mean) / std
    inp_tensor = np.transpose(img_rgb, (2,0,1))[np.newaxis, :].astype(np.float32)
    inp_tensor = np.ascontiguousarray(inp_tensor)

    t0 = time.time()
    enc_vals = sess_enc.run(None, {enc_input_name: inp_tensor})
    enc_dict = dict(zip(enc_out_names, enc_vals))
    print(f"[INFO] Encoder time : {(time.time()-t0)*1000:.1f} ms")

    img_embed = np.ascontiguousarray(enc_dict["image_embeddings"].astype(np.float32))   # [1,256,64,64]
    feats0    = np.ascontiguousarray(enc_dict["high_res_features1"].astype(np.float32)) # [1,32,256,256]
    feats1    = np.ascontiguousarray(enc_dict["high_res_features2"].astype(np.float32)) # [1,64,128,128]

    for nm, arr, shp in [
        ("image_embed", img_embed, (1,256,64,64)),
        ("feats0",      feats0,    (1,32,256,256)),
        ("feats1",      feats1,    (1,64,128,128)),
    ]:
        if tuple(arr.shape) != shp:
            print(f"[WARN] {nm} shape {arr.shape} != {shp}")

    disp_max = 1200
    scale = min(1.0, disp_max / max(W_org, H_org))
    disp_w, disp_h = int(W_org*scale), int(H_org*scale)
    disp_base = cv2.resize(img_bgr, (disp_w, disp_h))

    DEC_KEYS = {
        "point_coords":     next(n for n in [i.name for i in sess_dec.get_inputs()] if "point_coords" in n),
        "point_labels":     next(n for n in [i.name for i in sess_dec.get_inputs()] if "point_labels" in n),
        "image_embed":      next(n for n in [i.name for i in sess_dec.get_inputs()] if "image_embed" in n),
        "high_res_feats_0": next(n for n in [i.name for i in sess_dec.get_inputs()] if "high_res_feats_0" in n),
        "high_res_feats_1": next(n for n in [i.name for i in sess_dec.get_inputs()] if "high_res_feats_1" in n),
    }

    points, labels = [], []

    def run_decoder_points():
        if not points:
            cv2.imshow("SAM-2 Demo", disp_base); return
        pts, lbl = prepare_points(points, labels, (H_org,W_org), (enc_h,enc_w))
        # Ensure exact dtype & contiguity
        pts = np.ascontiguousarray(pts.astype(np.float32))
        lbl = np.ascontiguousarray(lbl.astype(np.float32))
        dec_inputs = {
            DEC_KEYS["point_coords"]:     pts,
            DEC_KEYS["point_labels"]:     lbl,
            DEC_KEYS["image_embed"]:      img_embed,
            DEC_KEYS["high_res_feats_0"]: feats0,
            DEC_KEYS["high_res_feats_1"]: feats1,
        }
        t = time.time()
        obj_ptr, mask_for_mem, pred_low = sess_dec.run(None, dec_inputs)
        print(f"[INFO] Decoder time : {(time.time()-t)*1000:.1f} ms")
        mask256 = pred_low[0,0]
        mask = cv2.resize(mask256, (W_org, H_org))
        mask255 = (mask>0).astype(np.uint8)*255
        overlay = green_overlay(img_bgr, mask255)
        for i,(px,py) in enumerate(points):
            col = (0,0,255) if labels[i]==1 else (255,0,0)
            cv2.circle(overlay,(px,py),6,col,-1)
        cv2.imshow("SAM-2 Demo", cv2.resize(overlay,(disp_w,disp_h)))

    rect_start = rect_end = None
    drawing = False

    def run_decoder_box():
        if rect_start is None or rect_end is None: return
        x1_d,y1_d = rect_start; x2_d,y2_d = rect_end
        x1,y1 = int(x1_d/scale), int(y1_d/scale)
        x2,y2 = int(x2_d/scale), int(y2_d/scale)
        x1,x2 = sorted((x1,x2)); y1,y2 = sorted((y1,y2))
        pts,lbl = prepare_rectangle((x1,y1,x2,y2),(H_org,W_org),(enc_h,enc_w))
        pts = np.ascontiguousarray(pts.astype(np.float32))
        lbl = np.ascontiguousarray(lbl.astype(np.float32))
        dec_inputs = {
            DEC_KEYS["point_coords"]:     pts,
            DEC_KEYS["point_labels"]:     lbl,
            DEC_KEYS["image_embed"]:      img_embed,
            DEC_KEYS["high_res_feats_0"]: feats0,
            DEC_KEYS["high_res_feats_1"]: feats1,
        }
        t = time.time()
        obj_ptr, mask_for_mem, pred_low = sess_dec.run(None, dec_inputs)
        print(f"[INFO] Decoder time : {(time.time()-t)*1000:.1f} ms")
        mask256 = pred_low[0,0]
        mask = cv2.resize(mask256,(W_org,H_org))
        mask255 = (mask>0).astype(np.uint8)*255
        overlay = green_overlay(img_bgr, mask255)
        disp = cv2.resize(overlay,(disp_w,disp_h))
        cv2.rectangle(disp, rect_start, rect_end,(0,255,255),2)
        cv2.imshow("SAM-2 Demo", disp)

    def mouse_cb(event,x,y,flags,param):
        nonlocal rect_start, rect_end, drawing
        if not mode_bbox:
            if event==cv2.EVENT_MBUTTONDOWN:
                points.clear(); labels.clear(); cv2.imshow("SAM-2 Demo", disp_base)
            elif event==cv2.EVENT_LBUTTONDOWN:
                points.append((int(x/scale), int(y/scale))); labels.append(1); run_decoder_points()
            elif event==cv2.EVENT_RBUTTONDOWN:
                points.append((int(x/scale), int(y/scale))); labels.append(0); run_decoder_points()
            return
        if event in (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_LBUTTONDBLCLK):
            rect_start = rect_end = None; cv2.imshow("SAM-2 Demo", disp_base); return
        if event==cv2.EVENT_LBUTTONDOWN:
            drawing=True; rect_start=rect_end=(x,y); cv2.imshow("SAM-2 Demo", disp_base)
        elif event==cv2.EVENT_MOUSEMOVE and drawing:
            rect_end=(x,y); vis=disp_base.copy()
            cv2.rectangle(vis,rect_start,rect_end,(0,255,255),2); cv2.imshow("SAM-2 Demo",vis)
        elif event==cv2.EVENT_LBUTTONUP and drawing:
            drawing=False; rect_end=(x,y); run_decoder_box()

    cv2.namedWindow("SAM-2 Demo", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("SAM-2 Demo", mouse_cb)
    cv2.imshow("SAM-2 Demo", disp_base)
    print("[INFO] Interactive mode ready.  ESC to quit.")
    while True:
        if cv2.waitKey(20) & 0xFF == 27: break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()