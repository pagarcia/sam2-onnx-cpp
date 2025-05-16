"""
onnx_test_video_bounding_box.py
────────────────────────────────────────────────────────────────────────────
Segment an entire video with SAM-2 ONNX using a single bounding-box prompt
drawn interactively on the first frame.

Timing metrics (ms) for every frame:
    Enc   : encoder
    Attn  : memory-attention (≥2nd frame)
    Dec   : decoder
    MemEnc: memory-encoder
"""

import os, sys, time, argparse
import cv2
import numpy as np
import onnxruntime as ort
from onnxruntime import InferenceSession
from PyQt5 import QtWidgets


# ───────────────────────── helpers ──────────────────────────
def print_system_info():
    print("[INFO] OS :", sys.platform)
    print("[INFO] ONNX Runtime providers :", ort.get_available_providers())


def prepare_image(frame_bgr, enc_shape):
    h_enc, w_enc = enc_shape
    rgb = cv2.cvtColor(cv2.resize(frame_bgr, (w_enc, h_enc)), cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb /= np.float32(255.0)
    rgb = (rgb - np.array([0.485,0.456,0.406], np.float32)) / np.array([0.229,0.224,0.225], np.float32)
    tensor = np.transpose(rgb, (2,0,1))[np.newaxis, :].astype(np.float32)
    return tensor, frame_bgr.shape[:2]          # original (H,W)


def prepare_box_prompt(rect, img_sz, enc_sz):
    if rect is None:
        return None, None
    x1,y1,x2,y2 = rect
    h_org,w_org = img_sz
    h_enc,w_enc = enc_sz
    pts = np.array([[x1,y1],[x2,y2]], np.float32)
    pts[:,0] = (pts[:,0]/w_org)*w_enc
    pts[:,1] = (pts[:,1]/h_org)*h_enc
    lbls = np.array([2.,3.], np.float32)
    return pts[np.newaxis,...], lbls[np.newaxis,...]


def decode(sess_dec, coords, labels, embed, f0, f1):
    if coords is None:
        coords = np.zeros((1,0,2), np.float32)
        labels = np.zeros((1,0),   np.float32)
    return sess_dec.run(
        None,
        {"point_coords":coords,
         "point_labels":labels,
         "image_embed":embed,
         "high_res_feats_0":f0,
         "high_res_feats_1":f1}
    )   # obj_ptr, mask_for_mem, pred_mask


# ────────────── first-frame interactive bounding box ─────────────
def interactive_select_box(first_bgr, sess_enc, sess_dec, enc_shape):
    tensor,(h_org,w_org)=prepare_image(first_bgr, enc_shape)
    enc_out=sess_enc.run(None,{sess_enc.get_inputs()[0].name:tensor})
    embed,f0,f1=enc_out[:3]

    disp_max=1200
    scale=min(1.0,disp_max/max(w_org,h_org))
    disp_w,disp_h=int(w_org*scale),int(h_org*scale)
    base=cv2.resize(first_bgr,(disp_w,disp_h))

    rect_s=rect_e=None; drawing=False

    def show(mask=None):
        vis=base.copy()
        if mask is not None:
            m=cv2.resize(mask,(disp_w,disp_h),cv2.INTER_NEAREST)
            green=np.zeros_like(vis); green[m>0]=(0,255,0)
            vis=cv2.addWeighted(vis,1.0,green,0.5,0)
        if rect_s and rect_e: cv2.rectangle(vis,rect_s,rect_e,(0,255,255),2)
        cv2.imshow("First Frame – SAM2",vis)

    def run():
        if not(rect_s and rect_e): show(); return
        x1_d,y1_d=rect_s; x2_d,y2_d=rect_e
        x1,y1,x2,y2=(int(x1_d/scale),int(y1_d/scale),int(x2_d/scale),int(y2_d/scale))
        x1,x2=sorted((x1,x2)); y1,y2=sorted((y1,y2))
        pts,lbls=prepare_box_prompt((x1,y1,x2,y2),(h_org,w_org),enc_shape)
        _,_,pred=decode(sess_dec,pts,lbls,embed,f0,f1)
        show((pred[0,0]>0).astype(np.uint8))

    def cb(event,x,y,flags,param):
        nonlocal rect_s,rect_e,drawing
        if event==cv2.EVENT_LBUTTONDOWN:
            drawing=True; rect_s=rect_e=(x,y); show()
        elif event==cv2.EVENT_MOUSEMOVE and drawing:
            rect_e=(x,y); show()
        elif event==cv2.EVENT_LBUTTONUP:
            drawing=False; rect_e=(x,y); run()
        elif event in (cv2.EVENT_RBUTTONDOWN,cv2.EVENT_LBUTTONDBLCLK):
            rect_s=rect_e=None; show()

    cv2.namedWindow("First Frame – SAM2"); cv2.setMouseCallback("First Frame – SAM2",cb)
    show(); print("[INFO] Draw box, release → preview. ESC/Enter to continue.")
    while True:
        if cv2.waitKey(20)&0xFF in (27,13): break
    cv2.destroyAllWindows()

    if rect_s and rect_e:
        x1_d,y1_d=rect_s; x2_d,y2_d=rect_e
        x1,y1,x2,y2=(int(x1_d/scale),int(y1_d/scale),int(x2_d/scale),int(y2_d/scale))
        x1,x2=sorted((x1,x2)); y1,y2=sorted((y1,y2))
        box=(x1,y1,x2,y2)
    else:
        box=None
    return box, embed, f0, f1, (h_org,w_org)


# ─────────────────────── processing loop ────────────────────────
def process_video(args):
    ckpt=os.path.join("checkpoints",args.model_size)
    sess_enc=InferenceSession(os.path.join(ckpt,f"image_encoder_{args.model_size}.onnx"))
    sess_dec=InferenceSession(os.path.join(ckpt,f"image_decoder_{args.model_size}.onnx"))
    sess_men=InferenceSession(os.path.join(ckpt,f"memory_encoder_{args.model_size}.onnx"))
    sess_mat=InferenceSession(os.path.join(ckpt,f"memory_attention_{args.model_size}.onnx"))
    enc_h,enc_w=sess_enc.get_inputs()[0].shape[2:]

    cap=cv2.VideoCapture(args.video)
    if not cap.isOpened(): sys.exit("ERROR: cannot open video")
    fps=cap.get(cv2.CAP_PROP_FPS)
    w_org,h_org=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_file=os.path.splitext(args.video)[0]+"_mask_overlay.mkv"
    writer=cv2.VideoWriter(out_file,cv2.VideoWriter_fourcc(*'XVID'),fps,(w_org,h_org))
    if not writer.isOpened(): sys.exit("ERROR: cannot open writer")

    # interactive first frame
    ret,first_bgr=cap.read()
    if not ret: sys.exit("ERROR: empty video")
    cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    box, embed0, f0_0, f1_0, (h_org, w_org) = interactive_select_box(first_bgr,sess_enc,sess_dec,(enc_h,enc_w))
    pts0,lbls0 = prepare_box_prompt(box,(h_org,w_org),(enc_h,enc_w)) if box else (None,None)

    mem_feats=mem_pos=None
    fidx=0
    while True:
        ret,frame=cap.read()
        if not ret or (args.max_frames>0 and fidx>=args.max_frames): break

        # ── encoder ──────────────────────────────────────────────
        t_enc=time.time()
        if fidx==0:
            embed,f0,f1 = embed0, f0_0, f1_0
        else:
            tensor,_ = prepare_image(frame,(enc_h,enc_w))
            enc_out = sess_enc.run(None,{sess_enc.get_inputs()[0].name:tensor})
            embed,f0,f1,flat_feat,vis_pos = enc_out
        enc_ms=(time.time()-t_enc)*1000

        # ── memory attention (frames >0) ─────────────────────────
        t_mat=time.time()
        if fidx>0 and mem_feats is not None:
            attn_in={
                "current_vision_feat":      embed,
                "current_vision_pos_embed": vis_pos,
                "memory_0": np.zeros((0,256),np.float32),
                "memory_1": mem_feats,
                "memory_pos_embed": mem_pos
            }
            embed = sess_mat.run(None,attn_in)[0]
            mat_ms=(time.time()-t_mat)*1000
        else:
            mat_ms=0.0

        # ── decoder ──────────────────────────────────────────────
        t_dec=time.time()
        if fidx==0:
            _,mask_for_mem,pred = decode(sess_dec,pts0,lbls0,embed,f0,f1)
        else:
            _,mask_for_mem,pred = decode(sess_dec,None,None,embed,f0,f1)
        dec_ms=(time.time()-t_dec)*1000

        # ── memory encoder ──────────────────────────────────────
        t_men=time.time()
        men_out=sess_men.run(None,{"mask_for_mem":mask_for_mem[:,0:1],"pix_feat":embed})
        mem_feats,mem_pos,_=men_out
        men_ms=(time.time()-t_men)*1000

        # ── overlay & write ─────────────────────────────────────
        mask=cv2.resize((pred[0,0]>0).astype(np.uint8),(w_org,h_org),cv2.INTER_LINEAR)
        green=np.zeros_like(frame); green[mask>0]=(0,255,0)
        writer.write(cv2.addWeighted(frame,1.0,green,0.5,0))

        # ── log ─────────────────────────────────────────────────
        if fidx==0:
            print(f"Frame {fidx:03d} - Enc:{enc_ms:.1f} | Dec:{dec_ms:.1f} | MemEnc:{men_ms:.1f}")
        else:
            print(f"Frame {fidx:03d} - Enc:{enc_ms:.1f} | Attn:{mat_ms:.1f} | Dec:{dec_ms:.1f} | MemEnc:{men_ms:.1f}")
        fidx+=1

    cap.release(); writer.release()
    print(f"Done! Wrote {fidx} frames to {out_file}")


# ───────────────────────── CLI wrapper ──────────────────────────
def main():
    print_system_info()
    ap=argparse.ArgumentParser(description="Video segmentation with single bounding-box prompt.")
    ap.add_argument("--model_size",default="tiny",choices=["base_plus","large","small","tiny"])
    ap.add_argument("--max_frames",type=int,default=0,help="0 = all")
    args=ap.parse_args()

    # choose video via dialog
    app=QtWidgets.QApplication(sys.argv)
    vid,_=QtWidgets.QFileDialog.getOpenFileName(
        None,"Select Video","","Videos (*.mp4 *.mkv *.avi *.mov *.m4v);;All files (*.*)"
    )
    if not vid: sys.exit("No video selected – exiting.")
    args.video=vid; print(f"[INFO] Selected video: {vid}")

    process_video(args)

if __name__=="__main__":
    main()
