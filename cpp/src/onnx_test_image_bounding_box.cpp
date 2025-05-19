// ─────────────────────── onnx_test_image_bounding_box.cpp ───────────────────────
#include "openFileDialog.h"
#include "SAM2.h"
#include "CVHelpers.h"

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include <iostream>
#include <thread>
#include <chrono>

using namespace std;
using namespace std::chrono;

/*────────────────────────── small helpers ──────────────────────────*/
/*────────────────────────── small helpers ──────────────────────────*/
// Replace the current overlayMask with this:
static cv::Mat overlayMask(const cv::Mat& img, const cv::Mat& mask)
{
    cv::Mat overlay;
    img.copyTo(overlay);                              // start from original

    // Paint bright-green only on masked pixels
    cv::Mat green(img.size(), img.type(), cv::Scalar(0, 255, 0));
    green.copyTo(overlay, mask);                      // mask is 0/255 (8UC1)

    // Alpha-blend overlay back onto the original image
    const double alpha = 0.3;                         // 30 % opacity
    cv::Mat blended;
    cv::addWeighted(img, 1.0 - alpha,
                    overlay, alpha,
                    0.0, blended);
    return blended;
}

/*────────────────────────── app state ─────────────────────────────*/
struct AppState {
    SAM2 sam;
    cv::Mat original;           // original BGR
    cv::Mat display;            // current display image
    SAM2Size origSize;          // original size

    // bounding-box (in original coords)
    bool   drawing=false;
    SAM2Rect rect;              // x,y,w,h  (w,h may be negative while dragging)
    bool   hasFinalRect=false;  // becomes true on mouse-up

    // convenience
    void resetRect() { drawing=false; hasFinalRect=false; rect=SAM2Rect(); }
};

/*──────────────────── segmentation & display update ───────────────────*/
static void updateDisplay(AppState* st, bool runSegmentation)
{
    // start with a copy of the original image
    st->display = st->original.clone();

    // draw current rectangle in yellow
    if (st->drawing || st->hasFinalRect) {
        SAM2Rect r = st->rect;
        // normalize so width & height are positive
        if (r.width < 0) { r.x += r.width; r.width  = -r.width; }
        if (r.height< 0) { r.y += r.height; r.height = -r.height; }
        cv::rectangle(st->display,
                      cv::Rect(r.x, r.y, r.width, r.height),
                      cv::Scalar(0,255,255), 2);
    }

    if (runSegmentation && st->hasFinalRect) {
        // build prompt: two corners => labels 2 & 3
        SAM2Prompts p;
        p.rects.push_back(st->rect);      // SAM2::setPrompts() will scale & emit labels 2/3

        st->sam.setPrompts(p, st->origSize);

        auto t0 = high_resolution_clock::now();
        Image<float> mask = st->sam.inferSingleFrame(st->origSize);
        auto ms  = duration_cast<milliseconds>(high_resolution_clock::now() - t0).count();
        cout << "[INFO] Segmentation took " << ms << " ms\n";

        cv::Mat maskCv = CVHelpers::imageToCvMatWithType(mask, CV_8UC1, 255.0);
        st->display = overlayMask(st->display, maskCv);
        // draw rectangle again so it stays on top
        SAM2Rect r = st->rect;
        if (r.width<0){r.x+=r.width; r.width=-r.width;}
        if (r.height<0){r.y+=r.height;r.height=-r.height;}
        cv::rectangle(st->display, cv::Rect(r.x,r.y,r.width,r.height),
                      cv::Scalar(0,255,255), 2);
    }

    cv::imshow("SAM-2 Bounding-Box", st->display);
}

/*──────────────────────── mouse callback ──────────────────────────*/
static void onMouse(int event,int x,int y,int,void* ud)
{
    auto st = static_cast<AppState*>(ud);

    if (event == cv::EVENT_RBUTTONDOWN || event == cv::EVENT_MBUTTONDOWN) {
        cout << "[INFO] Resetting rectangle\n";
        st->resetRect();
        updateDisplay(st, /*runSegmentation=*/false);
        return;
    }

    if (event == cv::EVENT_LBUTTONDOWN) {
        st->drawing=true; st->hasFinalRect=false;
        st->rect.x = x; st->rect.y = y;
        st->rect.width = 0; st->rect.height = 0;
        updateDisplay(st,false);
    }
    else if (event == cv::EVENT_MOUSEMOVE && st->drawing) {
        st->rect.width  = x - st->rect.x;
        st->rect.height = y - st->rect.y;
        updateDisplay(st,false);          // live preview only
    }
    else if (event == cv::EVENT_LBUTTONUP && st->drawing) {
        st->drawing=false; st->hasFinalRect=true;
        st->rect.width  = x - st->rect.x;
        st->rect.height = y - st->rect.y;
        cout << "[INFO] Box finalized: ("<<st->rect.x<<","<<st->rect.y
             <<") – ("<<x<<","<<y<<")\n";
        updateDisplay(st,true);           // run segmentation once
    }
}

/*─────────────────── main / CLI parsing (minimal) ───────────────────*/
int runOnnxTestImageBoundingBox(int argc,char** argv)
{
    string encoderPath="image_encoder.onnx";
    string decoderPath="image_decoder.onnx";
    string imagePath;               // empty => dialog
    int    threads=(int)std::thread::hardware_concurrency();
    if (threads<=0) threads=4;

    for(int i=1;i<argc;i++){
        string a=argv[i];
        if((a=="--encoder")&&i+1<argc) encoderPath=argv[++i];
        else if((a=="--decoder")&&i+1<argc) decoderPath=argv[++i];
        else if((a=="--image")  &&i+1<argc) imagePath =argv[++i];
        else if((a=="--threads")&&i+1<argc) threads   =stoi(argv[++i]);
        else if(a=="--help"||a=="-h"){
            cout << "Usage:\n  "<<argv[0]<<" [--encoder E.onnx] [--decoder D.onnx] "
                 <<"[--image img.jpg] [--threads N]\n";
            return 0;
        }
    }

    if(imagePath.empty()){
        cout<<"[INFO] No --image: opening file dialog…\n";
        const wchar_t* filter = L"Images\0*.jpg;*.jpeg;*.png;*.bmp\0All\0*.*\0";
        string chosen = openFileDialog(filter,L"Select an image");
        if(chosen.empty()){ cerr<<"[ERROR] No file chosen\n"; return 1; }
        imagePath=chosen;
    }

    cout<<"[INFO] encoder="<<encoderPath<<"\n"
        <<"       decoder="<<decoderPath<<"\n"
        <<"       image  ="<<imagePath<<"\n\n";

    // ── load image ────────────────────────────────────────────────
    cv::Mat imgBGR=cv::imread(imagePath);
    if(imgBGR.empty()){ cerr<<"[ERROR] could not load "<<imagePath<<"\n"; return 1; }

    // ── init SAM-2 (CPU by default; GPU if CUDA provider exists) ──
    bool cudaAvail=false;
    for(auto&p:Ort::GetAvailableProviders()) if(p=="CUDAExecutionProvider") cudaAvail=true;
    string device = cudaAvail ? "cuda:0" : "cpu";
    cout<<"[INFO] Initializing on "<<device<<" …\n";

    AppState st;
    if(!st.sam.initialize(encoderPath,decoderPath,threads,device)){
        cerr<<"[ERROR] SAM2 init failed\n"; return 1;
    }

    st.original   = imgBGR;
    st.origSize   = SAM2Size(imgBGR.cols,imgBGR.rows);

    // first pass: encoder only
    auto tEnc0 = high_resolution_clock::now();
    if(!st.sam.preprocessImage(CVHelpers::normalizeRGB(imgBGR,255.0))){
        cerr<<"[ERROR] preprocessImage failed\n"; return 1;
    }
    cout<<"[INFO] Encoder time "<< duration_cast<milliseconds>(
             high_resolution_clock::now()-tEnc0).count() <<" ms\n";

    // ── interactive UI ───────────────────────────────────────────
    st.display = st.original.clone();
    cv::namedWindow("SAM-2 Bounding-Box", cv::WINDOW_AUTOSIZE);
    cv::imshow("SAM-2 Bounding-Box", st.display);
    cv::setMouseCallback("SAM-2 Bounding-Box", onMouse, &st);

    cout<<"[INFO] Drag left button to draw a rectangle.\n"
        <<"       Release to segment; right-click to reset; ESC to quit.\n";

    while(true){
        if(cv::waitKey(50)==27) break;   // ESC
    }
    return 0;
}
