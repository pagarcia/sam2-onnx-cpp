// sam2-onnx-cpp/cpp/src/onnx_test_image.cpp
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

/* ────────────────────────────  helper: green overlay  ───────────────────────── */
static cv::Mat overlayMask(const cv::Mat& img, const cv::Mat& mask)
{
    cv::Mat overlay;         img.copyTo(overlay);
    cv::Mat green(img.size(), img.type(), cv::Scalar(0,255,0));
    green.copyTo(overlay, mask);                       // only where mask!=0
    const double alpha = .3;
    cv::Mat blended;
    cv::addWeighted(img, 1.0 - alpha, overlay, alpha, 0, blended);
    return blended;
}

/* ─────────────────────────────  state & prompt mode  ────────────────────────── */
enum class PromptMode { SEED_POINTS, BOUNDING_BOX };

struct AppState
{
    /* inference */
    SAM2    sam;
    cv::Mat original;              // original BGR
    cv::Mat display;               // current shown frame
    SAM2Size origSize;             // in pixels

    PromptMode mode = PromptMode::SEED_POINTS;

    /* seed-point prompt  ----------------------------------------------------- */
    vector<SAM2Point> points;      // original-space coords
    vector<int>       labels;      // 1=FG, 0=BG   (same size as points)

    /* bounding-box prompt  --------------------------------------------------- */
    bool     drawing      = false;
    SAM2Rect rect;                 // x,y,w,h – w/h may be negative while dragging
    bool     hasFinalRect = false;

    void resetSeeds() { points.clear(); labels.clear(); }
    void resetRect () { drawing=false; hasFinalRect=false; rect=SAM2Rect(); }
};

/* ───────────────────────  recompute mask & refresh GUI  ─────────────────────── */
static void updateDisplay(AppState* st, bool forceRunBBox = false)
{
    st->display = st->original.clone();

    // (1)  Draw current rectangle preview (bbox mode)
    if (st->mode == PromptMode::BOUNDING_BOX && (st->drawing || st->hasFinalRect))
    {
        SAM2Rect r = st->rect;
        if (r.width<0)  { r.x += r.width;  r.width  = -r.width; }
        if (r.height<0) { r.y += r.height; r.height = -r.height; }
        cv::rectangle(st->display, cv::Rect(r.x, r.y, r.width, r.height),
                      cv::Scalar(0,255,255), 2);
    }

    // (2)  Should we run the segmenter?
    bool needRun = false;
    if (st->mode == PromptMode::SEED_POINTS)
        needRun = !st->points.empty();
    else                    /* BOUNDING_BOX */
        needRun = forceRunBBox && st->hasFinalRect;

    if (needRun)
    {
        SAM2Prompts prm;

        if (st->mode == PromptMode::SEED_POINTS)
        {
            prm.points      = st->points;
            prm.pointLabels = st->labels;
        }
        else
        {
            prm.rects.push_back(st->rect);          // SAM2 normalises inside
        }

        st->sam.setPrompts(prm, st->origSize);

        auto t0 = high_resolution_clock::now();
        Image<float> mask = st->sam.inferSingleFrame(st->origSize);
        auto ms = duration_cast<milliseconds>(high_resolution_clock::now()-t0).count();
        cout << "[INFO] Segmentation took " << ms << " ms\n";

        cv::Mat maskCv = CVHelpers::imageToCvMatWithType(mask, CV_8UC1, 255.0);
        st->display = overlayMask(st->display, maskCv);
    }

    // (3)  Draw seeds (seed-point mode)
    if (st->mode == PromptMode::SEED_POINTS)
    {
        for (size_t i=0;i<st->points.size();++i)
        {
            cv::Scalar col = (st->labels[i]==1) ? cv::Scalar(0,0,255)
                                                : cv::Scalar(255,0,0);
            cv::circle(st->display, cv::Point(st->points[i].x, st->points[i].y),
                       5, col, -1);
        }
    }

    cv::imshow("SAM-2 Interactive", st->display);
}

/* ──────────────────────────  unified mouse callback  ───────────────────────── */
static void onMouse(int event,int x,int y,int,void* ud)
{
    auto st = static_cast<AppState*>(ud);

    /* ─────────── 1) prompt = seed-points ─────────── */
    if (st->mode == PromptMode::SEED_POINTS)
    {
        if (event == cv::EVENT_MBUTTONDOWN) { st->resetSeeds(); updateDisplay(st); return; }
        if (event!=cv::EVENT_LBUTTONDOWN && event!=cv::EVENT_RBUTTONDOWN) return;

        bool fg = (event==cv::EVENT_LBUTTONDOWN);
        cout << "[INFO] " << (fg?"Positive":"Negative")
             << " point ("<<x<<","<<y<<")\n";

        st->points.push_back(SAM2Point(x,y));
        st->labels.push_back(fg?1:0);
        updateDisplay(st);
        return;
    }

    /* ─────────── 2) prompt = bounding-box ─────────── */
    if (st->mode == PromptMode::BOUNDING_BOX)
    {
        if (event == cv::EVENT_RBUTTONDOWN || event == cv::EVENT_MBUTTONDOWN)
        { st->resetRect(); updateDisplay(st); return; }

        if (event == cv::EVENT_LBUTTONDOWN)
        {
            st->drawing = true; st->hasFinalRect = false;
            st->rect.x = x; st->rect.y = y; st->rect.width = st->rect.height = 0;
            updateDisplay(st);
        }
        else if (event == cv::EVENT_MOUSEMOVE && st->drawing)
        {
            st->rect.width  = x - st->rect.x;
            st->rect.height = y - st->rect.y;
            updateDisplay(st);
        }
        else if (event == cv::EVENT_LBUTTONUP && st->drawing)
        {
            st->drawing=false; st->hasFinalRect=true;
            st->rect.width  = x - st->rect.x;
            st->rect.height = y - st->rect.y;
            cout << "[INFO] Box finalised: ("<<st->rect.x<<","<<st->rect.y
                 <<") to ("<<x<<","<<y<<")\n";
            updateDisplay(st, /*forceRunBBox=*/true);
        }
    }
}

/* ───────────────────────────────  main runner  ─────────────────────────────── */
int runOnnxTestImage(int argc,char** argv)
{
    /* ---------------  CLI defaults & parse  --------------- */
    string encoderPath = "image_encoder.onnx";
    string decoderPath = "image_decoder.onnx";
    string imagePath;                     // empty → file dialog
    int    threads   = (int)thread::hardware_concurrency(); if(threads<=0) threads=4;
    PromptMode mode  = PromptMode::SEED_POINTS;             // default

    for(int i=2;i<argc;++i)   // argv[1] is "--onnx_test_image"
    {
        string a = argv[i];
        if((a=="--encoder") && i+1<argc)       encoderPath = argv[++i];
        else if((a=="--decoder") && i+1<argc)  decoderPath = argv[++i];
        else if((a=="--image")   && i+1<argc)  imagePath   = argv[++i];
        else if((a=="--threads") && i+1<argc)  threads     = stoi(argv[++i]);
        else if((a=="--prompt")  && i+1<argc)
        {
            string s = argv[++i];
            if      (s=="seed_points")  mode = PromptMode::SEED_POINTS;
            else if (s=="bounding_box") mode = PromptMode::BOUNDING_BOX;
            else { cerr<<"[ERROR] --prompt must be seed_points|bounding_box\n"; return 1; }
        }
        else if(a=="--help"||a=="-h")
        {
            cout <<
            "Usage:\n  Segment --onnx_test_image [options]\n\n"
            "Options:\n"
            "  --encoder  image_encoder.onnx     (default: image_encoder.onnx)\n"
            "  --decoder  image_decoder.onnx     (default: image_decoder.onnx)\n"
            "  --image    myimage.jpg            (file dialog if omitted)\n"
            "  --threads  N                      (#CPU threads, default: HW-concurrency)\n"
            "  --prompt   seed_points|bounding_box   (default: seed_points)\n\n"
            "L-click  (seed)   = FG point\n"
            "R-click  (seed)   = BG point\n"
            "M-click  (seed)   = reset points\n\n"
            "L-drag   (bbox)   = draw rectangle\n"
            "R/M-click(bbox)   = reset rectangle\n"
            "ESC                 quit\n";
            return 0;
        }
    }

    /* ---------------  select image if needed  --------------- */
    if(imagePath.empty())
    {
        const wchar_t* filter = L"Images\0*.jpg;*.jpeg;*.png;*.bmp\0All\0*.*\0";
        string chosen = openFileDialog(filter, L"Select an image");
        if(chosen.empty()){ cerr<<"[ERROR] No file chosen\n"; return 1; }
        imagePath = chosen;
    }

    cout<<"[INFO] encoder="<<encoderPath<<"\n"
        <<"       decoder="<<decoderPath<<"\n"
        <<"       image  ="<<imagePath<<"\n"
        <<"       prompt ="<<(mode==PromptMode::SEED_POINTS?"seed_points":"bounding_box")<<"\n"
        <<"       threads="<<threads<<"\n\n";

    /* ---------------  read image  --------------- */
    cv::Mat imgBGR = cv::imread(imagePath);
    if(imgBGR.empty()){ cerr<<"[ERROR] could not load "<<imagePath<<"\n"; return 1; }

    /* ---------------  init SAM-2 (GPU if available)  --------------- */
    bool cudaAvail=false;
    // OLD (kept here for reference – relied on GetAvailableProviders, which crashes in PCs with no GPU)
    // for(auto&p:Ort::GetAvailableProviders()) if(p=="CUDAExecutionProvider") cudaAvail=true;
    cudaAvail = SAM2::hasCudaDriver();
    string device = cudaAvail ? "cuda:0" : "cpu";
    cout<<"[INFO] Initialising on "<<device<<"\n";

    AppState st;
    if(!st.sam.initialize(encoderPath,decoderPath,threads,device))
    { cerr<<"[ERROR] SAM2 init failed\n"; return 1; }

    st.original  = imgBGR;
    st.origSize  = SAM2Size(imgBGR.cols, imgBGR.rows);
    st.mode      = mode;

    /* ---------------  preprocess  (encoder)  --------------- */
    auto tEnc0 = high_resolution_clock::now();
    if(!st.sam.preprocessImage(CVHelpers::normalizeRGB(imgBGR,255.0)))
    { cerr<<"[ERROR] preprocessImage failed\n"; return 1; }
    cout<<"[INFO] Encoder time "<< duration_cast<milliseconds>(
            high_resolution_clock::now()-tEnc0).count() <<" ms\n";

    /* ---------------  interactive UI  --------------- */
    st.display = st.original.clone();
    cv::namedWindow("SAM-2 Interactive", cv::WINDOW_AUTOSIZE);
    cv::imshow("SAM-2 Interactive", st.display);
    cv::setMouseCallback("SAM-2 Interactive", onMouse, &st);

    cout << "\n[INFO] Controls (" << (mode==PromptMode::SEED_POINTS? "seed-points":"bounding-box") << " mode):\n";
    if(mode==PromptMode::SEED_POINTS)
        cout << "  L-click = FG, R-click = BG, M-click = reset\n";
    else
        cout << "  L-drag  = draw box, R/M-click = reset box\n";
    cout << "  ESC = quit\n\n";

    while(cv::waitKey(50)!=27) {}    // ESC exits

    return 0;
}
