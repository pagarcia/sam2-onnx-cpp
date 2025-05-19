// ──────────────────────────────  onnx_test_video.cpp  ───────────────────────────
#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include "SAM2.h"
#include "openFileDialog.h"
#include "CVHelpers.h"

/* ════════════════════════════════════════════════════════════════════════════ *
 *                          0.  COMMON HELPERS                                  *
 * ════════════════════════════════════════════════════════════════════════════ */
enum class PromptMode { SEED_POINTS, BOUNDING_BOX };

static cv::Mat overlayMask(const cv::Mat& img, const cv::Mat& maskGray)
{
    cv::Mat overlay;  img.copyTo(overlay);
    cv::Mat green(img.size(), img.type(), cv::Scalar(0,255,0));
    green.copyTo(overlay, maskGray);                     // only where mask!=0
    cv::Mat blended;
    cv::addWeighted(img, 0.7, overlay, 0.3, 0, blended);
    return blended;
}

/* ════════════════════════════════════════════════════════════════════════════ *
 *                       1.  INTERACTIVE FIRST-FRAME                            *
 * ════════════════════════════════════════════════════════════════════════════ */
struct VideoAppState
{
    SAM2*  sam = nullptr;

    cv::Mat firstFrame;          // BGR
    cv::Mat displayFrame;        // what the user sees
    SAM2Size originalSize;       // W×H of original frame

    PromptMode mode = PromptMode::SEED_POINTS;

    /* Seed-point prompt ---------------------------------------------------- */
    std::vector<SAM2Point> points;
    std::vector<int>       labels;      // 1=FG, 0=BG
    void resetSeeds() { points.clear(); labels.clear(); }

    /* Bounding-box prompt -------------------------------------------------- */
    bool     drawing      = false;
    bool     hasRect      = false;
    SAM2Rect rect;                  // x,y,w,h  — w/h may be negative
    void resetRect() { drawing=false; hasRect=false; rect=SAM2Rect(); }
};

/* Re-runs decoder each time the user edits the first-frame prompt. */
static void updateFirstFrameDisplay(VideoAppState* st, bool forceRunBBox=false)
{
    st->displayFrame = st->firstFrame.clone();

    /* draw current rectangle (bbox mode) */
    if (st->mode==PromptMode::BOUNDING_BOX && (st->drawing || st->hasRect))
    {
        SAM2Rect r = st->rect;
        if (r.width <0){ r.x+=r.width;  r.width =-r.width;  }
        if (r.height<0){ r.y+=r.height; r.height=-r.height; }
        cv::rectangle(st->displayFrame,
                      cv::Rect(r.x,r.y,r.width,r.height),
                      cv::Scalar(0,255,255), 2);
    }

    /* decide if we need a segmentation run */
    bool needRun=false;
    if (st->mode==PromptMode::SEED_POINTS)
        needRun = !st->points.empty();
    else
        needRun = forceRunBBox && st->hasRect;

    if (needRun)
    {
        SAM2Prompts p;
        if (st->mode==PromptMode::SEED_POINTS)
        {
            p.points      = st->points;
            p.pointLabels = st->labels;
        }
        else
            p.rects.push_back(st->rect);

        st->sam->setPrompts(p, st->originalSize);

        auto t0 = std::chrono::high_resolution_clock::now();
        Image<float> mask = st->sam->inferSingleFrame(st->originalSize);
        double ms = std::chrono::duration<double,std::milli>(
                        std::chrono::high_resolution_clock::now()-t0).count();
        std::cout << "[INFO] Partial decode => " << ms << " ms\n";

        st->displayFrame = overlayMask(
            st->displayFrame,
            CVHelpers::imageToCvMatWithType(mask, CV_8UC1, 255.0) );
    }

    /* draw point seeds on top */
    if (st->mode==PromptMode::SEED_POINTS)
    {
        for (size_t i=0;i<st->points.size();++i)
        {
            cv::Scalar col = (st->labels[i]==1)? cv::Scalar(0,0,255)
                                               : cv::Scalar(255,0,0);
            cv::circle(st->displayFrame,
                       cv::Point(st->points[i].x, st->points[i].y),
                       5, col, -1);
        }
    }

    cv::imshow("First Frame - Interactive", st->displayFrame);
}

/* unified mouse callback for both prompt types */
static void onMouseFirstFrame(int ev,int x,int y,int,void* userdata)
{
    auto st = static_cast<VideoAppState*>(userdata);

    /* ────────────── SEED-POINTS MODE ────────────── */
    if (st->mode==PromptMode::SEED_POINTS)
    {
        if (ev==cv::EVENT_MBUTTONDOWN) { st->resetSeeds(); updateFirstFrameDisplay(st); return;}
        if (ev!=cv::EVENT_LBUTTONDOWN && ev!=cv::EVENT_RBUTTONDOWN) return;

        bool fg = (ev==cv::EVENT_LBUTTONDOWN);
        st->points.emplace_back(x,y);
        st->labels.push_back(fg?1:0);
        updateFirstFrameDisplay(st);
        return;
    }

    /* ────────────── BOUNDING-BOX MODE ───────────── */
    if (st->mode==PromptMode::BOUNDING_BOX)
    {
        if (ev==cv::EVENT_RBUTTONDOWN || ev==cv::EVENT_MBUTTONDOWN)
        { st->resetRect(); updateFirstFrameDisplay(st); return; }

        if (ev==cv::EVENT_LBUTTONDOWN)
        {
            st->drawing=true; st->hasRect=false;
            st->rect.x=x; st->rect.y=y; st->rect.width=st->rect.height=0;
            updateFirstFrameDisplay(st);
        }
        else if (ev==cv::EVENT_MOUSEMOVE && st->drawing)
        {
            st->rect.width  = x - st->rect.x;
            st->rect.height = y - st->rect.y;
            updateFirstFrameDisplay(st);
        }
        else if (ev==cv::EVENT_LBUTTONUP && st->drawing)
        {
            st->drawing=false; st->hasRect=true;
            st->rect.width  = x - st->rect.x;
            st->rect.height = y - st->rect.y;
            updateFirstFrameDisplay(st, /*forceRunBBox=*/true);
        }
    }
}

/* ════════════════════════════════════════════════════════════════════════════ *
 *                               2.  MAIN RUNNER                                *
 * ════════════════════════════════════════════════════════════════════════════ */
static void printVideoUsage(const char* argv0)
{
    std::cout <<
"Usage:  " << argv0 << " --onnx_test_video [options]\n\n"
"Options:\n"
"  --encoder   image_encoder.onnx\n"
"  --decoder   image_decoder.onnx\n"
"  --memattn   memory_attention.onnx\n"
"  --memenc    memory_encoder.onnx\n"
"  --video     clip.mkv / clip.mp4 (file-dialog if omitted)\n"
"  --threads   N             (#CPU threads)\n"
"  --max_frames N            (early stop)\n"
"  --prompt   seed_points | bounding_box   (default: seed_points)\n\n"
"Interactive first frame:\n"
"  seed_points:  L-click=FG   R-click=BG   M-click=reset\n"
"  bounding_box: L-drag box   R/M-click=reset\n"
"  ENTER = confirm prompt   •   ESC = skip prompt\n"
          << std::endl;
}

int runOnnxTestVideo(int argc,char** argv)
{
    /* ───── 1) CLI - defaults & parse ───── */
    std::string encPath="image_encoder.onnx", decPath="image_decoder.onnx";
    std::string memAttnPath="memory_attention.onnx", memEncPath="memory_encoder.onnx";
    std::string videoPath;
    int   threads = (int)std::thread::hardware_concurrency(); if(threads<=0) threads=4;
    int   maxFrames=0;
    PromptMode promptMode = PromptMode::SEED_POINTS;

    for(int i=1;i<argc;++i)             // argv[0] = program, argv[1] = --onnx_test_video
    {
        std::string a=argv[i];
        if((a=="--encoder")  && i+1<argc) encPath     = argv[++i];
        else if((a=="--decoder") && i+1<argc) decPath = argv[++i];
        else if((a=="--memattn") && i+1<argc) memAttnPath=argv[++i];
        else if((a=="--memenc")  && i+1<argc) memEncPath =argv[++i];
        else if((a=="--video")   && i+1<argc) videoPath  =argv[++i];
        else if((a=="--threads") && i+1<argc) threads    =std::stoi(argv[++i]);
        else if((a=="--max_frames")&&i+1<argc) maxFrames =std::stoi(argv[++i]);
        else if((a=="--prompt")&&i+1<argc){
            std::string s=argv[++i];
            if      (s=="seed_points")  promptMode=PromptMode::SEED_POINTS;
            else if (s=="bounding_box") promptMode=PromptMode::BOUNDING_BOX;
            else { std::cerr<<"[ERROR] --prompt must be seed_points|bounding_box\n"; return 1; }
        }
        else if(a=="--help"||a=="-h"){ printVideoUsage(argv[0]); return 0; }
    }

    /* ───── 2) select video if none given ───── */
    if(videoPath.empty())
    {
        const wchar_t* filt=L"Video\0*.mp4;*.mkv;*.avi;*.mov\0All\0*.*\0";
        std::string chosen=openFileDialog(filt,L"Select a Video");
        if(chosen.empty()){ std::cerr<<"[ERROR] No file chosen\n"; return 1;}
        videoPath=chosen;
    }

    /* ───── 3) print summary ───── */
    std::cout<<"[INFO] encoder    = "<<encPath<<"\n"
             <<"       decoder    = "<<decPath<<"\n"
             <<"       memAttn    = "<<memAttnPath<<"\n"
             <<"       memEnc     = "<<memEncPath<<"\n"
             <<"       video      = "<<videoPath<<"\n"
             <<"       prompt     = "<<(promptMode==PromptMode::SEED_POINTS?"seed_points":"bounding_box")<<"\n"
             <<"       threads    = "<<threads<<"\n\n";

    /* ───── 4) init SAM2 (GPU if possible) ───── */
    bool cuda=false; for(auto&p:Ort::GetAvailableProviders()) if(p=="CUDAExecutionProvider") cuda=true;
    std::string device = cuda? "cuda:0":"cpu";
    SAM2 sam;
    if(!sam.initializeVideo(encPath,decPath,memAttnPath,memEncPath,threads,device))
    { std::cerr<<"[ERROR] SAM2 init failed\n"; return 1;}

    /* ───── 5) open video & grab first frame ───── */
    cv::VideoCapture cap(videoPath);
    if(!cap.isOpened()){ std::cerr<<"[ERROR] Cannot open video\n"; return 1; }

    cv::Mat firstFrameBGR;
    if(!cap.read(firstFrameBGR)){ std::cerr<<"[ERROR] Cannot read first frame\n"; return 1;}

    VideoAppState st;
    st.sam = &sam;
    st.firstFrame = firstFrameBGR;
    st.originalSize = SAM2Size(firstFrameBGR.cols, firstFrameBGR.rows);
    st.mode = promptMode;

    /* preprocess encoder once */
    sam.preprocessImage(CVHelpers::normalizeRGB(firstFrameBGR));

    /* interactive window */
    st.displayFrame = st.firstFrame.clone();
    cv::namedWindow("First Frame - Interactive", cv::WINDOW_AUTOSIZE);
    cv::imshow ("First Frame - Interactive", st.displayFrame);
    cv::setMouseCallback("First Frame - Interactive", onMouseFirstFrame, &st);

    std::cout<<"\n[INFO] "<<(promptMode==PromptMode::SEED_POINTS?"Seed-point":"Bounding-box")
             <<" mode.  Add prompt on first frame, then press ENTER (ESC = skip).\n";

    while(true)
    {
        int k=cv::waitKey(50)&0xFF;
        if(k==13||k==10) break;   // ENTER
        if(k==27){                // ESC
            st.resetSeeds(); st.resetRect(); break;
        }
    }
    cv::destroyAllWindows();

    /* assemble final prompts for frame-0 */
    SAM2Prompts firstPrompts;
    if(promptMode==PromptMode::SEED_POINTS)
    {
        firstPrompts.points      = st.points;
        firstPrompts.pointLabels = st.labels;
    }
    else if(st.hasRect)
        firstPrompts.rects.push_back(st.rect);

    /* ───── 6) MAIN LOOP (memory pipeline) ───── */
    cap.set(cv::CAP_PROP_POS_FRAMES,0);

    std::string outVideo = videoPath.substr(0, videoPath.find_last_of('.')) + "_mask_overlay.avi";
    int fourcc=cv::VideoWriter::fourcc('M','J','P','G');
    double fps=cap.get(cv::CAP_PROP_FPS);
    cv::VideoWriter writer(outVideo,fourcc,fps,firstFrameBGR.size());
    if(!writer.isOpened()){ std::cerr<<"[ERROR] writer\n"; return 1;}

    int frameIdx=0;
    while(true)
    {
        if(maxFrames>0 && frameIdx>=maxFrames) break;
        cv::Mat frameBGR; cap>>frameBGR; if(frameBGR.empty()) break;

        SAM2Prompts p = (frameIdx==0)? firstPrompts : SAM2Prompts{};
        Image<float> mask = sam.inferMultiFrame(
            CVHelpers::normalizeRGB(frameBGR), p );

        cv::Mat overlay = overlayMask(
            frameBGR, CVHelpers::imageToCvMatWithType(mask, CV_8UC1, 255.0) );

        writer<<overlay;
        std::cout<<"[INFO] Frame "<<frameIdx++<<" done\r"<<std::flush;
    }
    std::cout<<"\n[INFO] Saved "<<outVideo<<"\n";
    return 0;
}
