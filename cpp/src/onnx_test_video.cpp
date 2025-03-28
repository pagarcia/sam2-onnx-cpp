// onnx_test_video.cpp
#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h> // for GetAvailableProviders()
#include "SAM2.h"
#include "openFileDialog.h"

// A small helper to print usage
static void printVideoUsage(const char* argv0)
{
    std::cout << "\nUsage:\n"
              << "  " << argv0
              << " --onnx_test_video"
              << " [--encoder <image_encoder.onnx>]"
              << " [--decoder <image_decoder.onnx>]"
              << " [--memattn <memory_attention.onnx>]"
              << " [--memenc  <memory_encoder.onnx>]"
              << " [--video <myvideo.mkv>]"
              << " [--threads <N>]"
              << " [--max_frames <N>]\n\n"
              << "Notes:\n"
              << "  * If --video is not specified, a file dialog opens.\n"
              << "  * GPU if CUDA is available, otherwise CPU.\n"
              << "  * On first frame, user can place seeds (L=FG, R=BG, M=reset). Press ENTER to finalize.\n"
              << "  * Then the entire video is segmented with those seeds as memory.\n"
              << std::endl;
}

// We'll store interactive seeds for the first frame
struct VideoAppState {
    SAM2* sam = nullptr;          // The SAM2 object
    cv::Mat firstFrame;           // Original BGR from the first frame
    cv::Mat displayFrame;         // Display with partial overlay
    cv::Size originalSize;        // e.g. 960×540
    cv::Size inputSize;           // e.g. 1024×1024 (model input)

    // The user-chosen seeds in original coords
    std::vector<cv::Point> points;
    std::vector<int>       labels; // 1=FG, 0=BG
};

// Overlays a binary mask in green
static cv::Mat overlayMask(const cv::Mat &bgr, const cv::Mat &maskGray)
{
    cv::Mat overlay;
    bgr.copyTo(overlay);
    cv::Mat green(bgr.size(), bgr.type(), cv::Scalar(0,255,0));
    green.copyTo(overlay, maskGray);

    cv::Mat blended;
    cv::addWeighted(bgr, 0.7, overlay, 0.3, 0, blended);
    return blended;
}

// Each time the user adds seeds on the first frame, we re-run single-frame inference
//   => partial mask overlay
static void updateFirstFrameDisplay(VideoAppState* st)
{
    // If no seeds, show plain
    if (st->points.empty()) {
        st->firstFrame.copyTo(st->displayFrame);
        cv::imshow("First Frame - Interactive", st->displayFrame);
        return;
    }

    // Build a Prompts from the user’s seeds
    Prompts prompts;
    prompts.points      = st->points;
    prompts.pointLabels = st->labels;

    // Let SAM know about them
    st->sam->setPrompts(prompts, st->originalSize);

    // Call single-frame => upsample => overlay
    auto t0 = std::chrono::high_resolution_clock::now();
    cv::Mat mask = st->sam->InferSingleFrame(st->originalSize);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double,std::milli>(t1 - t0).count();
    std::cout << "[INFO] Partial decode => " << ms << " ms\n";

    cv::Mat overlayed = overlayMask(st->firstFrame, mask);

    // Draw seeds => FG=red, BG=blue
    for (size_t i=0; i<st->points.size(); i++) {
        cv::Scalar color = (st->labels[i]==1)
                           ? cv::Scalar(0,0,255)
                           : cv::Scalar(255,0,0);
        cv::circle(overlayed, st->points[i], 5, color, -1);
    }

    overlayed.copyTo(st->displayFrame);
    cv::imshow("First Frame - Interactive", st->displayFrame);
}

// Mouse callback => L=FG, R=BG, M=reset
static void onMouseFirstFrame(int event, int x, int y, int, void* userdata)
{
    auto st = reinterpret_cast<VideoAppState*>(userdata);

    if (event == cv::EVENT_MBUTTONDOWN) {
        std::cout << "[INFO] Reset seeds.\n";
        st->points.clear();
        st->labels.clear();
        updateFirstFrameDisplay(st);
        return;
    }

    if (event==cv::EVENT_LBUTTONDOWN || event==cv::EVENT_RBUTTONDOWN) {
        bool fg = (event==cv::EVENT_LBUTTONDOWN);
        std::cout << "[INFO] " << (fg?"Foreground":"Background")
                  << " => ("<< x <<","<< y <<")\n";

        // clamp if out of range
        if (x<0) x=0;
        if (y<0) y=0;
        if (x>=st->originalSize.width)  x=st->originalSize.width -1;
        if (y>=st->originalSize.height) y=st->originalSize.height-1;

        st->points.push_back(cv::Point(x,y));
        st->labels.push_back(fg?1:0);

        // Show partial
        updateFirstFrameDisplay(st);
    }
}

/** The main function for runOnnxTestVideo(...) */
int runOnnxTestVideo(int argc, char** argv)
{
    // ------------------------------------------------------------------
    // 1) Parse CLI
    // ------------------------------------------------------------------
    std::string encoderPath     = "image_encoder.onnx";
    std::string decoderPath     = "image_decoder.onnx";
    std::string memAttentionPath= "memory_attention.onnx";
    std::string memEncoderPath  = "memory_encoder.onnx";
    std::string videoPath; // empty => file dialog
    int maxFrames = 0;

    int threads = (int)std::thread::hardware_concurrency();
    if (threads<=0) threads=4;

    for(int i=1;i<argc;i++){
        std::string arg=argv[i];
        if((arg=="--encoder") && i+1<argc) encoderPath=argv[++i];
        else if((arg=="--decoder") && i+1<argc) decoderPath=argv[++i];
        else if((arg=="--memattn") && i+1<argc) memAttentionPath=argv[++i];
        else if((arg=="--memenc")  && i+1<argc) memEncoderPath=argv[++i];
        else if((arg=="--video")   && i+1<argc) videoPath=argv[++i];
        else if((arg=="--threads") && i+1<argc) threads= std::stoi(argv[++i]);
        else if((arg=="--max_frames") && i+1<argc) maxFrames= std::stoi(argv[++i]);
        else if((arg=="--help")||(arg=="-h")){
            printVideoUsage(argv[0]);
            return 0;
        }
    }

    // If no --video => open file dialog
    if (videoPath.empty()) {
        std::cout << "[INFO] No video => opening file dialog...\n";
        const wchar_t* filter = L"Video Files\0*.mp4;*.mkv;*.avi;*.mov\0All Files\0*.*\0";
        const wchar_t* title  = L"Select a Video File";
        std::string chosen = openFileDialog(filter, title);
        if (chosen.empty()) {
            std::cerr << "[ERROR] No file selected => aborting.\n";
            return 1; // or an appropriate exit code
        } else {
            videoPath = chosen;
        }
    }    

    // ------------------------------------------------------------------
    // 2) Print all providers
    // ------------------------------------------------------------------
    {
        auto allProviders = Ort::GetAvailableProviders();
        std::cout<<"[INFO] ONNX Runtime providers:\n";
        for (auto &p:allProviders){
            std::cout<<"       "<<p<<"\n";
        }
        std::cout<<std::endl;
    }

    // Summarize
    std::cout<<"[INFO] Using:\n"
             <<"   encoder       = "<<encoderPath<<"\n"
             <<"   decoder       = "<<decoderPath<<"\n"
             <<"   memAttention  = "<<memAttentionPath<<"\n"
             <<"   memEncoder    = "<<memEncoderPath<<"\n"
             <<"   video         = "<<videoPath<<"\n"
             <<"   max_frames    = "<<maxFrames<<"\n"
             <<"   threads       = "<<threads<<"\n\n";

    // ------------------------------------------------------------------
    // 3) Attempt GPU => fallback CPU
    // ------------------------------------------------------------------
    bool cudaAvailable=false;
    {
        auto allProviders= Ort::GetAvailableProviders();
        for (auto &p: allProviders){
            if(p=="CUDAExecutionProvider"){
                cudaAvailable=true; break;
            }
        }
    }

    SAM2 sam;
    bool initOk=false, usedGPU=false;
    if(cudaAvailable){
        std::cout<<"[INFO] Attempting GPU => cuda:0\n";
        try {
            if(sam.initializeVideo(encoderPath, decoderPath,
                                   memAttentionPath, memEncoderPath,
                                   threads, "cuda:0"))
            {
                std::cout<<"[INFO] GPU session ok.\n";
                initOk=true; usedGPU=true;
            } else {
                std::cout<<"[WARN] GPU returned false => CPU.\n";
            }
        }
        catch(const std::exception &e){
            std::cout<<"[WARN] GPU init ex => "<< e.what()<<"\n";
        }
    }
    if(!initOk){
        std::cout<<"[INFO] Using CPU...\n";
        if(!sam.initializeVideo(encoderPath, decoderPath,
                                memAttentionPath, memEncoderPath,
                                threads, "cpu"))
        {
            std::cerr<<"[ERROR] CPU init failed.\n";
            return 1;
        }
        usedGPU=false; initOk=true;
    }
    if(usedGPU) std::cout<<"[INFO] *** GPU inference ***\n";
    else        std::cout<<"[INFO] *** CPU inference ***\n";

    // ------------------------------------------------------------------
    // 4) Open the video => read the first frame => interactive seeds
    // ------------------------------------------------------------------
    cv::VideoCapture cap(videoPath);
    if(!cap.isOpened()){
        std::cerr<<"[ERROR] Could not open video => "<<videoPath<<"\n";
        return 1;
    }
    double fps = cap.get(cv::CAP_PROP_FPS);
    int width  = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int totalF = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
    std::cout<<"[INFO] "<<videoPath<<" => "
             <<width<<"x"<<height
             <<", fps="<<fps
             <<", frames="<<totalF<<"\n\n";

    cv::Mat firstFrameBGR;
    if(!cap.read(firstFrameBGR)){
        std::cerr<<"[ERROR] Could not read first frame.\n";
        return 1;
    }

    // We'll do single-frame approach => user seeds => partial decode
    //   i.e. call sam.preprocessImage(...) + sam.InferSingleFrame(...)
    VideoAppState st;
    st.sam=&sam;
    firstFrameBGR.copyTo(st.firstFrame);
    st.originalSize = st.firstFrame.size();

    // The "inputSize" is e.g. 1024×1024
    st.inputSize = sam.getInputSize();
    if(st.inputSize.width<=0 || st.inputSize.height<=0){
        std::cerr<<"[ERROR] Invalid input size from sam.\n";
        return 1;
    }

    // Preprocess the first frame => single-frame usage
    cv::Mat resizedFirst;
    cv::resize(firstFrameBGR, resizedFirst, st.inputSize);

    auto preT0=std::chrono::steady_clock::now();
    if(!sam.preprocessImage(resizedFirst)){
        std::cerr<<"[ERROR] preprocessImage failed.\n";
        return 1;
    }
    auto preT1=std::chrono::steady_clock::now();
    double preMs= std::chrono::duration<double,std::milli>(preT1-preT0).count();
    std::cout<<"[INFO] first-frame preprocess => "<< preMs <<" ms\n";

    st.firstFrame.copyTo(st.displayFrame);

    // interactive window => user seeds
    cv::namedWindow("First Frame - Interactive", cv::WINDOW_AUTOSIZE);
    cv::imshow("First Frame - Interactive", st.displayFrame);
    cv::setMouseCallback("First Frame - Interactive", onMouseFirstFrame, &st);

    std::cout<<"[INFO] Place seeds on first frame => L=FG, R=BG, M=reset.\n"
             <<"       Press ENTER to finalize, or ESC => no seeds.\n";

    while(true){
        int key = cv::waitKey(50)&0xFF;
        if(key==13||key==10){ // ENTER
            std::cout<<"[INFO] user pressed ENTER => finalize seeds.\n";
            break;
        }
        if(key==27){ // ESC
            std::cout<<"[INFO] user pressed ESC => clearing seeds.\n";
            st.points.clear();
            st.labels.clear();
            break;
        }
    }
    cv::destroyAllWindows();

    // Now we do a multi-frame pass => memory pipeline
    //   frame0 => user seeds => memory
    //   subsequent => empty => memory tracking
    cap.set(cv::CAP_PROP_POS_FRAMES, 0);

    // Output .avi => e.g. "myvideo_mask_overlay.avi"
    // 1) First, extract the directory from videoPath
    std::string inputDir;
    {
        size_t slashPos = videoPath.find_last_of("/\\");
        if (slashPos == std::string::npos) {
            // No slash found => the video is in the current directory
            inputDir = ".";
        } else {
            // Extract everything up to (but not including) the slash
            inputDir = videoPath.substr(0, slashPos);
        }
    }

    // 2) Then, extract the filename stem (no directory, no extension)
    std::string baseName = videoPath;
    {
        size_t pos = baseName.find_last_of("/\\");
        if(pos != std::string::npos) {
            baseName = baseName.substr(pos+1); // remove any leading path
        }
        pos = baseName.find_last_of('.');
        if(pos != std::string::npos) {
            baseName = baseName.substr(0, pos); // remove extension
        }
    }

    // 3) Finally, combine them, ensuring we put the output in inputDir
    std::string outVideo = inputDir + "/" + baseName + "_mask_overlay.avi";

    int fourcc= cv::VideoWriter::fourcc('M','J','P','G');
    cv::VideoWriter writer(outVideo, fourcc, fps, cv::Size(width,height));
    if(!writer.isOpened()){
        std::cerr<<"[ERROR] Could not open writer => "<<outVideo<<"\n";
        return 1;
    }
    std::cout<<"[INFO] Writing => "<< outVideo <<"\n";

    // Convert user seeds => final prompts
    Prompts userPrompts;
    userPrompts.points      = st.points;
    userPrompts.pointLabels = st.labels;

    auto globalStart=std::chrono::steady_clock::now();
    int frameIndex=0;

    while(true){
        if(maxFrames>0 && frameIndex>=maxFrames){
            std::cout<<"[INFO] Reached max_frames="<<maxFrames<<".\n";
            break;
        }
        cv::Mat frameBGR;
        cap >> frameBGR;
        if(frameBGR.empty()){
            std::cout<<"[INFO] End of video.\n";
            break;
        }

        auto t0=std::chrono::steady_clock::now();

        // We must resize EVERY frame to inputSize => no mismatch
        cv::Mat resizedFrame;
        cv::resize(frameBGR, resizedFrame, st.inputSize);

        // On frame0 => seeds, else empty
        Prompts promptsToUse;
        if(frameIndex==0){
            promptsToUse= userPrompts;
            std::cout<<"[INFO] Frame0 => user seeds.\n";
        }

        cv::Mat mask= st.sam->InferMultiFrame(resizedFrame, frameBGR.size(), promptsToUse);
        auto t1= std::chrono::steady_clock::now();
        double ms= std::chrono::duration<double,std::milli>(t1-t0).count();
        std::cout<<"[INFO] Frame "<<frameIndex<<" => "<< ms <<" ms\n";

        if(mask.empty()){
            std::cerr<<"[ERROR] empty mask => break.\n";
            break;
        }

        // Use the same green overlay approach as the interactive frame
        cv::Mat overlayed = overlayMask(frameBGR, mask);

        // If frameIndex==0 => draw seeds => FG=red, BG=blue
        if (frameIndex == 0) {
            for (size_t i = 0; i < st.points.size(); i++) {
                cv::Scalar color = (st.labels[i] == 1)
                                    ? cv::Scalar(0, 0, 255)    // foreground => red
                                    : cv::Scalar(255, 0, 0);   // background => blue
                cv::circle(overlayed, st.points[i], 5, color, -1);
            }
        }

        // Write the final overlay
        writer << overlayed;

        frameIndex++;
    }

    writer.release();
    cap.release();

    auto globalEnd= std::chrono::steady_clock::now();
    double totalMs= std::chrono::duration<double,std::milli>(globalEnd-globalStart).count();
    std::cout<<"[INFO] Wrote "<< frameIndex <<" frames => "<< outVideo <<"\n"
             <<"       total "<< totalMs <<" ms => "
             << (totalMs/frameIndex) <<" ms/frame => "
             << (1000.0/(totalMs/frameIndex)) <<" FPS\n";
    return 0;
}