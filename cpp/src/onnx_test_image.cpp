// onnx_test_image.cpp
#include "openFileDialog.h"
#include <onnxruntime_cxx_api.h>  // for GetAvailableProviders()
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <thread>
#include <chrono>
#include "SAM2.h"

using namespace std;
using namespace std::chrono;

struct AppState {
    SAM2 sam;
    cv::Mat originalImage;
    cv::Mat displayImage;
    Size originalImageSize;   // original image size
    Size SAM2ImageSize;   // e.g. 1024x1024

    // Instead of storing scaled points, store them in original coords
    vector<Point> clickedPoints;
    vector<int> pointLabels; // 1=positive, 0=negative
};

static cv::Mat overlayMask(const cv::Mat& img, const cv::Mat& mask) {
    cv::Mat overlay;
    img.copyTo(overlay);
    cv::Mat greenLayer(img.size(), img.type(), cv::Scalar(0, 255, 0));
    greenLayer.copyTo(overlay, mask);
    cv::Mat blended;
    cv::addWeighted(img, 0.7, overlay, 0.3, 0, blended);
    return blended;
}

static void updateDisplay(AppState* state) {
    if (state->clickedPoints.empty()) {
        state->originalImage.copyTo(state->displayImage);
    } else {
        // Build a Prompts struct from the user’s clickedPoints
        Prompts prompts;
        prompts.points      = state->clickedPoints;  // in original coords
        prompts.pointLabels = state->pointLabels;

        // setPrompts => scales + stores
        state->sam.setPrompts(prompts, state->originalImageSize);

        auto t0 = high_resolution_clock::now();
        cv::Mat mask = state->sam.InferSingleFrame(state->originalImageSize);
        auto t1 = high_resolution_clock::now();
        auto ms = duration_cast<milliseconds>(t1 - t0).count();
        cout << "[INFO] Segmentation took " << ms << " ms." << endl;

        cv::Mat overlayed = overlayMask(state->originalImage, mask);

        // Draw seeds
        for (size_t i = 0; i < state->clickedPoints.size(); i++) {
            cv::Scalar color = (state->pointLabels[i] == 1)
                               ? cv::Scalar(0, 0, 255)   // red
                               : cv::Scalar(255, 0, 0); // blue
            cv::circle(overlayed, cv::Point(state->clickedPoints[i].x, state->clickedPoints[i].y), 5, color, -1);
        }
        overlayed.copyTo(state->displayImage);
    }
    cv::imshow("Interactive Segmentation", state->displayImage);
}

static void onMouse(int event, int x, int y, int, void* userdata) {
    AppState* state = (AppState*)userdata;
    
    if (event == cv::EVENT_MBUTTONDOWN) {
        // Reset
        cout << "[INFO] Reset. Clearing points.\n";
        state->clickedPoints.clear();
        state->pointLabels.clear();
        state->originalImage.copyTo(state->displayImage);
        cv::imshow("Interactive Segmentation", state->displayImage);
        return;
    }

    if (event != cv::EVENT_LBUTTONDOWN && event != cv::EVENT_RBUTTONDOWN)
        return;

    bool positive = (event == cv::EVENT_LBUTTONDOWN);
    cout << "[INFO] " << (positive ? "Positive" : "Negative")
         << " point at (" << x << ", " << y << ").\n";

    state->clickedPoints.push_back(Point(x, y));
    state->pointLabels.push_back(positive ? 1 : 0);

    updateDisplay(state);
}

// ----------------------------------------------------------------------------
//  RUN ONNX TEST IMAGE   (updated to parse CLI for --image, --encoder, --decoder, --threads )
// ----------------------------------------------------------------------------
int runOnnxTestImage(int argc, char** argv)
{
    // ---------------------------
    // 1) Parse CLI arguments
    // ---------------------------
    std::string encoderPath = "image_encoder.onnx";
    std::string decoderPath = "image_decoder.onnx";
    std::string imagePath;  // empty => file dialog
    int threads = (int)std::thread::hardware_concurrency();
    if (threads <= 0) threads = 4;

    for(int i=2; i<argc; i++){ 
        // i=2 if your main has "Segment --onnx_test_image" as first 2
        std::string arg = argv[i];
        if(arg == "--encoder" && i+1<argc) {
            encoderPath = argv[++i];
        } else if(arg == "--decoder" && i+1<argc) {
            decoderPath = argv[++i];
        } else if(arg == "--image" && i+1<argc) {
            imagePath = argv[++i];
        } else if(arg == "--threads" && i+1<argc) {
            threads = std::stoi(argv[++i]);
        } else if(arg == "--help" || arg == "-h") {
            cout << "\nUsage:\n"
                 << "  Segment --onnx_test_image [--encoder E.onnx] [--decoder D.onnx] [--image myimage.jpg] [--threads N]\n\n"
                 << "Notes:\n"
                 << "  * If --image is not specified, a file dialog opens.\n"
                 << "  * GPU if CUDA is available, otherwise CPU.\n"
                 << "  * L-click=foreground, R-click=background, M-click=reset. ESC=exit.\n"
                 << endl;
            return 0;
        }
    }

    // If no --image => open file dialog
    if(imagePath.empty()) {
        cout << "[INFO] No image => opening file dialog...\n";
        const wchar_t* filter = L"Image Files\0*.jpg;*.jpeg;*.png;*.bmp\0All Files\0*.*\0";
        const wchar_t* title  = L"Select an Image File";
        std::string chosen = openFileDialog(filter, title);
        if(chosen.empty()) {
            cerr << "[ERROR] No file selected => abort.\n";
            return 1;
        } else {
            imagePath = chosen;
        }
    }

    // 2) Print all providers
    {
        auto allProviders = Ort::GetAvailableProviders();
        cout << "[INFO] ONNX Runtime was built with support for these providers:\n";
        for (auto &prov : allProviders) {
            cout << "      " << prov << "\n";
        }
        cout << endl;
    }

    cout << "[INFO] Using:\n"
         << "   encoder  = " << encoderPath << "\n"
         << "   decoder  = " << decoderPath << "\n"
         << "   image    = " << imagePath << "\n"
         << "   threads  = " << threads << "\n\n";

    // 3) Load the image
    AppState state;
    state.originalImage = cv::imread(imagePath);
    if (state.originalImage.empty()) {
        cerr << "[ERROR] Could not load image from " << imagePath << endl;
        return -1;
    }
    state.originalImageSize.width = state.originalImage.size().width;
    state.originalImageSize.height = state.originalImage.size().height;

    // 4) Check for GPU availability
    bool cudaAvailable = false;
    {
        auto allProviders = Ort::GetAvailableProviders();
        for (auto &p : allProviders) {
            if (p == "CUDAExecutionProvider") {
                cudaAvailable = true;
                break;
            }
        }
    }

    bool initOk = false;
    bool usedGPU = false;

    // 5) Try GPU => fallback CPU
    if (cudaAvailable) {
        cout << "[INFO] Attempting to initialize GPU session (cuda:0)...\n";
        try {
            if (state.sam.initialize(encoderPath, decoderPath, threads, "cuda:0")) {
                cout << "[INFO] GPU session created successfully!\n";
                initOk = true;
                usedGPU = true;
            } else {
                cout << "[WARN] GPU session returned false => fallback to CPU.\n";
            }
        }
        catch (const std::exception &e) {
            cerr << "[WARN] GPU init exception: " << e.what() << "\n";
        }
    } else {
        cout << "[INFO] CUDAExecutionProvider not found => using CPU.\n";
    }

    if(!initOk) {
        cout << "[INFO] Initializing CPU session...\n";
        if(!state.sam.initialize(encoderPath, decoderPath, threads, "cpu")) {
            cerr << "[ERROR] Could not init SAM2 with CPU.\n";
            return -1;
        }
        initOk = true;
        usedGPU = false;
    }

    if(usedGPU) cout << "[INFO] *** GPU inference is in use ***\n\n";
    else        cout << "[INFO] *** CPU inference is in use ***\n\n";

    // 6) Resize to match the encoder input (e.g. 1024x1024)
    state.SAM2ImageSize = state.sam.getInputSize();
    if (state.SAM2ImageSize.width <= 0 || state.SAM2ImageSize.height <= 0) {
        cerr << "[ERROR] Invalid model input size.\n";
        return -1;
    }

    // 7) Preprocess => runs the "ImageEncoder"
    auto preStart = high_resolution_clock::now();
    if (!state.sam.preprocessImage(state.originalImage)) {
        cerr << "[ERROR] preprocessImage failed.\n";
        return -1;
    }
    auto preEnd = high_resolution_clock::now();
    auto preMs = duration_cast<milliseconds>(preEnd - preStart).count();
    cout << "[INFO] preprocessImage (encoding) took " << preMs << " ms.\n\n";

    // 8) Interactive segmentation
    state.originalImage.copyTo(state.displayImage);
    cv::namedWindow("Interactive Segmentation");
    cv::imshow("Interactive Segmentation", state.displayImage);
    cv::setMouseCallback("Interactive Segmentation", onMouse, &state);

    cout << "[INFO] L-click=positive, R-click=negative, M-click=reset. ESC=exit.\n";

    while(true) {
        int key = cv::waitKey(50);
        if (key == 27) { // ESC
            break;
        }
    }
    return 0;
}
