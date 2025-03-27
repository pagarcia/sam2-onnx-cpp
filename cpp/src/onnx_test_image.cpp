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
    cv::Size imageSize;   // original image size
    cv::Size inputSize;   // e.g. 1024x1024

    // Instead of storing scaled points, store them in original coords
    vector<cv::Point> clickedPoints;
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
        prompts.points = state->clickedPoints;  // in original coords
        prompts.pointLabels = state->pointLabels;

        // setPrompts => scales + stores
        state->sam.setPrompts(prompts, state->imageSize);

        auto t0 = high_resolution_clock::now();
        cv::Mat mask = state->sam.InferSingleFrame(state->imageSize);
        auto t1 = high_resolution_clock::now();
        auto ms = duration_cast<milliseconds>(t1 - t0).count();
        cout << "[INFO] Segmentation took " << ms << " ms." << endl;

        cv::Mat overlayed = overlayMask(state->originalImage, mask);

        // Draw seeds
        for (size_t i = 0; i < state->clickedPoints.size(); i++) {
            cv::Scalar color = (state->pointLabels[i] == 1)
                               ? cv::Scalar(0, 0, 255)   // red
                               : cv::Scalar(255, 0, 0); // blue
            cv::circle(overlayed, state->clickedPoints[i], 5, color, -1);
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

    state->clickedPoints.push_back(cv::Point(x, y));
    state->pointLabels.push_back(positive ? 1 : 0);

    updateDisplay(state);
}

int runOnnxTestImage()
{
    AppState state;

    // 1) Print all providers that ONNX Runtime was built with
    {
        auto allProviders = Ort::GetAvailableProviders();
        cout << "[INFO] ONNX Runtime was built with support for these providers:\n";
        for (auto &prov : allProviders) {
            cout << "      " << prov << "\n";
        }
    }

    // 2) Let user select an image
    string imagePath = openFileDialog();
    if (imagePath.empty()) {
        // If user canceled, fallback to a default
        imagePath = "sample.jpg";
    }
    state.originalImage = cv::imread(imagePath);
    if (state.originalImage.empty()) {
        cerr << "[ERROR] Could not load image from " << imagePath << endl;
        return -1;
    }
    state.imageSize = state.originalImage.size();

    // 3) Decide if GPU is even worth trying by checking for "CUDAExecutionProvider"
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

    // 4) Use the default ONNX model paths
    string encoderPath = "image_encoder.onnx";
    string decoderPath = "image_decoder.onnx";
    int threads = (int)std::thread::hardware_concurrency();

    bool initOk = false;
    bool usedGPU = false;

    // If we see that "CUDAExecutionProvider" is available, try "cuda:0"
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
        cout << "[INFO] CUDAExecutionProvider not found in this ORT build. Will use CPU.\n";
    }

    // If GPU init wasn't done, fallback to CPU
    if (!initOk) {
        cout << "[INFO] Initializing CPU session...\n";
        if(!state.sam.initialize(encoderPath, decoderPath, threads, "cpu")) {
            cerr << "[ERROR] Could not init SAM2 with CPU.\n";
            return -1;
        }
        initOk = true;
        usedGPU = false;
    }

    // Just print a final message:
    if (usedGPU) {
        cout << "[INFO] *** GPU inference is in use ***" << endl;
    } else {
        cout << "[INFO] *** CPU inference is in use ***" << endl;
    }

    // 5) Resize to match the encoder input (e.g. 1024x1024)
    state.inputSize = state.sam.getInputSize();
    if (state.inputSize.width <= 0 || state.inputSize.height <= 0) {
        cerr << "[ERROR] Invalid model input size.\n";
        return -1;
    }
    cv::Mat resized;
    cv::resize(state.originalImage, resized, state.inputSize);

    // 6) Preprocess => runs the "ImageEncoder" in your pipeline
    auto preStart = high_resolution_clock::now();
    if (!state.sam.preprocessImage(resized)) {
        cerr << "[ERROR] preprocessImage failed.\n";
        return -1;
    }
    auto preEnd = high_resolution_clock::now();
    auto preMs = duration_cast<milliseconds>(preEnd - preStart).count();
    cout << "[INFO] preprocessImage (encoding) took " << preMs << " ms." << endl;

    // 7) Setup the display and run interactive segmentation
    state.originalImage.copyTo(state.displayImage);
    cv::namedWindow("Interactive Segmentation");
    cv::imshow("Interactive Segmentation", state.displayImage);
    cv::setMouseCallback("Interactive Segmentation", onMouse, &state);

    cout << "[INFO] L-click=positive, R-click=negative, M-click=reset. ESC=exit.\n";
    while(true){
        int key = cv::waitKey(50);
        if(key == 27) { // ESC
            break;
        }
    }
    return 0;
}
