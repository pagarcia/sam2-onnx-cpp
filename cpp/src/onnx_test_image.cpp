// onnx_test_image.cpp
#include "openFileDialog.h"
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
    cv::Mat redLayer(img.size(), img.type(), cv::Scalar(0, 0, 255));
    redLayer.copyTo(overlay, mask);
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

    // 1) Let user select an image
    string imagePath = openFileDialog();
    if(imagePath.empty()){
        // If user canceled, fallback to a default
        imagePath = "sample.jpg";
    }
    state.originalImage = cv::imread(imagePath);
    if(state.originalImage.empty()){
        cerr << "[ERROR] Could not load image from " << imagePath << endl;
        return -1;
    }
    state.imageSize = state.originalImage.size();

    // 2) Use the default ONNX model paths
    string encoderPath = "image_encoder.onnx";
    string decoderPath = "image_decoder.onnx";
    int threads = (int)std::thread::hardware_concurrency();
    if(!state.sam.initialize(encoderPath, decoderPath, threads, "cpu")){
        cerr << "[ERROR] Could not init SAM2 with " << encoderPath << endl;
        return -1;
    }

    // 3) Resize to match the encoder input (e.g. 1024x1024)
    state.inputSize = state.sam.getInputSize();
    if(state.inputSize.width <= 0 || state.inputSize.height <= 0){
        cerr << "[ERROR] Invalid model input size.\n";
        return -1;
    }
    cv::Mat resized;
    cv::resize(state.originalImage, resized, state.inputSize);
    if(!state.sam.preprocessImage(resized)){
        cerr << "[ERROR] preprocessImage failed.\n";
        return -1;
    }

    // 4) Setup the display
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
