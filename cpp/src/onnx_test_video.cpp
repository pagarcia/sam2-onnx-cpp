// onnx_test_video.cpp

#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h> // for GetAvailableProviders()
#include "SAM2.h"

static void printVideoUsage(const char* argv0)
{
    std::cout << "Usage:\n"
              << "  " << argv0
              << " --onnx_test_video"
              << " --encoder <image_encoder.onnx>"
              << " --decoder <image_decoder.onnx>"
              << " --memattn <memory_attention.onnx>"
              << " --memenc <memory_encoder.onnx>"
              << " --video <sample.mkv>\n\n"
              << "Optional arguments:\n"
              << "  --threads <N>           Number of CPU threads\n"
              << "  --max_frames <N>        0 => process all frames.\n\n"
              << "Note: This version attempts GPU if CUDA is available,\n"
              << "      else falls back to CPU, ignoring any --device.\n"
              << std::endl;
}

/**
 * Modified function that attempts GPU first if CUDA is present
 * (like onnx_test_image.cpp), then falls back to CPU.
 */
int runOnnxTestVideo(int argc, char** argv)
{
    // 1) Default arguments
    std::string encoderPath      = "image_encoder.onnx";
    std::string decoderPath      = "image_decoder.onnx";
    std::string memAttentionPath = "memory_attention.onnx";
    std::string memEncoderPath   = "memory_encoder.onnx";
    std::string videoPath        = "sample_960x540.mkv";
    int maxFrames                = 0;  // 0 => no limit

    // Use hardware concurrency or fallback
    int threadsNumber = static_cast<int>(std::thread::hardware_concurrency());
    if (threadsNumber <= 0) {
        threadsNumber = 4;
    }

    // 2) Parse CLI
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if ((arg == "--encoder") && (i + 1 < argc)) {
            encoderPath = argv[++i];
        } else if ((arg == "--decoder") && (i + 1 < argc)) {
            decoderPath = argv[++i];
        } else if ((arg == "--memattn") && (i + 1 < argc)) {
            memAttentionPath = argv[++i];
        } else if ((arg == "--memenc") && (i + 1 < argc)) {
            memEncoderPath = argv[++i];
        } else if ((arg == "--video") && (i + 1 < argc)) {
            videoPath = argv[++i];
        } else if ((arg == "--threads") && (i + 1 < argc)) {
            threadsNumber = std::stoi(argv[++i]);
        } else if ((arg == "--max_frames") && (i + 1 < argc)) {
            maxFrames = std::stoi(argv[++i]);
        } else if ((arg == "--help") || (arg == "-h")) {
            printVideoUsage(argv[0]);
            return 0;
        }
    }

    // 3) Print providers
    {
        auto allProviders = Ort::GetAvailableProviders();
        std::cout << "[INFO] ONNX Runtime was built with these providers:\n";
        for (auto &prov : allProviders) {
            std::cout << "       " << prov << "\n";
        }
        std::cout << std::endl;
    }

    // 4) Print the final settings (minus device, since we autodetect)
    std::cout << "[INFO] Using:\n"
              << "  encoder        = " << encoderPath << "\n"
              << "  decoder        = " << decoderPath << "\n"
              << "  memAttention   = " << memAttentionPath << "\n"
              << "  memEncoder     = " << memEncoderPath << "\n"
              << "  video          = " << videoPath << "\n"
              << "  max_frames     = " << maxFrames << "\n"
              << "  threads        = " << threadsNumber << "\n\n";

    // 5) Decide if CUDA is available
    bool cudaAvailable = false;
    {
        auto allProviders = Ort::GetAvailableProviders();
        for (auto &p: allProviders) {
            if (p == "CUDAExecutionProvider") {
                cudaAvailable = true;
                break;
            }
        }
    }

    // 6) Create SAM2, attempt GPU => fallback to CPU
    SAM2 sam;
    bool initOk = false;
    bool usedGPU = false;

    if (cudaAvailable) {
        std::cout << "[INFO] Attempting GPU session (cuda:0)...\n";
        try {
            if (sam.initializeVideo(
                    encoderPath, decoderPath,
                    memAttentionPath, memEncoderPath,
                    threadsNumber, "cuda:0"))
            {
                std::cout << "[INFO] GPU session successfully created!\n";
                initOk = true;
                usedGPU = true;
            } else {
                std::cout << "[WARN] GPU session returned false => fallback to CPU.\n";
            }
        }
        catch (const std::exception &e) {
            std::cerr << "[WARN] GPU init exception => " << e.what() << "\n";
        }
    } else {
        std::cout << "[INFO] No CUDAExecutionProvider found; will use CPU.\n";
    }

    if (!initOk) {
        std::cout << "[INFO] Initializing CPU session...\n";
        if(!sam.initializeVideo(
                encoderPath, decoderPath,
                memAttentionPath, memEncoderPath,
                threadsNumber, "cpu"))
        {
            std::cerr << "[ERROR] Could not init SAM2 with CPU.\n";
            printVideoUsage(argv[0]);
            return 1;
        }
        initOk = true;
        usedGPU = false;
    }

    // 7) Summarize device
    if (usedGPU) {
        std::cout << "[INFO] *** GPU inference is in use ***" << std::endl;
    } else {
        std::cout << "[INFO] *** CPU inference is in use ***" << std::endl;
    }

    // 8) Open the video
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "[ERROR] Could not open video: " << videoPath << "\n";
        return 1;
    }
    double fps = cap.get(cv::CAP_PROP_FPS);
    int width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    std::cout << "[INFO] " << videoPath << " => "
              << width << "x" << height
              << ", fps=" << fps
              << ", frames=" << totalFrames << "\n\n";

    // 9) Output video => <basename>_mask_overlay.avi
    std::string baseName = videoPath;
    {
        // strip directory
        size_t pos = baseName.find_last_of("/\\");
        if (pos != std::string::npos) {
            baseName = baseName.substr(pos + 1);
        }
        // remove extension
        pos = baseName.find_last_of('.');
        if (pos != std::string::npos) {
            baseName = baseName.substr(0, pos);
        }
    }
    std::string outVideoPath = baseName + "_mask_overlay.avi";
    int fourcc = cv::VideoWriter::fourcc('M','J','P','G');
    cv::VideoWriter writer(outVideoPath, fourcc, fps, cv::Size(width,height));
    if (!writer.isOpened()) {
        std::cerr << "[ERROR] Could not open VideoWriter => " << outVideoPath << "\n";
        cap.release();
        return 1;
    }
    std::cout << "[INFO] Writing overlay => " << outVideoPath << "\n";

    // 10) Hard-coded single user prompt on the first frame
    Prompts userPrompt;
    userPrompt.points.push_back(cv::Point(510, 375));
    userPrompt.pointLabels.push_back(1); // label=1 => foreground

    // 11) Loop over frames
    int frameIndex = 0;
    auto globalStart = std::chrono::steady_clock::now();

    while (true) {
        if (maxFrames > 0 && frameIndex >= maxFrames) {
            std::cout << "[INFO] Reached max_frames=" << maxFrames << ". Stopping.\n";
            break;
        }
        cv::Mat frameBGR;
        cap >> frameBGR;
        if (frameBGR.empty()) {
            std::cout << "[INFO] End of video.\n";
            break;
        }

        auto frameStart = std::chrono::steady_clock::now();

        cv::Size origSize = frameBGR.size();
        cv::Size inputSize = sam.getInputSize(); 
        cv::Mat resized;
        cv::resize(frameBGR, resized, inputSize);

        // user prompt only on frame 0
        Prompts promptsToUse;
        if (frameIndex == 0) {
            promptsToUse = userPrompt;
            std::cout << "[INFO] Frame 0 => user prompt.\n";
        } else {
            std::cout << "[INFO] Frame " << frameIndex << " => no prompt.\n";
        }

        // multi-frame inference
        cv::Mat maskGray = sam.InferMultiFrame(resized, origSize, promptsToUse);

        auto frameEnd = std::chrono::steady_clock::now();
        double frameMs = std::chrono::duration<double, std::milli>(frameEnd - frameStart).count();
        std::cout << "[INFO] Frame " << frameIndex << " total time: " << frameMs << " ms\n";

        frameIndex++;
        if (maskGray.empty()) {
            std::cerr << "[ERROR] Frame " << frameIndex << " => empty mask.\n";
            break;
        }

        // Build a color overlay (red for mask)
        cv::Mat colorMask(origSize, CV_8UC3, cv::Scalar(0,0,0));
        for (int r=0; r<colorMask.rows; r++) {
            const uchar* mrow = maskGray.ptr<uchar>(r);
            cv::Vec3b* crow   = colorMask.ptr<cv::Vec3b>(r);
            for (int c=0; c<colorMask.cols; c++) {
                if (mrow[c] == 255) {
                    crow[c] = cv::Vec3b(0,0,255);
                }
            }
        }
        // Optionally draw user point on frame0
        if (frameIndex == 1) {
            if (510 < width && 375 < height) {
                cv::circle(colorMask, cv::Point(510,375),
                           5, cv::Scalar(0,255,255), -1);
            }
        }
        cv::Mat overlay;
        float alpha = 0.5f;
        cv::addWeighted(frameBGR, 1.0, colorMask, alpha, 0.0, overlay);

        // Write
        writer << overlay;
    }

    cap.release();
    writer.release();
    std::cout << "[INFO] Done! Wrote " << frameIndex
              << " frames with overlay => " << outVideoPath << "\n";

    auto globalEnd = std::chrono::steady_clock::now();
    double totalMs = std::chrono::duration<double, std::milli>(globalEnd - globalStart).count();
    std::cout << "[INFO] Processed " << frameIndex << " frames in "
              << totalMs << " ms => average "
              << (totalMs / frameIndex) << " ms/frame => "
              << (1000.0 / (totalMs / frameIndex)) << " FPS.\n";

    return 0;
}
