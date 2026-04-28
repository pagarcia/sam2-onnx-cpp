#include "SAM2.h"
#include "ArtifactResolver.h"
#include "CVHelpers.h"
#include "openFileDialog.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <exception>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kJumpFrames = 10;
const char* kWindowName = "Video Anchors - SAM2";

enum class PromptMode { SEED_POINTS, BOUNDING_BOX };

struct AnchorEditorState
{
    SAM2* sam = nullptr;
    cv::VideoCapture* capture = nullptr;
    PromptMode mode = PromptMode::SEED_POINTS;
    int totalFrames = 0;
    int currentFrameIndex = 0;

    cv::Mat currentFrame;
    cv::Mat displayFrame;
    SAM2Size originalSize;

    std::map<int, SAM2Prompts> anchors;
    std::map<int, CachedEncoderOutputs> anchorEncoderCaches;

    std::vector<SAM2Point> currentPoints;
    std::vector<int> currentLabels;

    bool drawing = false;
    bool hasFinalRect = false;
    SAM2Rect rect;
};

cv::Mat overlayMask(const cv::Mat& image, const cv::Mat& maskGray)
{
    cv::Mat overlay = image.clone();
    cv::Mat green(image.size(), image.type(), cv::Scalar(0, 255, 0));
    green.copyTo(overlay, maskGray);

    cv::Mat blended;
    cv::addWeighted(image, 0.7, overlay, 0.3, 0.0, blended);
    return blended;
}

std::string promptModeName(PromptMode mode)
{
    return mode == PromptMode::SEED_POINTS ? "seed_points" : "bounding_box";
}

std::string lowerCopy(std::string value)
{
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

std::string normalizeDeviceArg(const std::string& raw)
{
    const std::string value = lowerCopy(raw);
    if (value.empty() || value == "auto") {
        return "auto";
    }
    if (value == "cpu") {
        return "cpu";
    }
    if (value == "cuda") {
        return "cuda:0";
    }
    if (value.rfind("cuda:", 0) == 0) {
        return value;
    }
    if (value == "dml" || value == "directml") {
        return "dml:0";
    }
    if (value.rfind("dml:", 0) == 0) {
        return value;
    }
    return "";
}

void appendDeviceCandidate(std::vector<std::string>* devices, const std::string& device)
{
    if (std::find(devices->begin(), devices->end(), device) == devices->end()) {
        devices->push_back(device);
    }
}

std::vector<std::string> buildDeviceCandidates(const std::string& requestedDevice, bool forceCpu)
{
    std::vector<std::string> devices;
    if (requestedDevice == "auto") {
        if (!forceCpu && SAM2::hasCudaDriver()) {
            appendDeviceCandidate(&devices, "cuda:0");
        }
        if (!forceCpu && SAM2::hasDirectMLProvider()) {
            appendDeviceCandidate(&devices, "dml:0");
        }
        appendDeviceCandidate(&devices, "cpu");
        return devices;
    }

    appendDeviceCandidate(&devices, requestedDevice);
    if (requestedDevice != "cpu") {
        appendDeviceCandidate(&devices, "cpu");
    }
    return devices;
}

int clampFrameIndex(int frameIndex, int totalFrames)
{
    if (totalFrames <= 0) {
        return 0;
    }
    return std::max(0, std::min(frameIndex, totalFrames - 1));
}

int resolveFrameCount(cv::VideoCapture& cap, int maxFrames)
{
    int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    if (totalFrames <= 0) {
        if (maxFrames > 0) {
            return maxFrames;
        }
        return 0;
    }

    if (maxFrames > 0) {
        totalFrames = std::min(totalFrames, maxFrames);
    }
    return std::max(totalFrames, 1);
}

bool loadFrameAt(cv::VideoCapture& cap, int frameIndex, cv::Mat* frameOut)
{
    cap.set(cv::CAP_PROP_POS_FRAMES, frameIndex);
    cv::Mat frame;
    if (!cap.read(frame) || frame.empty()) {
        return false;
    }
    *frameOut = frame;
    return true;
}

SAM2Rect normalizedRect(const SAM2Rect& rect)
{
    SAM2Rect normalized = rect;
    if (normalized.width < 0) {
        normalized.x += normalized.width;
        normalized.width = -normalized.width;
    }
    if (normalized.height < 0) {
        normalized.y += normalized.height;
        normalized.height = -normalized.height;
    }
    return normalized;
}

void drawHud(cv::Mat* image, int frameIndex, int totalFrames, size_t anchorCount, PromptMode mode)
{
    std::vector<std::string> lines;
    lines.push_back(
        "Frame " + std::to_string(frameIndex + 1) + "/" + std::to_string(totalFrames)
        + " | Anchors: " + std::to_string(anchorCount));
    lines.push_back("Prompt: " + promptModeName(mode));
    lines.push_back("A/D: +/-1 frame | J/L: +/-10 frames");
    lines.push_back("Enter/Space: run video | Esc/Q: finish | C: clear frame");
    if (mode == PromptMode::SEED_POINTS) {
        lines.push_back("L-click: FG | R-click: BG | M-click: clear current frame");
    } else {
        lines.push_back("L-drag: box | R/M-click: clear current frame");
    }

    int y = 28;
    for (const auto& line : lines) {
        cv::putText(
            *image,
            line,
            cv::Point(12, y),
            cv::FONT_HERSHEY_SIMPLEX,
            0.7,
            cv::Scalar(15, 15, 15),
            4,
            cv::LINE_AA);
        cv::putText(
            *image,
            line,
            cv::Point(12, y),
            cv::FONT_HERSHEY_SIMPLEX,
            0.7,
            cv::Scalar(255, 255, 255),
            1,
            cv::LINE_AA);
        y += 28;
    }
}

void clearCurrentPrompt(AnchorEditorState* state)
{
    state->currentPoints.clear();
    state->currentLabels.clear();
    state->drawing = false;
    state->hasFinalRect = false;
    state->rect = SAM2Rect();
}

SAM2Prompts buildCurrentPrompts(const AnchorEditorState& state)
{
    SAM2Prompts prompts;
    if (state.mode == PromptMode::SEED_POINTS) {
        prompts.points = state.currentPoints;
        prompts.pointLabels = state.currentLabels;
    } else if (state.hasFinalRect) {
        prompts.rects.push_back(normalizedRect(state.rect));
    }
    return prompts;
}

void loadCurrentPromptFromAnchor(AnchorEditorState* state)
{
    clearCurrentPrompt(state);

    const auto it = state->anchors.find(state->currentFrameIndex);
    if (it == state->anchors.end()) {
        return;
    }

    if (state->mode == PromptMode::SEED_POINTS) {
        state->currentPoints = it->second.points;
        state->currentLabels = it->second.pointLabels;
        return;
    }

    if (!it->second.rects.empty()) {
        state->rect = normalizedRect(it->second.rects.front());
        state->hasFinalRect = true;
    }
}

void storeCurrentPromptToAnchor(AnchorEditorState* state)
{
    SAM2Prompts prompts = buildCurrentPrompts(*state);

    const bool hasPrompt =
        (!prompts.points.empty() && prompts.points.size() == prompts.pointLabels.size())
        || !prompts.rects.empty();

    if (!hasPrompt) {
        state->anchors.erase(state->currentFrameIndex);
        state->anchorEncoderCaches.erase(state->currentFrameIndex);
        return;
    }

    state->anchors[state->currentFrameIndex] = std::move(prompts);
    CachedEncoderOutputs cachedOutputs;
    if (state->sam->captureCachedEncoderOutputs(&cachedOutputs)) {
        state->anchorEncoderCaches[state->currentFrameIndex] = std::move(cachedOutputs);
    }
}

bool preprocessCurrentFrame(AnchorEditorState* state)
{
    state->originalSize = SAM2Size(state->currentFrame.cols, state->currentFrame.rows);
    const auto cacheIt = state->anchorEncoderCaches.find(state->currentFrameIndex);
    if (cacheIt != state->anchorEncoderCaches.end()) {
        if (state->sam->restoreCachedEncoderOutputs(cacheIt->second)) {
            return true;
        }
        std::cerr << "[WARN] Failed to restore cached encoder outputs for frame "
                  << state->currentFrameIndex << ", recomputing.\n";
    }

    try {
        return state->sam->preprocessImage(CVHelpers::normalizeRGB(state->currentFrame, 255.0f));
    }
    catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to preprocess frame " << state->currentFrameIndex
                  << " => " << e.what() << '\n';
        return false;
    }
}

void renderAnchorEditor(AnchorEditorState* state)
{
    state->displayFrame = state->currentFrame.clone();

    bool shouldRun = false;
    if (state->mode == PromptMode::SEED_POINTS) {
        shouldRun = !state->currentPoints.empty();
    } else {
        shouldRun = state->hasFinalRect;
    }

    if (shouldRun) {
        try {
            const SAM2Prompts prompts = buildCurrentPrompts(*state);
            state->sam->setPrompts(prompts, state->originalSize);

            const auto start = std::chrono::high_resolution_clock::now();
            Image<float> mask = state->sam->inferSingleFrame(state->originalSize);
            const double ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - start).count();
            std::cout << "[INFO] Partial decode => " << ms << " ms\n";

            state->displayFrame = overlayMask(
                state->displayFrame,
                CVHelpers::imageToCvMatWithType(mask, CV_8UC1, 255.0));
        }
        catch (const std::exception& e) {
            std::cerr << "[ERROR] Preview decode failed at frame " << state->currentFrameIndex
                      << " => " << e.what() << '\n';
        }
    }

    if (state->mode == PromptMode::SEED_POINTS) {
        for (size_t i = 0; i < state->currentPoints.size(); ++i) {
            const cv::Scalar color =
                state->currentLabels[i] == 1 ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0);
            cv::circle(
                state->displayFrame,
                cv::Point(state->currentPoints[i].x, state->currentPoints[i].y),
                5,
                color,
                -1);
        }
    } else if (state->drawing || state->hasFinalRect) {
        const SAM2Rect rect = normalizedRect(state->rect);
        cv::rectangle(
            state->displayFrame,
            cv::Rect(rect.x, rect.y, rect.width, rect.height),
            cv::Scalar(0, 255, 255),
            2);
    }

    drawHud(
        &state->displayFrame,
        state->currentFrameIndex,
        state->totalFrames,
        state->anchors.size(),
        state->mode);
    cv::imshow(kWindowName, state->displayFrame);
}

bool gotoFrame(AnchorEditorState* state, int frameIndex)
{
    storeCurrentPromptToAnchor(state);

    state->currentFrameIndex = clampFrameIndex(frameIndex, state->totalFrames);
    if (!loadFrameAt(*state->capture, state->currentFrameIndex, &state->currentFrame)) {
        std::cerr << "[ERROR] Could not read frame " << state->currentFrameIndex << '\n';
        return false;
    }
    if (!preprocessCurrentFrame(state)) {
        return false;
    }

    loadCurrentPromptFromAnchor(state);
    renderAnchorEditor(state);
    return true;
}

SAM2Point clampPointToFrame(const AnchorEditorState& state, int x, int y)
{
    const int clampedX = std::max(0, std::min(x, state.currentFrame.cols - 1));
    const int clampedY = std::max(0, std::min(y, state.currentFrame.rows - 1));
    return SAM2Point(clampedX, clampedY);
}

void onMouseAnchorEditor(int event, int x, int y, int, void* userData)
{
    auto* state = static_cast<AnchorEditorState*>(userData);
    if (!state || state->currentFrame.empty()) {
        return;
    }

    if (state->mode == PromptMode::SEED_POINTS) {
        if (event == cv::EVENT_MBUTTONDOWN) {
            clearCurrentPrompt(state);
            storeCurrentPromptToAnchor(state);
            renderAnchorEditor(state);
            return;
        }

        if (event != cv::EVENT_LBUTTONDOWN && event != cv::EVENT_RBUTTONDOWN) {
            return;
        }

        const SAM2Point point = clampPointToFrame(*state, x, y);
        state->currentPoints.push_back(point);
        state->currentLabels.push_back(event == cv::EVENT_LBUTTONDOWN ? 1 : 0);
        storeCurrentPromptToAnchor(state);
        renderAnchorEditor(state);
        return;
    }

    if (event == cv::EVENT_RBUTTONDOWN || event == cv::EVENT_MBUTTONDOWN) {
        clearCurrentPrompt(state);
        storeCurrentPromptToAnchor(state);
        renderAnchorEditor(state);
        return;
    }

    if (event == cv::EVENT_LBUTTONDOWN) {
        const SAM2Point point = clampPointToFrame(*state, x, y);
        state->drawing = true;
        state->hasFinalRect = false;
        state->rect = SAM2Rect(point.x, point.y, 0, 0);
        renderAnchorEditor(state);
        return;
    }

    if (event == cv::EVENT_MOUSEMOVE && state->drawing) {
        const SAM2Point point = clampPointToFrame(*state, x, y);
        state->rect.width = point.x - state->rect.x;
        state->rect.height = point.y - state->rect.y;
        renderAnchorEditor(state);
        return;
    }

    if (event == cv::EVENT_LBUTTONUP && state->drawing) {
        const SAM2Point point = clampPointToFrame(*state, x, y);
        state->drawing = false;
        state->rect.width = point.x - state->rect.x;
        state->rect.height = point.y - state->rect.y;
        state->rect = normalizedRect(state->rect);
        state->hasFinalRect = state->rect.width > 1 && state->rect.height > 1;
        if (!state->hasFinalRect) {
            state->rect = SAM2Rect();
        }
        storeCurrentPromptToAnchor(state);
        renderAnchorEditor(state);
    }
}

bool collectAnchorPrompts(SAM2* sam,
                          const std::string& videoPath,
                          PromptMode mode,
                          int maxFrames,
                          std::map<int, SAM2Prompts>* anchorsOut,
                          std::map<int, CachedEncoderOutputs>* anchorEncoderCachesOut,
                          int* totalFramesOut)
{
    cv::VideoCapture capture(videoPath);
    if (!capture.isOpened()) {
        std::cerr << "[ERROR] Cannot open video for anchor selection.\n";
        return false;
    }

    const int totalFrames = resolveFrameCount(capture, maxFrames);
    if (totalFrames <= 0) {
        std::cerr << "[ERROR] Could not determine video frame count. Use --max_frames.\n";
        return false;
    }

    AnchorEditorState state;
    state.sam = sam;
    state.capture = &capture;
    state.mode = mode;
    state.totalFrames = totalFrames;

    if (!loadFrameAt(capture, 0, &state.currentFrame)) {
        std::cerr << "[ERROR] Could not read frame 0.\n";
        return false;
    }
    if (!preprocessCurrentFrame(&state)) {
        return false;
    }

    cv::namedWindow(kWindowName, cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback(kWindowName, onMouseAnchorEditor, &state);
    renderAnchorEditor(&state);

    std::cout << "[INFO] Multi-anchor " << promptModeName(mode) << " mode ready.\n";
    std::cout << "[INFO] A/D = +/-1 frame, J/L = +/-10 frames, C = clear current frame,\n"
                 "       Enter/Space = run video, Esc/Q = finish annotation.\n";

    while (true) {
        const int key = cv::waitKey(20) & 0xFF;
        if (key == 13 || key == 10 || key == 32 || key == 27 || key == 'q' || key == 'Q') {
            storeCurrentPromptToAnchor(&state);
            break;
        }

        if (key == 'a' || key == 'A') {
            gotoFrame(&state, state.currentFrameIndex - 1);
        } else if (key == 'd' || key == 'D') {
            gotoFrame(&state, state.currentFrameIndex + 1);
        } else if (key == 'j' || key == 'J') {
            gotoFrame(&state, state.currentFrameIndex - kJumpFrames);
        } else if (key == 'l' || key == 'L') {
            gotoFrame(&state, state.currentFrameIndex + kJumpFrames);
        } else if (key == 'c' || key == 'C') {
            clearCurrentPrompt(&state);
            storeCurrentPromptToAnchor(&state);
            renderAnchorEditor(&state);
        }
    }

    cv::destroyAllWindows();
    *anchorsOut = std::move(state.anchors);
    if (anchorEncoderCachesOut) {
        *anchorEncoderCachesOut = std::move(state.anchorEncoderCaches);
    }
    *totalFramesOut = totalFrames;
    return true;
}

} // namespace

static void printVideoUsage(const char* argv0)
{
    std::cout <<
        "Usage: " << argv0 << " --onnx_test_video [options]\n\n"
        "Options:\n"
        "  --encoder    image_encoder.onnx\n"
        "  --decoder    image_decoder.onnx\n"
        "  --memattn    memory_attention.onnx\n"
        "  --memenc     memory_encoder.onnx\n"
        "  --video      clip.mkv / clip.mp4 (file dialog if omitted)\n"
        "  --device     auto | cpu | cuda[:N] | dml[:N]  (default: auto)\n"
        "  --threads    N   (#CPU threads)\n"
        "  --max_frames N   (early stop / navigation cap)\n"
        "  --prompt     seed_points | bounding_box   (default: seed_points)\n\n"
        "Anchor selection:\n"
        "  A/D = +/-1 frame, J/L = +/-10 frames, C = clear current frame\n"
        "  Enter/Space = run video, Esc/Q = finish annotation\n"
        "  seed_points:  L-click = FG, R-click = BG, M-click = clear current frame\n"
        "  bounding_box: L-drag box, R/M-click = clear current frame\n"
        << std::endl;
}

int runOnnxTestVideo(int argc, char** argv)
{
    std::string encoderPath = "image_encoder.onnx";
    std::string decoderPath = "image_decoder.onnx";
    std::string memAttnPath = "memory_attention.onnx";
    std::string memEncPath = "memory_encoder.onnx";
    std::string videoPath;
    std::string requestedDevice = "auto";

    int threads = static_cast<int>(std::thread::hardware_concurrency());
    if (threads <= 0) {
        threads = 4;
    }
    bool threadsExplicit = false;

    int maxFrames = 0;
    PromptMode mode = PromptMode::SEED_POINTS;

    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--encoder" && i + 1 < argc) {
            encoderPath = argv[++i];
        } else if (arg == "--decoder" && i + 1 < argc) {
            decoderPath = argv[++i];
        } else if (arg == "--memattn" && i + 1 < argc) {
            memAttnPath = argv[++i];
        } else if (arg == "--memenc" && i + 1 < argc) {
            memEncPath = argv[++i];
        } else if (arg == "--video" && i + 1 < argc) {
            videoPath = argv[++i];
        } else if (arg == "--device" && i + 1 < argc) {
            requestedDevice = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            threads = std::stoi(argv[++i]);
            threadsExplicit = true;
        } else if (arg == "--max_frames" && i + 1 < argc) {
            maxFrames = std::stoi(argv[++i]);
        } else if (arg == "--prompt" && i + 1 < argc) {
            const std::string value = argv[++i];
            if (value == "seed_points") {
                mode = PromptMode::SEED_POINTS;
            } else if (value == "bounding_box") {
                mode = PromptMode::BOUNDING_BOX;
            } else {
                std::cerr << "[ERROR] --prompt must be seed_points|bounding_box\n";
                return 1;
            }
        } else if (arg == "--help" || arg == "-h") {
            printVideoUsage(argv[0]);
            return 0;
        }
    }

    if (videoPath.empty()) {
        const wchar_t* filter = L"Video\0*.mp4;*.mkv;*.avi;*.mov\0All\0*.*\0";
        const std::string chosen = openFileDialog(filter, L"Select a Video");
        if (chosen.empty()) {
            std::cerr << "[ERROR] No file chosen\n";
            return 1;
        }
        videoPath = chosen;
    }

    const std::string requestedEncoderPath = encoderPath;
    const std::string requestedDecoderPath = decoderPath;
    const std::string requestedMemAttnPath = memAttnPath;
    const std::string requestedMemEncPath = memEncPath;
    requestedDevice = normalizeDeviceArg(requestedDevice);
    if (requestedDevice.empty()) {
        std::cerr << "[ERROR] --device must be auto|cpu|cuda[:N]|dml[:N]\n";
        return 1;
    }
    const bool forceCpu = ArtifactResolver::isLowCostCpuProfile();
    const std::vector<std::string> deviceCandidates = buildDeviceCandidates(requestedDevice, forceCpu);
    std::string device = deviceCandidates.empty() ? "cpu" : deviceCandidates.front();
    if (!threadsExplicit) {
        threads = ArtifactResolver::preferredRuntimeThreads(threads, device);
    }

    const std::string runtimeProfile = ArtifactResolver::preferredRuntimeProfile();
    std::cout << "[INFO] runtime_profile = " << (runtimeProfile.empty() ? "default" : runtimeProfile) << "\n"
              << "       encoder         = " << encoderPath << "\n"
              << "       decoder         = " << decoderPath << "\n"
              << "       memAttn         = " << memAttnPath << "\n"
              << "       memEnc          = " << memEncPath << "\n"
              << "       video           = " << videoPath << "\n"
              << "       prompt          = " << promptModeName(mode) << "\n"
              << "       device          = " << requestedDevice
              << " (cuda=" << (SAM2::hasCudaDriver() ? "yes" : "no")
              << ", dml=" << (SAM2::hasDirectMLProvider() ? "yes" : "no") << ")\n"
              << "       threads         = " << threads << "\n\n";

    SAM2 sam;
    std::string selectedRuntimeMode;
    auto initializeOnDevice = [&](const std::string& initDevice) -> bool {
        const int initThreads = threadsExplicit
            ? threads
            : ArtifactResolver::preferredRuntimeThreads(threads, initDevice);
        const std::string resolvedEncoder =
            ArtifactResolver::preferQuantizedEncoderPath(requestedEncoderPath, initDevice);
        const auto runtimeSelection = ArtifactResolver::resolveVideoRuntimePaths(
            requestedDecoderPath,
            requestedMemAttnPath,
            requestedMemEncPath,
            false,
            initDevice);

        std::cout << "[INFO] Initialising on " << initDevice
                  << " with " << initThreads << " thread(s)\n";
        std::cout << "[INFO] Resolved encoder    = " << resolvedEncoder << "\n"
                  << "       decoder init       = " << runtimeSelection.decoderInitPath << "\n"
                  << "       decoder propagate  = " << runtimeSelection.decoderPropagatePath << "\n"
                  << "       mem attention      = " << runtimeSelection.memoryAttentionPath << "\n"
                  << "       mem encoder        = " << runtimeSelection.memoryEncoderPath << "\n"
                  << "       video artifacts    = " << runtimeSelection.mode << "\n\n";

        try {
            if (sam.initializeVideo(
                    resolvedEncoder,
                    runtimeSelection.decoderInitPath,
                    runtimeSelection.decoderPropagatePath,
                    runtimeSelection.memoryAttentionPath,
                    runtimeSelection.memoryEncoderPath,
                    initThreads,
                    initDevice)) {
                encoderPath = resolvedEncoder;
                decoderPath = runtimeSelection.decoderInitPath;
                memAttnPath = runtimeSelection.memoryAttentionPath;
                memEncPath = runtimeSelection.memoryEncoderPath;
                selectedRuntimeMode = runtimeSelection.mode;
                device = initDevice;
                threads = initThreads;
                return true;
            }
        } catch (const std::exception& e) {
            std::cerr << "[WARN] SAM2 init on " << initDevice << " failed: " << e.what() << "\n";
        }
        return false;
    };

    bool initialized = false;
    for (const std::string& candidateDevice : deviceCandidates) {
        if (initializeOnDevice(candidateDevice)) {
            initialized = true;
            if (candidateDevice != deviceCandidates.front()) {
                std::cout << "[INFO] Fallback runtime initialised successfully: "
                          << candidateDevice << "\n";
            }
            break;
        }
        if (candidateDevice != "cpu") {
            std::cerr << "[WARN] Runtime " << candidateDevice
                      << " unavailable; trying next candidate.\n";
        }
    }
    if (!initialized) {
        std::cerr << "[ERROR] SAM2 init failed\n";
        return 1;
    }

    std::map<int, SAM2Prompts> anchors;
    std::map<int, CachedEncoderOutputs> anchorEncoderCaches;
    int totalFrames = 0;
    if (!collectAnchorPrompts(
            &sam,
            videoPath,
            mode,
            maxFrames,
            &anchors,
            &anchorEncoderCaches,
            &totalFrames)) {
        return 1;
    }
    if (anchors.empty()) {
        std::cerr << "[ERROR] No anchor annotations were provided.\n";
        return 1;
    }

    std::cout << "[INFO] Anchor frames:";
    for (const auto& entry : anchors) {
        std::cout << ' ' << entry.first;
    }
    std::cout << "\n";

    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "[ERROR] Cannot reopen video for processing.\n";
        return 1;
    }

    const double fpsRaw = cap.get(cv::CAP_PROP_FPS);
    const double fps = fpsRaw > 0.0 ? fpsRaw : 25.0;
    const int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    if (width <= 0 || height <= 0) {
        std::cerr << "[ERROR] Could not determine video resolution.\n";
        return 1;
    }

    const size_t dot = videoPath.find_last_of('.');
    const std::string stem = dot == std::string::npos ? videoPath : videoPath.substr(0, dot);
    const std::string outVideo = stem + "_" + selectedRuntimeMode + "_mask_overlay.avi";
    cv::VideoWriter writer(
        outVideo,
        cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
        fps,
        cv::Size(width, height));
    if (!writer.isOpened()) {
        std::cerr << "[ERROR] Could not open output writer.\n";
        return 1;
    }

    sam.resetMemory();

    SAM2Prompts emptyPrompts;
    bool activeSegment = false;
    int writtenFrames = 0;

    for (int frameIndex = 0; frameIndex < totalFrames; ++frameIndex) {
        cv::Mat frameBGR;
        if (!cap.read(frameBGR) || frameBGR.empty()) {
            break;
        }

        const auto anchorIt = anchors.find(frameIndex);
        if (anchorIt != anchors.end()) {
            sam.resetMemory();
            activeSegment = true;
            std::cout << "[INFO] Anchor frame " << frameIndex << " reset interval\n";
        }

        if (!activeSegment) {
            writer << frameBGR;
            std::cout << "[INFO] Frame " << frameIndex << " inactive (before first anchor)\n";
            ++writtenFrames;
            continue;
        }

        const SAM2Prompts& prompts = anchorIt != anchors.end() ? anchorIt->second : emptyPrompts;
        Image<float> mask;
        if (anchorIt != anchors.end()) {
            const auto cacheIt = anchorEncoderCaches.find(frameIndex);
            if (cacheIt != anchorEncoderCaches.end() && sam.restoreCachedEncoderOutputs(cacheIt->second)) {
                mask = sam.inferMultiFrameCached(SAM2Size(frameBGR.cols, frameBGR.rows), prompts);
            } else {
                mask = sam.inferMultiFrame(CVHelpers::normalizeRGB(frameBGR, 255.0f), prompts);
            }
        } else {
            mask = sam.inferMultiFrame(CVHelpers::normalizeRGB(frameBGR, 255.0f), prompts);
        }

        writer << overlayMask(
            frameBGR,
            CVHelpers::imageToCvMatWithType(mask, CV_8UC1, 255.0));
        ++writtenFrames;
    }

    writer.release();
    cap.release();

    std::cout << "[INFO] Saved " << outVideo << " (" << writtenFrames << " frames)\n";
    return 0;
}
