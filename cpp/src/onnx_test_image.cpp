// sam2-onnx-cpp/cpp/src/onnx_test_image.cpp
#include "openFileDialog.h"
#include "SAM2.h"
#include "CVHelpers.h"
#include "ArtifactResolver.h"

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cctype>
#include <exception>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <chrono>
#include <vector>

using namespace std;
using namespace std::chrono;

static string lowerCopy(string value)
{
    transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char ch) { return static_cast<char>(tolower(ch)); });
    return value;
}

static string normalizeDeviceArg(const string& raw)
{
    string value = lowerCopy(raw);
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

static void appendDeviceCandidate(vector<string>* devices, const string& device)
{
    if (find(devices->begin(), devices->end(), device) == devices->end()) {
        devices->push_back(device);
    }
}

static vector<string> buildDeviceCandidates(const string& requestedDevice, bool forceCpu)
{
    vector<string> devices;
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

static vector<string> splitCommaList(const string& value)
{
    vector<string> items;
    string item;
    stringstream ss(value);
    while (getline(ss, item, ',')) {
        if (!item.empty()) {
            items.push_back(item);
        }
    }
    return items;
}

static bool parseIntList(const string& spec, vector<int>* values)
{
    values->clear();
    for (const string& part : splitCommaList(spec)) {
        try {
            size_t consumed = 0;
            const int value = stoi(part, &consumed);
            if (consumed != part.size()) {
                return false;
            }
            values->push_back(value);
        } catch (...) {
            return false;
        }
    }
    return !values->empty();
}

static bool parseBoxSpec(const string& spec, SAM2Rect* rectOut)
{
    vector<int> values;
    if (!parseIntList(spec, &values) || values.size() != 4) {
        return false;
    }

    const int x1 = values[0];
    const int y1 = values[1];
    const int x2 = values[2];
    const int y2 = values[3];
    SAM2Rect rect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1));
    if (rect.width <= 1 || rect.height <= 1) {
        return false;
    }

    *rectOut = rect;
    return true;
}

static bool parsePointSpec(const string& spec, int defaultLabel, SAM2Point* pointOut, int* labelOut)
{
    vector<int> values;
    if (!parseIntList(spec, &values) || (values.size() != 2 && values.size() != 3)) {
        return false;
    }

    const int label = values.size() == 3 ? values[2] : defaultLabel;
    if (label != 0 && label != 1) {
        return false;
    }

    *pointOut = SAM2Point(values[0], values[1]);
    *labelOut = label;
    return true;
}

static string safeDeviceSuffix(string device)
{
    for (char& ch : device) {
        if (!isalnum(static_cast<unsigned char>(ch))) {
            ch = '_';
        }
    }
    return device;
}

static string overlayPathForDevice(const string& requestedPath, const string& device, bool multiDevice)
{
    if (requestedPath.empty() || !multiDevice) {
        return requestedPath;
    }

    filesystem::path path(requestedPath);
    const string suffix = "_" + safeDeviceSuffix(device);
    return (path.parent_path() / (path.stem().string() + suffix + path.extension().string())).string();
}

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

struct CliPrompt
{
    bool hasBox = false;
    SAM2Rect box;
    vector<SAM2Point> points;
    vector<int> labels;
};

struct ImageRunResult
{
    bool ok = false;
    string device;
    double firstEncoderMs = 0.0;
    double avgEncoderMs = 0.0;
    double minEncoderMs = 0.0;
    double maxEncoderMs = 0.0;
    double firstDecodeMs = 0.0;
    double avgDecodeMs = 0.0;
    double minDecodeMs = 0.0;
    double maxDecodeMs = 0.0;
};

static bool buildPromptsFromCli(const CliPrompt& cliPrompt,
                                PromptMode mode,
                                SAM2Prompts* promptsOut,
                                string* errorOut)
{
    SAM2Prompts prompts;
    if (mode == PromptMode::BOUNDING_BOX) {
        if (!cliPrompt.hasBox) {
            *errorOut = "--box is required with --prompt bounding_box in --no_gui/--benchmark mode";
            return false;
        }
        prompts.rects.push_back(cliPrompt.box);
    } else {
        if (cliPrompt.points.empty()) {
            *errorOut = "at least one --point, --fg, or --bg is required with seed_points in --no_gui/--benchmark mode";
            return false;
        }
        prompts.points = cliPrompt.points;
        prompts.pointLabels = cliPrompt.labels;
    }

    *promptsOut = std::move(prompts);
    return true;
}

static bool runImageNoGuiOnDevice(const string& initDevice,
                                  const string& requestedEncoderPath,
                                  const string& requestedDecoderPath,
                                  PromptMode mode,
                                  const SAM2Prompts& prompts,
                                  const cv::Mat& imgBGR,
                                  int baseThreads,
                                  bool threadsExplicit,
                                  int warmupRuns,
                                  int measuredRuns,
                                  const string& saveOverlayPath,
                                  ImageRunResult* resultOut)
{
    ImageRunResult result;
    result.device = initDevice;

    if (initDevice.rfind("cuda:", 0) == 0 && !SAM2::hasCudaDriver()) {
        cerr << "[WARN] " << initDevice << " unavailable: CUDA runtime/device not detected.\n";
        if (resultOut) *resultOut = result;
        return false;
    }
    if (initDevice.rfind("dml", 0) == 0 && !SAM2::hasDirectMLProvider()) {
        cerr << "[WARN] " << initDevice << " unavailable: DirectML provider not detected.\n";
        if (resultOut) *resultOut = result;
        return false;
    }

    const int initThreads = threadsExplicit
        ? baseThreads
        : ArtifactResolver::preferredRuntimeThreads(baseThreads, initDevice);
    const string resolvedEncoder =
        ArtifactResolver::preferQuantizedEncoderPath(requestedEncoderPath, initDevice);
    const auto decoderSelection = ArtifactResolver::resolveImageDecoderPath(
        requestedDecoderPath,
        mode == PromptMode::SEED_POINTS ? "seed_points" : "bounding_box",
        false,
        initDevice);

    cout << "[INFO] Benchmark init " << initDevice << " with " << initThreads << " thread(s)\n";
    cout << "[INFO] Resolved encoder: " << resolvedEncoder << "\n";
    cout << "[INFO] Image artifacts : " << decoderSelection.mode
         << " (" << decoderSelection.path << ")\n";

    SAM2 sam;
    if (!sam.initialize(resolvedEncoder, decoderSelection.path, initThreads, initDevice)) {
        cerr << "[ERROR] SAM2 init failed on " << initDevice << "\n";
        if (resultOut) *resultOut = result;
        return false;
    }

    const int safeWarmups = max(0, warmupRuns);
    const int safeRuns = max(1, measuredRuns);
    const SAM2Size originalSize(imgBGR.cols, imgBGR.rows);
    const Image<float> normalizedImage = CVHelpers::normalizeRGB(imgBGR, 255.0);

    vector<double> encoderTimes;
    encoderTimes.reserve(static_cast<size_t>(safeRuns));
    for (int i = 0; i < safeWarmups + safeRuns; ++i) {
        const auto encStart = high_resolution_clock::now();
        if (!sam.preprocessImage(normalizedImage)) {
            cerr << "[ERROR] preprocessImage failed on " << initDevice << "\n";
            if (resultOut) *resultOut = result;
            return false;
        }
        const double ms = duration<double, milli>(high_resolution_clock::now() - encStart).count();
        if (i >= safeWarmups) {
            encoderTimes.push_back(ms);
        }
    }

    result.firstEncoderMs = encoderTimes.front();
    result.minEncoderMs = *min_element(encoderTimes.begin(), encoderTimes.end());
    result.maxEncoderMs = *max_element(encoderTimes.begin(), encoderTimes.end());
    double encoderSum = 0.0;
    for (double ms : encoderTimes) {
        encoderSum += ms;
    }
    result.avgEncoderMs = encoderSum / static_cast<double>(encoderTimes.size());

    sam.setPrompts(prompts, originalSize);

    vector<double> decodeTimes;
    decodeTimes.reserve(static_cast<size_t>(safeRuns));
    Image<float> lastMask;

    for (int i = 0; i < safeWarmups + safeRuns; ++i) {
        const auto decStart = high_resolution_clock::now();
        Image<float> mask = sam.inferSingleFrame(originalSize);
        const double ms = duration<double, milli>(high_resolution_clock::now() - decStart).count();
        if (mask.getWidth() <= 0 || mask.getHeight() <= 0) {
            cerr << "[ERROR] decoder returned an empty mask on " << initDevice << "\n";
            if (resultOut) *resultOut = result;
            return false;
        }
        if (i >= safeWarmups) {
            decodeTimes.push_back(ms);
            lastMask = std::move(mask);
        }
    }

    result.firstDecodeMs = decodeTimes.front();
    result.minDecodeMs = *min_element(decodeTimes.begin(), decodeTimes.end());
    result.maxDecodeMs = *max_element(decodeTimes.begin(), decodeTimes.end());
    double sum = 0.0;
    for (double ms : decodeTimes) {
        sum += ms;
    }
    result.avgDecodeMs = sum / static_cast<double>(decodeTimes.size());
    result.ok = true;

    cout << "[RESULT] device=" << initDevice
         << " encoder_first_ms=" << result.firstEncoderMs
         << " encoder_avg_ms=" << result.avgEncoderMs
         << " encoder_min_ms=" << result.minEncoderMs
         << " encoder_max_ms=" << result.maxEncoderMs
         << " decode_first_ms=" << result.firstDecodeMs
         << " decode_avg_ms=" << result.avgDecodeMs
         << " decode_min_ms=" << result.minDecodeMs
         << " decode_max_ms=" << result.maxDecodeMs
         << " runs=" << safeRuns
         << " warmup=" << safeWarmups
         << "\n";

    if (!saveOverlayPath.empty()) {
        const cv::Mat maskCv = CVHelpers::imageToCvMatWithType(lastMask, CV_8UC1, 255.0);
        const cv::Mat overlay = overlayMask(imgBGR, maskCv);
        if (!cv::imwrite(saveOverlayPath, overlay)) {
            cerr << "[WARN] Failed to save overlay to " << saveOverlayPath << "\n";
        } else {
            cout << "[INFO] Saved overlay to " << saveOverlayPath << "\n";
        }
    }

    if (resultOut) *resultOut = result;
    return true;
}

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
            st->drawing=false;
            st->rect.width  = x - st->rect.x;
            st->rect.height = y - st->rect.y;
            SAM2Rect r = st->rect;
            if (r.width<0)  { r.x += r.width;  r.width  = -r.width; }
            if (r.height<0) { r.y += r.height; r.height = -r.height; }
            if (r.width <= 1 || r.height <= 1) {
                st->hasFinalRect = false;
                st->rect = SAM2Rect();
                cout << "[INFO] Ignored empty box; drag a rectangle with non-zero width and height.\n";
                updateDisplay(st);
                return;
            }
            st->rect = r;
            st->hasFinalRect=true;
            cout << "[INFO] Box finalised: ("<<st->rect.x<<","<<st->rect.y
                 <<") to ("<<st->rect.x + st->rect.width<<","<<st->rect.y + st->rect.height<<")\n";
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
    bool   threadsExplicit = false;
    string requestedDevice = "auto";
    PromptMode mode  = PromptMode::SEED_POINTS;             // default
    CliPrompt cliPrompt;
    bool noGui = false;
    bool benchmark = false;
    int benchmarkRuns = 5;
    int benchmarkWarmups = 1;
    string benchmarkDevicesSpec;
    string saveOverlayPath;

    for(int i=2;i<argc;++i)   // argv[1] is "--onnx_test_image"
    {
        string a = argv[i];
        if((a=="--encoder") && i+1<argc)       encoderPath = argv[++i];
        else if((a=="--decoder") && i+1<argc)  decoderPath = argv[++i];
        else if((a=="--image")   && i+1<argc)  imagePath   = argv[++i];
        else if((a=="--device")  && i+1<argc)  requestedDevice = argv[++i];
        else if((a=="--threads") && i+1<argc)  { threads = stoi(argv[++i]); threadsExplicit = true; }
        else if(a=="--no_gui")                 noGui = true;
        else if(a=="--benchmark")              { benchmark = true; noGui = true; }
        else if((a=="--runs") && i+1<argc)     benchmarkRuns = max(1, stoi(argv[++i]));
        else if((a=="--warmup") && i+1<argc)   benchmarkWarmups = max(0, stoi(argv[++i]));
        else if((a=="--benchmark_devices") && i+1<argc) benchmarkDevicesSpec = argv[++i];
        else if((a=="--save_overlay") && i+1<argc) saveOverlayPath = argv[++i];
        else if((a=="--box") && i+1<argc)
        {
            SAM2Rect box;
            if (!parseBoxSpec(argv[++i], &box)) {
                cerr<<"[ERROR] --box must be x1,y1,x2,y2 with non-zero width and height\n";
                return 1;
            }
            cliPrompt.box = box;
            cliPrompt.hasBox = true;
            mode = PromptMode::BOUNDING_BOX;
        }
        else if((a=="--point") && i+1<argc)
        {
            SAM2Point point;
            int label = 1;
            if (!parsePointSpec(argv[++i], 1, &point, &label)) {
                cerr<<"[ERROR] --point must be x,y or x,y,label where label is 0|1\n";
                return 1;
            }
            cliPrompt.points.push_back(point);
            cliPrompt.labels.push_back(label);
            mode = PromptMode::SEED_POINTS;
        }
        else if((a=="--fg") && i+1<argc)
        {
            SAM2Point point;
            int label = 1;
            if (!parsePointSpec(argv[++i], 1, &point, &label)) {
                cerr<<"[ERROR] --fg must be x,y\n";
                return 1;
            }
            cliPrompt.points.push_back(point);
            cliPrompt.labels.push_back(1);
            mode = PromptMode::SEED_POINTS;
        }
        else if((a=="--bg") && i+1<argc)
        {
            SAM2Point point;
            int label = 0;
            if (!parsePointSpec(argv[++i], 0, &point, &label)) {
                cerr<<"[ERROR] --bg must be x,y\n";
                return 1;
            }
            cliPrompt.points.push_back(point);
            cliPrompt.labels.push_back(0);
            mode = PromptMode::SEED_POINTS;
        }
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
            "  --device   auto|cpu|cuda[:N]|dml[:N]  (default: auto)\n"
            "  --threads  N                      (#CPU threads, default: HW-concurrency)\n"
            "  --prompt   seed_points|bounding_box   (default: seed_points)\n"
            "  --no_gui                           (run one prompt and exit)\n"
            "  --benchmark                        (run deterministic timing and exit)\n"
            "  --benchmark_devices cpu,dml:0,cuda:0\n"
            "  --runs N --warmup N                (benchmark decode runs, default: 5/1)\n"
            "  --box x1,y1,x2,y2                  (bbox prompt; implies bounding_box)\n"
            "  --fg x,y --bg x,y                  (seed prompts; can repeat)\n"
            "  --point x,y[,label]                (label: 1=FG, 0=BG)\n"
            "  --save_overlay out.png             (non-GUI output overlay)\n\n"
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

    /* ---------------  read image  --------------- */
    cv::Mat imgBGR = cv::imread(imagePath);
    if(imgBGR.empty()){ cerr<<"[ERROR] could not load "<<imagePath<<"\n"; return 1; }

    /* ---------------  init SAM-2 (GPU if available)  --------------- */
    const string requestedEncoderPath = encoderPath;
    const string requestedDecoderPath = decoderPath;
    requestedDevice = normalizeDeviceArg(requestedDevice);
    if (requestedDevice.empty()) {
        cerr<<"[ERROR] --device must be auto|cpu|cuda[:N]|dml[:N]\n";
        return 1;
    }
    const bool forceCpu = ArtifactResolver::isLowCostCpuProfile();
    const vector<string> deviceCandidates = buildDeviceCandidates(requestedDevice, forceCpu);
    string device = deviceCandidates.empty() ? "cpu" : deviceCandidates.front();
    if (!threadsExplicit) {
        threads = ArtifactResolver::preferredRuntimeThreads(threads, device);
    }

    if (noGui) {
        SAM2Prompts prompts;
        string promptError;
        if (!buildPromptsFromCli(cliPrompt, mode, &prompts, &promptError)) {
            cerr << "[ERROR] " << promptError << "\n";
            return 1;
        }

        vector<string> runDevices;
        if (benchmark) {
            const string devicesSpec = benchmarkDevicesSpec.empty()
                ? "cpu,dml:0,cuda:0"
                : benchmarkDevicesSpec;
            for (const string& item : splitCommaList(devicesSpec)) {
                const string normalized = normalizeDeviceArg(item);
                if (normalized.empty()) {
                    cerr << "[ERROR] --benchmark_devices contains invalid device: " << item << "\n";
                    return 1;
                }
                if (normalized == "auto") {
                    for (const string& candidate : buildDeviceCandidates("auto", forceCpu)) {
                        appendDeviceCandidate(&runDevices, candidate);
                    }
                } else {
                    appendDeviceCandidate(&runDevices, normalized);
                }
            }
        } else {
            runDevices = deviceCandidates;
        }

        bool anyOk = false;
        const bool multiDevice = runDevices.size() > 1;
        for (const string& runDevice : runDevices) {
            ImageRunResult result;
            const string overlayPath = overlayPathForDevice(saveOverlayPath, runDevice, multiDevice);
            const bool ok = runImageNoGuiOnDevice(
                runDevice,
                requestedEncoderPath,
                requestedDecoderPath,
                mode,
                prompts,
                imgBGR,
                threads,
                threadsExplicit,
                benchmark ? benchmarkWarmups : 0,
                benchmark ? benchmarkRuns : 1,
                overlayPath,
                &result);
            anyOk = anyOk || ok;
            if (ok && !benchmark) {
                break;
            }
        }

        return anyOk ? 0 : 1;
    }

    const string runtimeProfile = ArtifactResolver::preferredRuntimeProfile();
    cout<<"[INFO] runtime_profile="<<(runtimeProfile.empty() ? "default" : runtimeProfile)<<"\n"
        <<"       encoder="<<encoderPath<<"\n"
        <<"       decoder="<<decoderPath<<"\n"
        <<"       image  ="<<imagePath<<"\n"
        <<"       prompt ="<<(mode==PromptMode::SEED_POINTS?"seed_points":"bounding_box")<<"\n"
        <<"       device ="<<requestedDevice
        <<" (cuda="<<(SAM2::hasCudaDriver() ? "yes" : "no")
        <<", dml="<<(SAM2::hasDirectMLProvider() ? "yes" : "no")<<")\n"
        <<"       threads="<<threads<<"\n\n";

    AppState st;
    auto initializeOnDevice = [&](const string& initDevice) -> bool {
        const int initThreads = threadsExplicit
            ? threads
            : ArtifactResolver::preferredRuntimeThreads(threads, initDevice);
        const string resolvedEncoder =
            ArtifactResolver::preferQuantizedEncoderPath(requestedEncoderPath, initDevice);
        const auto decoderSelection = ArtifactResolver::resolveImageDecoderPath(
            requestedDecoderPath,
            mode==PromptMode::SEED_POINTS ? "seed_points" : "bounding_box",
            false,
            initDevice);

        cout<<"[INFO] Initialising on "<<initDevice<<" with "<<initThreads<<" thread(s)\n";
        cout<<"[INFO] Resolved encoder: "<<resolvedEncoder<<"\n";
        cout<<"[INFO] Image artifacts : "<<decoderSelection.mode<<" ("<<decoderSelection.path<<")\n";

        try {
            if (st.sam.initialize(resolvedEncoder, decoderSelection.path, initThreads, initDevice)) {
                encoderPath = resolvedEncoder;
                decoderPath = decoderSelection.path;
                device = initDevice;
                threads = initThreads;
                return true;
            }
        } catch (const std::exception& e) {
            cerr<<"[WARN] SAM2 init on "<<initDevice<<" failed: "<<e.what()<<"\n";
        }
        return false;
    };

    bool initialized = false;
    for (const string& candidateDevice : deviceCandidates) {
        if (initializeOnDevice(candidateDevice)) {
            initialized = true;
            if (candidateDevice != deviceCandidates.front()) {
                cout<<"[INFO] Fallback runtime initialised successfully: "<<candidateDevice<<"\n";
            }
            break;
        }
        if (candidateDevice != "cpu") {
            cerr<<"[WARN] Runtime "<<candidateDevice<<" unavailable; trying next candidate.\n";
        }
    }
    if (!initialized) {
        cerr<<"[ERROR] SAM2 init failed\n";
        return 1;
    }

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
