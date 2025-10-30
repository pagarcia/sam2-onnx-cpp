// sam2-onnx-cpp/cpp/src/SAM2Session.cpp
#include "SAM2.h"
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <cstring> // for memcpy

#ifdef _WIN32
  #include <windows.h>
/** Utility to convert std::string to wide string on Windows. */
static std::wstring strToWstr(const std::string &str)
{
    int sizeNeeded = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), NULL, 0);
    std::wstring wstr(sizeNeeded, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), &wstr[0], sizeNeeded);
    return wstr;
}
#endif

#ifdef __linux__
  #include <dlfcn.h>  // for dlopen/dlsym on Linux
#endif

// --------------------
// Constructor / Destructor
// --------------------
SAM2::SAM2() {}
SAM2::~SAM2() {
    clearSessions();
}

// --------------------
// Basic checks / helpers
// --------------------
bool SAM2::modelExists(const std::string &modelPath)
{
    std::ifstream f(modelPath.c_str());
    return f.good();
}

bool SAM2::clearSessions()
{
    try {
        // Reset all session pointers
        m_imgEncoderSession.reset();
        m_imgDecoderSession.reset();
        m_memAttentionSession.reset();
        m_memEncoderSession.reset();

        // Clear out shape/data vectors
        m_inputShapeEncoder.clear();
        m_outputShapeEncoder.clear();
        m_highResFeatures1Shape.clear();
        m_highResFeatures2Shape.clear();

        m_outputTensorValuesEncoder.clear();
        m_highResFeatures1.clear();
        m_highResFeatures2.clear();

        // Reset memory states for multi-frame usage
        m_hasMemory = false;
        m_maskMemFeatures.clear();
        m_maskMemFeaturesShape.clear();
        m_maskMemPosEnc.clear();
        m_maskMemPosEncShape.clear();
        m_temporalCode.clear();
        m_temporalCodeShape.clear();
    }
    catch(...) {
        return false;
    }
    return true;
}

void SAM2::resetMemory()
{
    m_hasMemory          = false;
    m_maskMemFeatures.clear();
    m_maskMemFeaturesShape.clear();
    m_maskMemPosEnc.clear();
    m_maskMemPosEncShape.clear();
    m_temporalCode.clear();
    m_temporalCodeShape.clear();
}

SAM2Size SAM2::getInputSize()
{
    // Typically [1,3,1024,1024] => shape[2]=1024 (H), shape[3]=1024 (W)
    if (m_inputShapeEncoder.size() >= 4) {
        return SAM2Size(
            static_cast<int>(m_inputShapeEncoder[3]),
            static_cast<int>(m_inputShapeEncoder[2])
        );
    }
    return SAM2Size(0, 0);
}

/* Small helper to check ORT "C" API calls in this TU */
static inline void _ortThrowIf(OrtStatus* st, const char* what) {
    if (st) {
        const char* msg = Ort::GetApi().GetErrorMessage(st);
        std::ostringstream oss;
        oss << what << " : " << (msg ? msg : "(null)");
        Ort::GetApi().ReleaseStatus(st);
        throw std::runtime_error(oss.str());
    }
}

void SAM2::setupSessionOptions(Ort::SessionOptions &options,
                               int threadsNumber,
                               GraphOptimizationLevel optLevel,
                               const std::string &device)
{
    // Conservative defaults help stability on macOS CPU
    options.SetIntraOpNumThreads(std::max(1, threadsNumber));
    options.SetInterOpNumThreads(1);
    options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    options.SetGraphOptimizationLevel(optLevel);

#if defined(__APPLE__)
    options.DisableMemPattern();
    // If you ever hit hardened-runtime allocator warnings, you can also:
    // options.DisableCpuMemArena();
#endif

    if (device == "cpu") {
        std::cout << "[DEBUG] Using CPU execution provider." << std::endl;
        int use_arena = 1;
        _ortThrowIf(OrtSessionOptionsAppendExecutionProvider_CPU(options, use_arena),
                    "Append CPU EP failed");
    }
    else if (device.rfind("cuda:", 0) == 0) {
        std::cout << "[DEBUG] Using CUDA execution provider." << std::endl;
        int gpuId = 0;
        try { gpuId = std::stoi(device.substr(5)); } catch (...) { gpuId = 0; }

#if !defined(__APPLE__)
        OrtCUDAProviderOptions cudaOpts{};  // minimal
        cudaOpts.device_id = gpuId;
        options.AppendExecutionProvider_CUDA(cudaOpts);
#endif
        // CPU fallback (shape ops / constant folding)
        int use_arena = 1;
        _ortThrowIf(OrtSessionOptionsAppendExecutionProvider_CPU(options, use_arena),
                    "Append CPU EP (fallback) failed");
    }
    else if (device.rfind("coreml", 0) == 0) {
#ifdef __APPLE__
        std::cout << "[DEBUG] Using CoreML execution provider." << std::endl;
        uint32_t coreml_flags = 0; // COREML_FLAG_USE_NONE
        _ortThrowIf(OrtSessionOptionsAppendExecutionProvider_CoreML(options, coreml_flags),
                    "Append CoreML EP failed");
#else
        std::cout << "[WARN] CoreML requested but not supported on this platform. Defaulting to CPU.\n";
        int use_arena = 1;
        _ortThrowIf(OrtSessionOptionsAppendExecutionProvider_CPU(options, use_arena),
                    "Append CPU EP (fallback) failed");
#endif
    }
    else {
        std::cout << "[DEBUG] Unknown device type. Defaulting to CPU execution provider.\n";
        int use_arena = 1;
        _ortThrowIf(OrtSessionOptionsAppendExecutionProvider_CPU(options, use_arena),
                    "Append default CPU EP failed");
    }
}

// --------------------
// Initialize methods
// --------------------
bool SAM2::initialize(const std::string &encoderPath,
                      const std::string &decoderPath,
                      int threadsNumber,
                      std::string device)
{
    clearSessions();

    if (!modelExists(encoderPath) || !modelExists(decoderPath)) {
        std::cerr << "[ERROR] Model file not found.\n";
        return false;
    }

    // Base options: encoder vs decoder
    setupSessionOptions(m_encoderOptions, threadsNumber,
                        GraphOptimizationLevel::ORT_ENABLE_EXTENDED, device);
    setupSessionOptions(m_decoderOptions, threadsNumber,
                        GraphOptimizationLevel::ORT_DISABLE_ALL, device);
    m_decoderOptions.AddConfigEntry("session.disable_gemm_fast_gelu_fusion", "1");

    // ---- macOS CPU "safe" mode for the encoder ----
#if defined(__APPLE__)
    {
        const bool cpu_only = (device.rfind("cuda:",0) != 0) && (device.rfind("coreml",0) != 0);
        if (cpu_only) {
            m_encoderOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
            m_encoderOptions.AddConfigEntry("session.disable_gemm_fast_gelu_fusion", "1");
            m_encoderOptions.AddConfigEntry("session.disable_prepacking", "1");
            m_encoderOptions.DisableMemPattern();
        }
    }
#endif

    try {
#ifdef _WIN32
        std::wstring wEnc = strToWstr(encoderPath);
        std::wstring wDec = strToWstr(decoderPath);

        m_imgEncoderSession = std::make_unique<Ort::Session>(m_encoderEnv, wEnc.c_str(), m_encoderOptions);
        m_imgDecoderSession = std::make_unique<Ort::Session>(m_decoderEnv, wDec.c_str(), m_decoderOptions);
#else
        m_imgEncoderSession = std::make_unique<Ort::Session>(m_encoderEnv, encoderPath.c_str(), m_encoderOptions);
        m_imgDecoderSession = std::make_unique<Ort::Session>(m_decoderEnv, decoderPath.c_str(), m_decoderOptions);
#endif

        // ---------- Shape handling ----------
        const bool mac_cpu =
        #ifdef __APPLE__
            (device.rfind("cuda:",0) != 0) && (device.rfind("coreml",0) != 0);
        #else
            false;
        #endif

        if (mac_cpu) {
            // Avoid ORT shape introspection (crash path) on macOS CPU.
            m_inputShapeEncoder      = {1, 3,   1024, 1024};
            m_outputShapeEncoder     = {1, 256,   64,   64};
            m_highResFeatures1Shape  = {1, 32,   256,  256};
            m_highResFeatures2Shape  = {1, 64,   128,  128};
            std::cout << "[INFO] macOS CPU: using fixed SAM2 shapes.\n";
        } else {
            // Normal, safe shape queries
            auto encInputInfo = m_imgEncoderSession->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
            m_inputShapeEncoder = encInputInfo.GetShape();

            auto out0Info = m_imgEncoderSession->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();
            m_outputShapeEncoder = out0Info.GetShape();

            auto out1Info = m_imgEncoderSession->GetOutputTypeInfo(1).GetTensorTypeAndShapeInfo();
            m_highResFeatures1Shape = out1Info.GetShape();

            auto out2Info = m_imgEncoderSession->GetOutputTypeInfo(2).GetTensorTypeAndShapeInfo();
            m_highResFeatures2Shape = out2Info.GetShape();
        }

        // Gather node info (names etc.)
        m_imgEncoderInputNodes  = getSessionNodes(m_imgEncoderSession.get(), true);
        m_imgEncoderOutputNodes = getSessionNodes(m_imgEncoderSession.get(), false);
        m_imgDecoderInputNodes  = getSessionNodes(m_imgDecoderSession.get(), true);
        m_imgDecoderOutputNodes = getSessionNodes(m_imgDecoderSession.get(), false);

        std::cout << "[DEBUG] SAM2::initialize() success.\n";
    }
    catch(const std::exception &e) {
        std::cerr << "[ERROR] SAM2::initialize() => " << e.what() << std::endl;
        return false;
    }

    return true;
}

bool SAM2::initializeVideo(const std::string &encoderPath,
                           const std::string &decoderPath,
                           const std::string &memAttentionPath,
                           const std::string &memEncoderPath,
                           int threadsNumber,
                           std::string device)
{
    if(!initialize(encoderPath, decoderPath, threadsNumber, device)) {
        std::cerr << "[ERROR] initializeVideo => base init failed.\n";
        return false;
    }
    if(!modelExists(memAttentionPath) || !modelExists(memEncoderPath)) {
        std::cerr << "[ERROR] memory models not found.\n";
        return false;
    }

    setupSessionOptions(m_memAttentionOptions, threadsNumber, GraphOptimizationLevel::ORT_DISABLE_ALL, device);
    setupSessionOptions(m_memEncoderOptions,    threadsNumber, GraphOptimizationLevel::ORT_DISABLE_ALL, device);
    m_memEncoderOptions.AddConfigEntry("session.disable_gemm_fast_gelu_fusion", "1");

#if defined(__APPLE__)
    {
        const bool cpu_only = (device.rfind("cuda:",0) != 0) && (device.rfind("coreml",0) != 0);
        if (cpu_only) {
            m_memAttentionOptions.AddConfigEntry("session.disable_gemm_fast_gelu_fusion", "1");
            m_memAttentionOptions.AddConfigEntry("session.disable_prepacking", "1");
            m_memAttentionOptions.DisableMemPattern();

            m_memEncoderOptions.AddConfigEntry("session.disable_prepacking", "1");
            m_memEncoderOptions.DisableMemPattern();
        }
    }
#endif

    try {
#ifdef _WIN32
        std::wstring wAttn = strToWstr(memAttentionPath);
        std::wstring wEnc2 = strToWstr(memEncoderPath);

        m_memAttentionSession = std::make_unique<Ort::Session>(m_memAttentionEnv, wAttn.c_str(), m_memAttentionOptions);
        m_memEncoderSession   = std::make_unique<Ort::Session>(m_memEncoderEnv,  wEnc2.c_str(), m_memEncoderOptions);
#else
        m_memAttentionSession = std::make_unique<Ort::Session>(m_memAttentionEnv, memAttentionPath.c_str(), m_memAttentionOptions);
        m_memEncoderSession   = std::make_unique<Ort::Session>(m_memEncoderEnv,  memEncoderPath.c_str(),  m_memEncoderOptions);
#endif

        m_memAttentionInputNodes  = getSessionNodes(m_memAttentionSession.get(), true);
        m_memAttentionOutputNodes = getSessionNodes(m_memAttentionSession.get(), false);
        m_memEncoderInputNodes    = getSessionNodes(m_memEncoderSession.get(),    true);
        m_memEncoderOutputNodes   = getSessionNodes(m_memEncoderSession.get(),    false);

        std::cout<<"[DEBUG] SAM2::initializeVideo() => memAttn+memEnc loaded.\n";
    }
    catch(const std::exception &e){
        std::cerr<<"[ERROR] initVideo => "<< e.what()<<"\n";
        return false;
    }
    return true;
}

// --------------------
// The single runSession helper
// --------------------
std::variant<std::vector<Ort::Value>, std::string>
SAM2::runSession(Ort::Session* session,
                 const std::vector<SAM2Node> &inputNodes,
                 const std::vector<SAM2Node> &outputNodes,
                 const std::vector<Ort::Value> &inputTensors,
                 const std::string &debugName)
{
    if(!session){
        return std::string("[ERROR] runSession("+debugName+"): session is null.\n");
    }
    std::vector<const char*> inNames, outNames;
    inNames.reserve(inputNodes.size());
    outNames.reserve(outputNodes.size());

    for(const auto &nd : inputNodes) {
        inNames.push_back(nd.name.c_str());
    }
    for(const auto &nd : outputNodes) {
        outNames.push_back(nd.name.c_str());
    }

    try {
        auto outputs = session->Run(
            Ort::RunOptions{nullptr},
            inNames.data(),
            const_cast<Ort::Value*>(inputTensors.data()),  // ORT wants non-const
            inputTensors.size(),
            outNames.data(),
            outNames.size()
        );
        return outputs; // success => vector<Ort::Value>
    }
    catch(const std::exception &e){
        std::ostringstream oss;
        oss << "[ERROR] runSession(" << debugName << ") => " << e.what();
        return oss.str();
    }
}

// --------------------
// Pipeline-step methods
// --------------------
std::variant<std::vector<Ort::Value>, std::string>
SAM2::runImageEncoderSession(const std::vector<Ort::Value> &inputTensors)
{
    return runSession(m_imgEncoderSession.get(),
                      m_imgEncoderInputNodes,
                      m_imgEncoderOutputNodes,
                      inputTensors,
                      "imgEncoderSession");
}

std::variant<std::vector<Ort::Value>, std::string>
SAM2::runImageDecoderSession(const std::vector<Ort::Value> &inputTensors)
{
    return runSession(m_imgDecoderSession.get(),
                      m_imgDecoderInputNodes,
                      m_imgDecoderOutputNodes,
                      inputTensors,
                      "imgDecoderSession");
}

std::variant<std::vector<Ort::Value>, std::string>
SAM2::runMemAttentionSession(const std::vector<Ort::Value> &inputTensors)
{
    return runSession(m_memAttentionSession.get(),
                      m_memAttentionInputNodes,
                      m_memAttentionOutputNodes,
                      inputTensors,
                      "memAttentionSession");
}

std::variant<std::vector<Ort::Value>, std::string>
SAM2::runMemEncoderSession(const std::vector<Ort::Value> &inputTensors)
{
    return runSession(m_memEncoderSession.get(),
                      m_memEncoderInputNodes,
                      m_memEncoderOutputNodes,
                      inputTensors,
                      "memEncoderSession");
}


std::vector<SAM2Node> SAM2::getSessionNodes(Ort::Session* session, bool isInput)
{
    std::vector<SAM2Node> nodes;
    Ort::AllocatorWithDefaultOptions alloc;
    size_t count = isInput ? session->GetInputCount() : session->GetOutputCount();
    for(size_t i = 0; i < count; i++){
        SAM2Node node;
        auto namePtr = isInput ? session->GetInputNameAllocated(i, alloc)
                               : session->GetOutputNameAllocated(i, alloc);
        node.name = std::string(namePtr.get());
        auto shape = isInput ? session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape()
                             : session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        node.dim.assign(shape.begin(), shape.end());
        nodes.push_back(std::move(node));
    }
    return nodes;
}

// ─── SAM2::hasCudaDriver – lean release version ──────────────────────
bool SAM2::hasCudaDriver()
{
    static int cached = -1;  // -1 ⇒ not checked yet
    if (cached != -1) return cached;

#if defined(_WIN32)
    // Windows: try to load CUDA runtime DLL and query device count.
    static HMODULE hCUDART =
        LoadLibraryExW(L"cudart64_12.dll", nullptr, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!hCUDART) { cached = 0; return false; }

    using GetCnt = int (__cdecl*)(int*);
    auto fn = reinterpret_cast<GetCnt>(GetProcAddress(hCUDART, "cudaGetDeviceCount"));
    if (!fn) { cached = 0; return false; }

    int n = 0;
    int err = fn(&n); // 0 == cudaSuccess
    cached = (err == 0 && n > 0) ? 1 : 0;
    return cached;

#elif defined(__linux__)
    // Linux: dlopen libcudart and dlsym cudaGetDeviceCount.
    void* hCUDART = dlopen("libcudart.so.12", RTLD_LAZY | RTLD_LOCAL);
    if (!hCUDART) { cached = 0; return false; }

    using GetCnt = int (*)(int*);
    auto fn = reinterpret_cast<GetCnt>(dlsym(hCUDART, "cudaGetDeviceCount"));
    if (!fn) { cached = 0; return false; }

    int n = 0;
    int err = fn(&n);
    cached = (err == 0 && n > 0) ? 1 : 0;
    return cached;

#else
    // macOS (and other platforms): CUDA is not available.
    cached = 0;
    return false;
#endif
}
