#include "SAM2.h"
#include "ArtifactResolver.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#ifdef _WIN32
  #include <windows.h>
static std::wstring strToWstr(const std::string &str)
{
    const int sizeNeeded = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), static_cast<int>(str.size()), nullptr, 0);
    std::wstring wstr(sizeNeeded, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), static_cast<int>(str.size()), &wstr[0], sizeNeeded);
    return wstr;
}
#endif

#ifdef __linux__
  #include <dlfcn.h>
#endif

namespace {

std::string lowerCopy(const std::string &value)
{
    std::string lowered = value;
    std::transform(
        lowered.begin(),
        lowered.end(),
        lowered.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return lowered;
}

size_t getenvSizeT(const char* name, size_t fallback, size_t minValue)
{
    const char* value = std::getenv(name);
    if (!value || !*value) {
        return fallback;
    }

    try {
        const size_t parsed = static_cast<size_t>(std::stoull(value));
        return std::max(parsed, minValue);
    } catch (...) {
        return fallback;
    }
}

size_t preferredVideoMemoryFrameLimit()
{
    if (ArtifactResolver::isLowCostCpuProfile()) {
        return getenvSizeT("SAM2_ORT_VIDEO_MAX_MEMORY_FRAMES", 3u, 1u);
    }

    const std::string autoPolicy = ArtifactResolver::preferredVideoAutoPolicy();
    const size_t fallback = autoPolicy == "speed" ? 4u : 7u;
    return getenvSizeT("SAM2_ORT_VIDEO_MAX_MEMORY_FRAMES", fallback, 1u);
}

size_t preferredVideoObjectPointerLimit()
{
    if (ArtifactResolver::isLowCostCpuProfile()) {
        return getenvSizeT("SAM2_ORT_VIDEO_MAX_OBJECT_POINTERS", 4u, 1u);
    }

    const std::string autoPolicy = ArtifactResolver::preferredVideoAutoPolicy();
    const size_t fallback = autoPolicy == "speed" ? 8u : 16u;
    return getenvSizeT("SAM2_ORT_VIDEO_MAX_OBJECT_POINTERS", fallback, 1u);
}

std::vector<SAM2Node> getSessionNodesInternal(Ort::Session* session, bool isInput, bool includeShapes)
{
    std::vector<SAM2Node> nodes;
    if (!session) {
        return nodes;
    }

    Ort::AllocatorWithDefaultOptions alloc;
    const size_t count = isInput ? session->GetInputCount() : session->GetOutputCount();
    nodes.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        SAM2Node node;
        auto namePtr = isInput ? session->GetInputNameAllocated(i, alloc)
                               : session->GetOutputNameAllocated(i, alloc);
        node.name = std::string(namePtr.get());
        if (includeShapes) {
            auto shape = isInput ? session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape()
                                 : session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            node.dim.assign(shape.begin(), shape.end());
        }
        nodes.push_back(std::move(node));
    }

    return nodes;
}

static inline void ortThrowIf(OrtStatus* st, const char* what)
{
    if (!st) {
        return;
    }

    const char* msg = Ort::GetApi().GetErrorMessage(st);
    std::ostringstream oss;
    oss << what << " : " << (msg ? msg : "(null)");
    Ort::GetApi().ReleaseStatus(st);
    throw std::runtime_error(oss.str());
}

} // namespace

SAM2::SAM2() {}

SAM2::~SAM2()
{
    clearSessions();
}

bool SAM2::modelExists(const std::string &modelPath)
{
    std::ifstream f(modelPath.c_str());
    return f.good();
}

bool SAM2::clearSessions()
{
    try {
        m_imgEncoderSession.reset();
        m_imgDecoderSession.reset();
        m_videoPropDecoderSession.reset();
        m_memAttentionSession.reset();
        m_memEncoderSession.reset();

        m_imgEncoderInputNodes.clear();
        m_imgEncoderOutputNodes.clear();
        m_imgDecoderInputNodes.clear();
        m_imgDecoderOutputNodes.clear();
        m_videoPropDecoderInputNodes.clear();
        m_videoPropDecoderOutputNodes.clear();
        m_memAttentionInputNodes.clear();
        m_memAttentionOutputNodes.clear();
        m_memEncoderInputNodes.clear();
        m_memEncoderOutputNodes.clear();

        m_imgEncoderInputNames.clear();
        m_imgEncoderOutputNames.clear();
        m_imgDecoderInputNames.clear();
        m_imgDecoderOutputNames.clear();
        m_videoPropDecoderInputNames.clear();
        m_videoPropDecoderOutputNames.clear();
        m_memAttentionInputNames.clear();
        m_memAttentionOutputNames.clear();
        m_memEncoderInputNames.clear();
        m_memEncoderOutputNames.clear();
        m_imgDecoderImageOutputNames.clear();
        m_imgDecoderVideoOutputNames.clear();
        m_videoPropDecoderVideoOutputNames.clear();
        m_memEncoderStateOutputNames.clear();

        m_inputShapeEncoder.clear();
        m_encoderEmbedIndex = -1;
        m_encoderCurrentVisionFeatIndex = -1;
        m_encoderHighRes0Index = -1;
        m_encoderHighRes1Index = -1;
        m_encoderVisionPosIndex = -1;
        m_memAttentionSingleFrameOnly = false;
        m_memAttentionUsesObjectPointers = false;
        m_maxMemoryFrames = 7;
        m_maxObjectPointers = 16;
        m_videoFrameIndex = 0;

        m_promptPointCoords.clear();
        m_promptPointLabels.clear();
        m_promptPointCoordsScratch.clear();
        m_promptPointLabelsScratch.clear();
        m_cachedEncoderOutputs.clear();

        m_hasDedicatedPropDecoder = false;
        m_cudaMemoryInfo = Ort::MemoryInfo{nullptr};
        m_useCudaOutputBinding = false;
        m_cudaDeviceId = 0;
        m_device = "cpu";
        resetMemory();
    }
    catch (...) {
        return false;
    }

    return true;
}

void SAM2::resetMemory()
{
    m_hasMemory = false;
    m_memoryStateOutputs.clear();
    m_conditioningMemoryFrames.clear();
    m_recentMemoryFrames.clear();
    m_conditioningObjectPointers.clear();
    m_recentObjectPointers.clear();
    m_temporalCode.clear();
    m_temporalCodeShape.clear();
    m_memoryFeaturesScratch.clear();
    m_memoryFeaturesShapeScratch.clear();
    m_memoryPosScratch.clear();
    m_memoryPosShapeScratch.clear();
    m_objectPointerScratch.clear();
    m_objectPointerShapeScratch.clear();
    m_objectPointerOffsetsScratch.clear();
    m_objectPointerOffsetsShapeScratch.clear();
    m_emptyMemory0Scratch.clear();
    m_videoFrameIndex = 0;
}

SAM2Size SAM2::getInputSize()
{
    if (m_inputShapeEncoder.size() >= 4) {
        return SAM2Size(
            static_cast<int>(m_inputShapeEncoder[3]),
            static_cast<int>(m_inputShapeEncoder[2]));
    }
    return SAM2Size(0, 0);
}

void SAM2::setupSessionOptions(Ort::SessionOptions &options,
                               int threadsNumber,
                               GraphOptimizationLevel optLevel,
                               const std::string &device)
{
    const int safeThreads = (threadsNumber > 0) ? threadsNumber : 1;
    options.SetIntraOpNumThreads(safeThreads);
    options.SetInterOpNumThreads(1);
    options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    options.SetGraphOptimizationLevel(optLevel);

#if defined(__APPLE__)
    options.DisableMemPattern();
#endif

    if (!shouldUseCpuArena(device)) {
        options.DisableCpuMemArena();
    }

    const int cpuArena = shouldUseCpuArena(device) ? 1 : 0;

    if (device == "cpu") {
        ortThrowIf(
            OrtSessionOptionsAppendExecutionProvider_CPU(options, cpuArena),
            "Append CPU EP failed");
        return;
    }

    if (device.rfind("cuda:", 0) == 0) {
#if !defined(__APPLE__)
        int gpuId = 0;
        try {
            gpuId = std::stoi(device.substr(5));
        } catch (...) {
            gpuId = 0;
        }

        OrtCUDAProviderOptions cudaOpts{};
        cudaOpts.device_id = gpuId;
        options.AppendExecutionProvider_CUDA(cudaOpts);
#endif
        ortThrowIf(
            OrtSessionOptionsAppendExecutionProvider_CPU(options, cpuArena),
            "Append CPU EP (fallback) failed");
        return;
    }

    if (device.rfind("coreml", 0) == 0) {
#ifdef __APPLE__
        constexpr uint32_t coremlFlags = 0;
        ortThrowIf(
            OrtSessionOptionsAppendExecutionProvider_CoreML(options, coremlFlags),
            "Append CoreML EP failed");
#else
        ortThrowIf(
            OrtSessionOptionsAppendExecutionProvider_CPU(options, cpuArena),
            "Append CPU EP (fallback) failed");
#endif
        return;
    }

    ortThrowIf(
        OrtSessionOptionsAppendExecutionProvider_CPU(options, cpuArena),
        "Append default CPU EP failed");
}

std::vector<SAM2Node> SAM2::getSessionNodes(Ort::Session* session, bool isInput)
{
    return getSessionNodesInternal(session, isInput, true);
}

std::vector<const char*> SAM2::collectNodeNames(const std::vector<SAM2Node> &nodes)
{
    std::vector<const char*> names;
    names.reserve(nodes.size());
    for (const auto &node : nodes) {
        names.push_back(node.name.c_str());
    }
    return names;
}

std::vector<const char*> SAM2::selectNodeNames(const std::vector<SAM2Node> &nodes,
                                               std::initializer_list<const char*> preferredKeys)
{
    std::vector<const char*> selected;
    std::set<std::string> seen;

    for (const char* key : preferredKeys) {
        const std::string loweredKey = lowerCopy(key);
        for (const auto &node : nodes) {
            const std::string loweredName = lowerCopy(node.name);
            if (loweredName.find(loweredKey) != std::string::npos && seen.insert(node.name).second) {
                selected.push_back(node.name.c_str());
                break;
            }
        }
    }

    return selected.empty() ? collectNodeNames(nodes) : selected;
}

int SAM2::findNodeIndex(const std::vector<SAM2Node> &nodes, const std::string &key)
{
    const std::string loweredKey = lowerCopy(key);
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (lowerCopy(nodes[i].name).find(loweredKey) != std::string::npos) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

int SAM2::findNameIndex(const std::vector<const char*> &names, const std::string &key)
{
    const std::string loweredKey = lowerCopy(key);
    for (size_t i = 0; i < names.size(); ++i) {
        const char* name = names[i];
        if (!name) {
            continue;
        }
        if (lowerCopy(name).find(loweredKey) != std::string::npos) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

Ort::MemoryInfo SAM2::cloneMemoryInfo(const Ort::ConstMemoryInfo &memoryInfo)
{
    const std::string allocatorName = memoryInfo.GetAllocatorName();
    return Ort::MemoryInfo(
        allocatorName.c_str(),
        memoryInfo.GetAllocatorType(),
        memoryInfo.GetDeviceId(),
        memoryInfo.GetMemoryType());
}

Ort::MemoryInfo SAM2::cloneTensorMemoryInfo(const Ort::Value &tensor)
{
    return cloneMemoryInfo(tensor.GetTensorMemoryInfo());
}

bool SAM2::isStaticOptimizableModel(const std::string &modelPath)
{
    const std::string lowered = lowerCopy(modelPath);
    return lowered.find("image_decoder_points.onnx") != std::string::npos
        || lowered.find("image_decoder_box.onnx") != std::string::npos
        || lowered.find("video_decoder_init.onnx") != std::string::npos
        || lowered.find("video_decoder_propagate.onnx") != std::string::npos
        || lowered.find("memory_attention_no_objptr_1frame.onnx") != std::string::npos;
}

bool SAM2::shouldUseCpuArena(const std::string &device)
{
#if defined(__APPLE__)
    return device != "cpu";
#else
    (void)device;
    return true;
#endif
}

bool SAM2::initializeNamedSession(std::unique_ptr<Ort::Session> *sessionOut,
                                  const Ort::Env &env,
                                  const std::string &modelPath,
                                  const Ort::SessionOptions &options,
                                  std::vector<SAM2Node> *inputNodes,
                                  std::vector<SAM2Node> *outputNodes,
                                  std::vector<const char*> *inputNames,
                                  std::vector<const char*> *outputNames,
                                  bool skipShapeMetadata)
{
    try {
        std::unique_ptr<Ort::Session> session;

#ifdef _WIN32
        const std::wstring widePath = strToWstr(modelPath);
        session = std::make_unique<Ort::Session>(env, widePath.c_str(), options);
#else
        session = std::make_unique<Ort::Session>(env, modelPath.c_str(), options);
#endif

        if (inputNodes) {
            *inputNodes = getSessionNodesInternal(session.get(), true, !skipShapeMetadata);
        }
        if (outputNodes) {
            *outputNodes = getSessionNodesInternal(session.get(), false, !skipShapeMetadata);
        }
        if (inputNames && inputNodes) {
            *inputNames = collectNodeNames(*inputNodes);
        }
        if (outputNames && outputNodes) {
            *outputNames = collectNodeNames(*outputNodes);
        }

        *sessionOut = std::move(session);
        return true;
    }
    catch (const std::exception &e) {
        std::cerr << "[ERROR] Failed to load session " << modelPath << " => " << e.what() << '\n';
        return false;
    }
}

Ort::Session* SAM2::getPropDecoderSession()
{
    return m_hasDedicatedPropDecoder ? m_videoPropDecoderSession.get() : m_imgDecoderSession.get();
}

const std::vector<SAM2Node>& SAM2::getPropDecoderInputNodes() const
{
    return m_hasDedicatedPropDecoder ? m_videoPropDecoderInputNodes : m_imgDecoderInputNodes;
}

const std::vector<const char*>& SAM2::getPropDecoderInputNames() const
{
    return m_hasDedicatedPropDecoder ? m_videoPropDecoderInputNames : m_imgDecoderInputNames;
}

const std::vector<const char*>& SAM2::getPropDecoderOutputNames() const
{
    return m_hasDedicatedPropDecoder ? m_videoPropDecoderOutputNames : m_imgDecoderOutputNames;
}

const std::vector<const char*>& SAM2::getPropDecoderVideoOutputNames() const
{
    return m_hasDedicatedPropDecoder ? m_videoPropDecoderVideoOutputNames : m_imgDecoderVideoOutputNames;
}

bool SAM2::initialize(const std::string &encoderPath,
                      const std::string &decoderPath,
                      int threadsNumber,
                      std::string device)
{
    clearSessions();
    m_device = device;
    m_useCudaOutputBinding = false;
    m_cudaDeviceId = 0;
    m_cudaMemoryInfo = Ort::MemoryInfo{nullptr};

#if !defined(__APPLE__)
    if (device.rfind("cuda:", 0) == 0) {
        try {
            m_cudaDeviceId = std::stoi(device.substr(5));
        } catch (...) {
            m_cudaDeviceId = 0;
        }
        m_cudaMemoryInfo = Ort::MemoryInfo("Cuda", OrtDeviceAllocator, m_cudaDeviceId, OrtMemTypeDefault);
        m_useCudaOutputBinding = true;
    }
#endif

    if (!modelExists(encoderPath) || !modelExists(decoderPath)) {
        std::cerr << "[ERROR] Model file not found.\n";
        return false;
    }

    const bool macCpu =
#if defined(__APPLE__)
        device.rfind("cuda:", 0) != 0 && device.rfind("coreml", 0) != 0;
#else
        false;
#endif

    Ort::SessionOptions encoderOptions;
    Ort::SessionOptions decoderOptions;

    setupSessionOptions(
        encoderOptions,
        threadsNumber,
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED,
        device);
    setupSessionOptions(
        decoderOptions,
        threadsNumber,
        isStaticOptimizableModel(decoderPath)
            ? GraphOptimizationLevel::ORT_ENABLE_EXTENDED
            : GraphOptimizationLevel::ORT_DISABLE_ALL,
        device);
    decoderOptions.AddConfigEntry("session.disable_gemm_fast_gelu_fusion", "1");

#if defined(__APPLE__)
    if (macCpu) {
        encoderOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
        encoderOptions.AddConfigEntry("session.disable_gemm_fast_gelu_fusion", "1");
        encoderOptions.AddConfigEntry("session.disable_prepacking", "1");

        decoderOptions.AddConfigEntry("session.disable_prepacking", "1");
    }
#endif

    if (!initializeNamedSession(
            &m_imgEncoderSession,
            m_encoderEnv,
            encoderPath,
            encoderOptions,
            &m_imgEncoderInputNodes,
            &m_imgEncoderOutputNodes,
            &m_imgEncoderInputNames,
            &m_imgEncoderOutputNames,
            macCpu)) {
        return false;
    }

    if (!initializeNamedSession(
            &m_imgDecoderSession,
            m_decoderEnv,
            decoderPath,
            decoderOptions,
            &m_imgDecoderInputNodes,
            &m_imgDecoderOutputNodes,
            &m_imgDecoderInputNames,
            &m_imgDecoderOutputNames,
            macCpu)) {
        return false;
    }

    if (macCpu) {
        m_inputShapeEncoder = {1, 3, 1024, 1024};
    } else if (!m_imgEncoderInputNodes.empty()) {
        m_inputShapeEncoder = m_imgEncoderInputNodes.front().dim;
    }

    if (m_inputShapeEncoder.size() < 4) {
        std::cerr << "[ERROR] Could not determine encoder input shape.\n";
        return false;
    }

    m_encoderEmbedIndex = findNodeIndex(m_imgEncoderOutputNodes, "image_embeddings");
    if (m_encoderEmbedIndex < 0) {
        m_encoderEmbedIndex = 0;
    }

    m_encoderCurrentVisionFeatIndex = findNodeIndex(m_imgEncoderOutputNodes, "current_vision_feat");
    if (m_encoderCurrentVisionFeatIndex < 0) {
        m_encoderCurrentVisionFeatIndex = m_encoderEmbedIndex;
    }

    m_encoderHighRes0Index = findNodeIndex(m_imgEncoderOutputNodes, "high_res_features1");
    if (m_encoderHighRes0Index < 0) {
        m_encoderHighRes0Index = findNodeIndex(m_imgEncoderOutputNodes, "high_res_feats_0");
    }
    if (m_encoderHighRes0Index < 0) {
        m_encoderHighRes0Index = 1;
    }

    m_encoderHighRes1Index = findNodeIndex(m_imgEncoderOutputNodes, "high_res_features2");
    if (m_encoderHighRes1Index < 0) {
        m_encoderHighRes1Index = findNodeIndex(m_imgEncoderOutputNodes, "high_res_feats_1");
    }
    if (m_encoderHighRes1Index < 0) {
        m_encoderHighRes1Index = 2;
    }

    m_encoderVisionPosIndex = findNodeIndex(m_imgEncoderOutputNodes, "vision_pos_embed");

    if (m_encoderEmbedIndex >= static_cast<int>(m_imgEncoderOutputNodes.size())
        || m_encoderCurrentVisionFeatIndex >= static_cast<int>(m_imgEncoderOutputNodes.size())
        || m_encoderHighRes0Index >= static_cast<int>(m_imgEncoderOutputNodes.size())
        || m_encoderHighRes1Index >= static_cast<int>(m_imgEncoderOutputNodes.size())) {
        std::cerr << "[ERROR] Encoder outputs do not expose the expected tensors.\n";
        return false;
    }

    if (findNodeIndex(m_imgDecoderOutputNodes, "pred_mask") < 0) {
        std::cerr << "[ERROR] Decoder is missing pred_mask output.\n";
        return false;
    }

    m_imgDecoderImageOutputNames = selectNodeNames(m_imgDecoderOutputNodes, {"pred_mask"});
    m_imgDecoderVideoOutputNames = selectNodeNames(m_imgDecoderOutputNodes, {"obj_ptr", "mask_for_mem", "pred_mask"});

    return true;
}

bool SAM2::initializeVideo(const std::string &encoderPath,
                           const std::string &decoderPath,
                           const std::string &memAttentionPath,
                           const std::string &memEncoderPath,
                           int threadsNumber,
                           std::string device)
{
    return initializeVideo(
        encoderPath,
        decoderPath,
        decoderPath,
        memAttentionPath,
        memEncoderPath,
        threadsNumber,
        std::move(device));
}

bool SAM2::initializeVideo(const std::string &encoderPath,
                           const std::string &decoderInitPath,
                           const std::string &decoderPropPath,
                           const std::string &memAttentionPath,
                           const std::string &memEncoderPath,
                           int threadsNumber,
                           std::string device)
{
    if (!initialize(encoderPath, decoderInitPath, threadsNumber, device)) {
        std::cerr << "[ERROR] initializeVideo => base init failed.\n";
        return false;
    }

    m_maxMemoryFrames = preferredVideoMemoryFrameLimit();
    m_maxObjectPointers = preferredVideoObjectPointerLimit();

    if (!modelExists(memAttentionPath) || !modelExists(memEncoderPath)) {
        std::cerr << "[ERROR] Memory model files not found.\n";
        return false;
    }

    if (findNodeIndex(m_imgDecoderOutputNodes, "mask_for_mem") < 0
        || findNodeIndex(m_imgDecoderOutputNodes, "pred_mask") < 0) {
        std::cerr << "[ERROR] Initial video decoder must expose mask_for_mem and pred_mask.\n";
        return false;
    }

    const bool macCpu =
#if defined(__APPLE__)
        device.rfind("cuda:", 0) != 0 && device.rfind("coreml", 0) != 0;
#else
        false;
#endif

    if (!decoderPropPath.empty() && decoderPropPath != decoderInitPath) {
        if (!modelExists(decoderPropPath)) {
            std::cerr << "[ERROR] Propagation decoder file not found.\n";
            return false;
        }

        Ort::SessionOptions propDecoderOptions;
        setupSessionOptions(
            propDecoderOptions,
            threadsNumber,
            isStaticOptimizableModel(decoderPropPath)
                ? GraphOptimizationLevel::ORT_ENABLE_EXTENDED
                : GraphOptimizationLevel::ORT_DISABLE_ALL,
            device);
        propDecoderOptions.AddConfigEntry("session.disable_gemm_fast_gelu_fusion", "1");

#if defined(__APPLE__)
        if (macCpu) {
            propDecoderOptions.AddConfigEntry("session.disable_prepacking", "1");
        }
#endif

        if (!initializeNamedSession(
                &m_videoPropDecoderSession,
                m_videoPropDecoderEnv,
                decoderPropPath,
                propDecoderOptions,
                &m_videoPropDecoderInputNodes,
                &m_videoPropDecoderOutputNodes,
                &m_videoPropDecoderInputNames,
                &m_videoPropDecoderOutputNames,
                macCpu)) {
            return false;
        }

        if (findNodeIndex(m_videoPropDecoderOutputNodes, "mask_for_mem") < 0
            || findNodeIndex(m_videoPropDecoderOutputNodes, "pred_mask") < 0) {
            std::cerr << "[ERROR] Propagation decoder must expose mask_for_mem and pred_mask.\n";
            return false;
        }

        m_videoPropDecoderVideoOutputNames =
            selectNodeNames(m_videoPropDecoderOutputNodes, {"obj_ptr", "mask_for_mem", "pred_mask"});
        m_hasDedicatedPropDecoder = true;
    } else {
        m_hasDedicatedPropDecoder = false;
        m_videoPropDecoderSession.reset();
        m_videoPropDecoderInputNodes.clear();
        m_videoPropDecoderOutputNodes.clear();
        m_videoPropDecoderInputNames.clear();
        m_videoPropDecoderOutputNames.clear();
        m_videoPropDecoderVideoOutputNames.clear();
    }

    Ort::SessionOptions memAttentionOptions;
    Ort::SessionOptions memEncoderOptions;

    setupSessionOptions(
        memAttentionOptions,
        threadsNumber,
        isStaticOptimizableModel(memAttentionPath)
            ? GraphOptimizationLevel::ORT_ENABLE_EXTENDED
            : GraphOptimizationLevel::ORT_DISABLE_ALL,
        device);
    setupSessionOptions(
        memEncoderOptions,
        threadsNumber,
        GraphOptimizationLevel::ORT_DISABLE_ALL,
        device);

    memAttentionOptions.AddConfigEntry("session.disable_gemm_fast_gelu_fusion", "1");
    memEncoderOptions.AddConfigEntry("session.disable_gemm_fast_gelu_fusion", "1");

#if defined(__APPLE__)
    if (macCpu) {
        memAttentionOptions.AddConfigEntry("session.disable_prepacking", "1");
        memEncoderOptions.AddConfigEntry("session.disable_prepacking", "1");
    }
#endif

    if (!initializeNamedSession(
            &m_memAttentionSession,
            m_memAttentionEnv,
            memAttentionPath,
            memAttentionOptions,
            &m_memAttentionInputNodes,
            &m_memAttentionOutputNodes,
            &m_memAttentionInputNames,
            &m_memAttentionOutputNames,
            macCpu)) {
        return false;
    }

    if (!initializeNamedSession(
            &m_memEncoderSession,
            m_memEncoderEnv,
            memEncoderPath,
            memEncoderOptions,
            &m_memEncoderInputNodes,
            &m_memEncoderOutputNodes,
            &m_memEncoderInputNames,
            &m_memEncoderOutputNames,
            macCpu)) {
        return false;
    }

    if (m_memAttentionOutputNodes.empty()
        || findNodeIndex(m_memAttentionOutputNodes, "fused_feat") < 0) {
        std::cerr << "[ERROR] Memory attention session has no outputs.\n";
        return false;
    }

    if (findNodeIndex(m_memEncoderOutputNodes, "maskmem_features") < 0
        || findNodeIndex(m_memEncoderOutputNodes, "maskmem_pos_enc") < 0) {
        std::cerr << "[ERROR] Memory encoder is missing required outputs.\n";
        return false;
    }

    m_memEncoderStateOutputNames =
        selectNodeNames(m_memEncoderOutputNodes, {"maskmem_features", "maskmem_pos_enc", "temporal_code"});

    const int memory1InputIndex = findNodeIndex(m_memAttentionInputNodes, "memory_1");
    if (memory1InputIndex >= 0) {
        const auto &dims = m_memAttentionInputNodes[static_cast<size_t>(memory1InputIndex)].dim;
        if (!dims.empty() && dims[0] == 1) {
            m_memAttentionSingleFrameOnly = true;
        }
    }

    const int temporalCodeIndex = findNodeIndex(m_memEncoderOutputNodes, "temporal_code");
    if (temporalCodeIndex >= 0) {
        const auto &dims = m_memEncoderOutputNodes[static_cast<size_t>(temporalCodeIndex)].dim;
        if (!dims.empty() && dims[0] > 0) {
            m_maxMemoryFrames = std::min(m_maxMemoryFrames, static_cast<size_t>(dims[0]));
        }
    }

    if (findNodeIndex(m_memAttentionInputNodes, "obj_ptr_offsets") >= 0) {
        m_memAttentionUsesObjectPointers = true;
    }

    return true;
}

MemoryFrameState SAM2::captureMemoryFrameState(const std::vector<Ort::Value> &memoryOutputs)
{
    if (memoryOutputs.size() < 2) {
        throw std::runtime_error("Memory encoder did not return enough tensors to build a memory frame.");
    }

    MemoryFrameState frame;
    extractTensorData(memoryOutputs[0], frame.features, frame.featuresShape);
    extractTensorData(memoryOutputs[1], frame.pos, frame.posShape);
    return frame;
}

void SAM2::updateTemporalCode(const std::vector<Ort::Value> &memoryOutputs)
{
    if (memoryOutputs.size() < 3) {
        return;
    }

    extractTensorData(memoryOutputs[2], m_temporalCode, m_temporalCodeShape);
    if (!m_temporalCodeShape.empty() && m_temporalCodeShape[0] > 0) {
        m_maxMemoryFrames = std::min(m_maxMemoryFrames, static_cast<size_t>(m_temporalCodeShape[0]));
    }
}

void SAM2::trimRecentMemoryFrames()
{
    size_t maxRecent = 0;
    if (m_memAttentionSingleFrameOnly) {
        maxRecent = 1;
    } else if (m_maxMemoryFrames > m_conditioningMemoryFrames.size()) {
        maxRecent = m_maxMemoryFrames - m_conditioningMemoryFrames.size();
    }

    if (m_recentMemoryFrames.size() > maxRecent) {
        m_recentMemoryFrames.resize(maxRecent);
    }
}

void SAM2::trimRecentObjectPointers()
{
    size_t maxRecent = 0;
    if (m_maxObjectPointers > m_conditioningObjectPointers.size()) {
        maxRecent = m_maxObjectPointers - m_conditioningObjectPointers.size();
    }

    if (m_recentObjectPointers.size() > maxRecent) {
        m_recentObjectPointers.resize(maxRecent);
    }
}

void SAM2::storeConditioningMemory(const std::vector<Ort::Value> &memoryOutputs)
{
    updateTemporalCode(memoryOutputs);
    m_conditioningMemoryFrames.clear();
    m_conditioningMemoryFrames.push_back(captureMemoryFrameState(memoryOutputs));
    trimRecentMemoryFrames();
    m_hasMemory = !m_conditioningMemoryFrames.empty() || !m_recentMemoryFrames.empty();
}

void SAM2::appendRecentMemory(const std::vector<Ort::Value> &memoryOutputs)
{
    updateTemporalCode(memoryOutputs);
    m_recentMemoryFrames.insert(m_recentMemoryFrames.begin(), captureMemoryFrameState(memoryOutputs));
    trimRecentMemoryFrames();
    m_hasMemory = !m_conditioningMemoryFrames.empty() || !m_recentMemoryFrames.empty();
}

void SAM2::storeConditioningObjectPointer(const std::vector<Ort::Value> &decoderOutputs,
                                          const std::vector<const char*> &outputNames,
                                          int frameIndex)
{
    if (!m_memAttentionUsesObjectPointers) {
        return;
    }

    for (size_t i = 0; i < outputNames.size() && i < decoderOutputs.size(); ++i) {
        if (lowerCopy(outputNames[i]).find("obj_ptr") == std::string::npos) {
            continue;
        }

        ObjectPointerFrameState state;
        std::vector<int64_t> shape;
        extractTensorData(decoderOutputs[i], state.value, shape);
        state.frameIndex = frameIndex;
        m_conditioningObjectPointers.clear();
        m_conditioningObjectPointers.push_back(std::move(state));
        trimRecentObjectPointers();
        return;
    }
}

void SAM2::appendRecentObjectPointer(const std::vector<Ort::Value> &decoderOutputs,
                                     const std::vector<const char*> &outputNames,
                                     int frameIndex)
{
    if (!m_memAttentionUsesObjectPointers) {
        return;
    }

    for (size_t i = 0; i < outputNames.size() && i < decoderOutputs.size(); ++i) {
        if (lowerCopy(outputNames[i]).find("obj_ptr") == std::string::npos) {
            continue;
        }

        ObjectPointerFrameState state;
        std::vector<int64_t> shape;
        extractTensorData(decoderOutputs[i], state.value, shape);
        state.frameIndex = frameIndex;
        m_recentObjectPointers.insert(m_recentObjectPointers.begin(), std::move(state));
        trimRecentObjectPointers();
        return;
    }
}

std::vector<Ort::Value> SAM2::buildDecoderInputs(const std::vector<SAM2Node> &inputNodes,
                                                 Ort::Value &primaryFeature,
                                                 Ort::Value &highResFeatures0,
                                                 Ort::Value &highResFeatures1,
                                                 const std::vector<float> *promptCoords,
                                                 const std::vector<float> *promptLabels)
{
    const std::vector<float>* coords = promptCoords;
    const std::vector<float>* labels = promptLabels;
    m_promptPointCoordsScratch.clear();
    m_promptPointLabelsScratch.clear();

    const int pointCoordsIndex = findNodeIndex(inputNodes, "point_coords");
    const int pointLabelsIndex = findNodeIndex(inputNodes, "point_labels");
    if (pointCoordsIndex >= 0 || pointLabelsIndex >= 0) {
        coords = coords ? coords : &m_promptPointCoordsScratch;
        labels = labels ? labels : &m_promptPointLabelsScratch;

        int availablePoints = static_cast<int>(labels->size());
        if (coords->size() / 2 < static_cast<size_t>(availablePoints)) {
            availablePoints = static_cast<int>(coords->size() / 2);
        }

        int fixedPointSlots = -1;
        if (pointCoordsIndex >= 0) {
            const auto &dims = inputNodes[pointCoordsIndex].dim;
            if (dims.size() >= 2 && dims[1] > 0) {
                fixedPointSlots = static_cast<int>(dims[1]);
            }
        }

        if (fixedPointSlots > 0) {
            const int usedPoints = std::min(availablePoints, fixedPointSlots);
            m_promptPointCoordsScratch.assign(static_cast<size_t>(fixedPointSlots) * 2, 0.0f);
            m_promptPointLabelsScratch.assign(static_cast<size_t>(fixedPointSlots), -1.0f);

            if (availablePoints > fixedPointSlots) {
                std::cerr << "[WARN] Truncating prompts from " << availablePoints
                          << " to " << fixedPointSlots << " for fixed-shape decoder.\n";
            }

            for (int i = 0; i < usedPoints; ++i) {
                m_promptPointCoordsScratch[static_cast<size_t>(i) * 2 + 0] = (*coords)[static_cast<size_t>(i) * 2 + 0];
                m_promptPointCoordsScratch[static_cast<size_t>(i) * 2 + 1] = (*coords)[static_cast<size_t>(i) * 2 + 1];
                m_promptPointLabelsScratch[static_cast<size_t>(i)] = (*labels)[static_cast<size_t>(i)];
            }
            coords = &m_promptPointCoordsScratch;
            labels = &m_promptPointLabelsScratch;
        } else if (availablePoints < static_cast<int>(labels->size()) || availablePoints * 2 < static_cast<int>(coords->size())) {
            m_promptPointCoordsScratch.assign(
                coords->begin(),
                coords->begin() + static_cast<ptrdiff_t>(availablePoints) * 2);
            m_promptPointLabelsScratch.assign(labels->begin(), labels->begin() + availablePoints);
            coords = &m_promptPointCoordsScratch;
            labels = &m_promptPointLabelsScratch;
        }
    }

    std::vector<Ort::Value> inputs;
    inputs.reserve(inputNodes.size());

    for (const auto &node : inputNodes) {
        const std::string lowered = lowerCopy(node.name);

        if (lowered.find("point_coords") != std::string::npos) {
            const int64_t pointCount = labels ? static_cast<int64_t>(labels->size()) : 0;
            const std::vector<int64_t> shape = {1, pointCount, 2};
            inputs.push_back(createTensor<float>(m_memoryInfo, coords ? *coords : m_promptPointCoordsScratch, shape));
            continue;
        }

        if (lowered.find("point_labels") != std::string::npos) {
            const int64_t pointCount = labels ? static_cast<int64_t>(labels->size()) : 0;
            const std::vector<int64_t> shape = {1, pointCount};
            inputs.push_back(createTensor<float>(m_memoryInfo, labels ? *labels : m_promptPointLabelsScratch, shape));
            continue;
        }

        if (lowered.find("image_embed") != std::string::npos) {
            const Ort::MemoryInfo featureMemoryInfo = cloneTensorMemoryInfo(primaryFeature);
            inputs.push_back(createTensorView<float>(featureMemoryInfo, primaryFeature));
            continue;
        }

        if (lowered.find("high_res_feats_0") != std::string::npos
            || lowered.find("high_res_features1") != std::string::npos) {
            const Ort::MemoryInfo featureMemoryInfo = cloneTensorMemoryInfo(highResFeatures0);
            inputs.push_back(createTensorView<float>(featureMemoryInfo, highResFeatures0));
            continue;
        }

        if (lowered.find("high_res_feats_1") != std::string::npos
            || lowered.find("high_res_features2") != std::string::npos) {
            const Ort::MemoryInfo featureMemoryInfo = cloneTensorMemoryInfo(highResFeatures1);
            inputs.push_back(createTensorView<float>(featureMemoryInfo, highResFeatures1));
            continue;
        }

        std::ostringstream oss;
        oss << "Unsupported decoder input node: " << node.name;
        throw std::runtime_error(oss.str());
    }

    return inputs;
}

std::vector<Ort::Value> SAM2::buildMemAttentionInputs(Ort::Value &currentVisionFeat,
                                                      Ort::Value *currentVisionPosEmbed,
                                                      const std::vector<float> &emptyMemory0)
{
    if (m_conditioningMemoryFrames.empty() && m_recentMemoryFrames.empty()) {
        throw std::runtime_error("Memory attention requested before memory encoder produced state.");
    }

    std::vector<const MemoryFrameState*> selectedFrames;
    std::vector<size_t> selectedTPos;
    if (m_memAttentionSingleFrameOnly) {
        if (!m_recentMemoryFrames.empty()) {
            selectedFrames.push_back(&m_recentMemoryFrames.front());
            selectedTPos.push_back(1);
        } else if (!m_conditioningMemoryFrames.empty()) {
            selectedFrames.push_back(&m_conditioningMemoryFrames.front());
            selectedTPos.push_back(0);
        }
    } else {
        for (const auto &frame : m_conditioningMemoryFrames) {
            selectedFrames.push_back(&frame);
            selectedTPos.push_back(0);
        }

        size_t maxRecent = 0;
        if (m_maxMemoryFrames > selectedFrames.size()) {
            maxRecent = m_maxMemoryFrames - selectedFrames.size();
        }

        const size_t recentCount = std::min(maxRecent, m_recentMemoryFrames.size());
        for (size_t i = 0; i < recentCount; ++i) {
            selectedFrames.push_back(&m_recentMemoryFrames[i]);
            selectedTPos.push_back(i + 1);
        }
    }

    if (selectedFrames.empty()) {
        throw std::runtime_error("Memory attention has no selected memory frames.");
    }

    const auto &featureShape = selectedFrames.front()->featuresShape;
    const auto &posShape = selectedFrames.front()->posShape;
    if (featureShape.size() < 4 || posShape.size() < 3) {
        throw std::runtime_error("Memory attention state has unexpected tensor shapes.");
    }

    m_memoryFeaturesShapeScratch = featureShape;
    m_memoryFeaturesShapeScratch[0] = static_cast<int64_t>(selectedFrames.size());
    m_memoryFeaturesScratch.clear();
    m_memoryFeaturesScratch.reserve(selectedFrames.size() * selectedFrames.front()->features.size());
    for (const auto *frame : selectedFrames) {
        if (frame->featuresShape != featureShape) {
            throw std::runtime_error("Memory frame feature shapes do not match.");
        }
        m_memoryFeaturesScratch.insert(
            m_memoryFeaturesScratch.end(),
            frame->features.begin(),
            frame->features.end());
    }

    m_memoryPosShapeScratch = posShape;
    m_memoryPosShapeScratch[0] = posShape[0] * static_cast<int64_t>(selectedFrames.size());
    m_memoryPosScratch.clear();
    m_memoryPosScratch.reserve(selectedFrames.size() * selectedFrames.front()->pos.size());

    size_t temporalCapacity = 0;
    size_t temporalSliceSize = 0;
    if (!m_temporalCode.empty() && !m_temporalCodeShape.empty() && m_temporalCodeShape[0] > 0) {
        temporalCapacity = static_cast<size_t>(m_temporalCodeShape[0]);
        if (m_temporalCodeShape.size() > 1) {
            std::vector<int64_t> temporalSliceShape(
                m_temporalCodeShape.begin() + 1,
                m_temporalCodeShape.end());
            temporalSliceSize = computeElementCount(temporalSliceShape);
        }
    }

    const size_t posChannels = static_cast<size_t>(posShape.back());
    for (size_t i = 0; i < selectedFrames.size(); ++i) {
        const auto &frame = *selectedFrames[i];
        if (frame.posShape != posShape) {
            throw std::runtime_error("Memory frame position shapes do not match.");
        }

        const bool useTemporal =
            temporalCapacity > selectedTPos[i]
            && temporalSliceSize == posChannels;
        const float* temporalOffset = nullptr;
        if (useTemporal) {
            const size_t temporalIndex = temporalCapacity - selectedTPos[i] - 1;
            temporalOffset = m_temporalCode.data() + temporalIndex * temporalSliceSize;
        }

        const size_t tokenCount = frame.pos.size() / posChannels;
        if (!useTemporal) {
            m_memoryPosScratch.insert(
                m_memoryPosScratch.end(),
                frame.pos.begin(),
                frame.pos.end());
            continue;
        }

        const size_t start = m_memoryPosScratch.size();
        m_memoryPosScratch.resize(start + frame.pos.size());
        for (size_t token = 0; token < tokenCount; ++token) {
            const size_t tokenOffset = token * posChannels;
            for (size_t channel = 0; channel < posChannels; ++channel) {
                m_memoryPosScratch[start + tokenOffset + channel] =
                    frame.pos[tokenOffset + channel] + temporalOffset[channel];
            }
        }
    }

    m_objectPointerScratch.clear();
    m_objectPointerShapeScratch = {0, 256};
    m_objectPointerOffsetsScratch.clear();
    m_objectPointerOffsetsShapeScratch = {0};

    if (m_memAttentionUsesObjectPointers) {
        std::vector<const ObjectPointerFrameState*> selectedPointers;
        for (const auto &pointer : m_conditioningObjectPointers) {
            selectedPointers.push_back(&pointer);
        }

        size_t maxRecentPointers = 0;
        if (m_maxObjectPointers > selectedPointers.size()) {
            maxRecentPointers = m_maxObjectPointers - selectedPointers.size();
        }

        const size_t recentCount = std::min(maxRecentPointers, m_recentObjectPointers.size());
        for (size_t i = 0; i < recentCount; ++i) {
            selectedPointers.push_back(&m_recentObjectPointers[i]);
        }

        m_objectPointerShapeScratch[0] = static_cast<int64_t>(selectedPointers.size());
        m_objectPointerOffsetsShapeScratch[0] = static_cast<int64_t>(selectedPointers.size());
        m_objectPointerScratch.reserve(selectedPointers.size() * 256);
        m_objectPointerOffsetsScratch.reserve(selectedPointers.size());

        for (const auto *pointer : selectedPointers) {
            if (pointer->value.size() != 256) {
                throw std::runtime_error("Object pointer tensor does not have the expected size of 256.");
            }
            m_objectPointerScratch.insert(
                m_objectPointerScratch.end(),
                pointer->value.begin(),
                pointer->value.end());
            m_objectPointerOffsetsScratch.push_back(
                static_cast<float>(std::max(0, m_videoFrameIndex - pointer->frameIndex)));
        }
    }

    std::vector<Ort::Value> inputs;
    inputs.reserve(m_memAttentionInputNodes.size());

    for (const auto &node : m_memAttentionInputNodes) {
        const std::string lowered = lowerCopy(node.name);

        if (lowered.find("current_vision_feat") != std::string::npos) {
            const Ort::MemoryInfo featureMemoryInfo = cloneTensorMemoryInfo(currentVisionFeat);
            inputs.push_back(createTensorView<float>(featureMemoryInfo, currentVisionFeat));
            continue;
        }

        if (lowered.find("current_vision_pos_embed") != std::string::npos) {
            if (!currentVisionPosEmbed) {
                throw std::runtime_error("Memory attention requires current_vision_pos_embed, but encoder did not provide it.");
            }
            const Ort::MemoryInfo featureMemoryInfo = cloneTensorMemoryInfo(*currentVisionPosEmbed);
            inputs.push_back(createTensorView<float>(featureMemoryInfo, *currentVisionPosEmbed));
            continue;
        }

        if (lowered.find("memory_0") != std::string::npos) {
            if (m_memAttentionUsesObjectPointers) {
                inputs.push_back(createTensor<float>(m_memoryInfo, m_objectPointerScratch, m_objectPointerShapeScratch));
            } else {
                const std::vector<int64_t> shape = {0, 256};
                m_emptyMemory0Scratch = emptyMemory0;
                inputs.push_back(createTensor<float>(m_memoryInfo, m_emptyMemory0Scratch, shape));
            }
            continue;
        }

        if (lowered.find("obj_ptr_offsets") != std::string::npos) {
            inputs.push_back(createTensor<float>(m_memoryInfo, m_objectPointerOffsetsScratch, m_objectPointerOffsetsShapeScratch));
            continue;
        }

        if (lowered.find("memory_1") != std::string::npos) {
            inputs.push_back(createTensor<float>(m_memoryInfo, m_memoryFeaturesScratch, m_memoryFeaturesShapeScratch));
            continue;
        }

        if (lowered.find("memory_pos_embed") != std::string::npos) {
            inputs.push_back(createTensor<float>(m_memoryInfo, m_memoryPosScratch, m_memoryPosShapeScratch));
            continue;
        }

        std::ostringstream oss;
        oss << "Unsupported memory-attention input node: " << node.name;
        throw std::runtime_error(oss.str());
    }

    return inputs;
}

std::vector<Ort::Value> SAM2::buildMemEncoderInputs(Ort::Value &maskForMem,
                                                    Ort::Value &pixFeat)
{
    std::vector<Ort::Value> inputs;
    inputs.reserve(m_memEncoderInputNodes.size());

    for (const auto &node : m_memEncoderInputNodes) {
        const std::string lowered = lowerCopy(node.name);

        if (lowered.find("mask_for_mem") != std::string::npos) {
            const Ort::MemoryInfo maskMemoryInfo = cloneTensorMemoryInfo(maskForMem);
            auto maskShape = maskForMem.GetTensorTypeAndShapeInfo().GetShape();
            if (maskShape.size() >= 4 && maskShape[1] > 1) {
                maskShape[1] = 1;
                inputs.push_back(
                    createTensorView<float>(
                        maskMemoryInfo,
                        maskForMem.GetTensorMutableData<float>(),
                        maskShape));
            } else {
                inputs.push_back(createTensorView<float>(maskMemoryInfo, maskForMem));
            }
            continue;
        }

        if (lowered.find("pix_feat") != std::string::npos) {
            const Ort::MemoryInfo pixFeatMemoryInfo = cloneTensorMemoryInfo(pixFeat);
            inputs.push_back(createTensorView<float>(pixFeatMemoryInfo, pixFeat));
            continue;
        }

        std::ostringstream oss;
        oss << "Unsupported memory-encoder input node: " << node.name;
        throw std::runtime_error(oss.str());
    }

    return inputs;
}

std::variant<std::vector<Ort::Value>, std::string>
SAM2::runSession(Ort::Session* session,
                 const std::vector<const char*> &inputNames,
                 const std::vector<const char*> &outputNames,
                 const std::vector<Ort::Value> &inputTensors,
                 const std::string &debugName)
{
    if (!session) {
        return std::string("[ERROR] runSession(" + debugName + "): session is null.");
    }

    try {
        auto outputs = session->Run(
            Ort::RunOptions{nullptr},
            inputNames.data(),
            const_cast<Ort::Value*>(inputTensors.data()),
            inputTensors.size(),
            outputNames.data(),
            outputNames.size());
        return outputs;
    }
    catch (const std::exception &e) {
        std::ostringstream oss;
        oss << "[ERROR] runSession(" << debugName << ") => " << e.what();
        return oss.str();
    }
}

std::variant<std::vector<Ort::Value>, std::string>
SAM2::runSessionWithOutputMemory(Ort::Session* session,
                                 const std::vector<const char*> &inputNames,
                                 const std::vector<const char*> &outputNames,
                                 const std::vector<Ort::Value> &inputTensors,
                                 const std::vector<const Ort::MemoryInfo*> &outputMemoryInfos,
                                 const std::string &debugName)
{
    if (!session) {
        return std::string("[ERROR] runSessionWithOutputMemory(" + debugName + "): session is null.");
    }

    if (inputNames.size() != inputTensors.size()) {
        std::ostringstream oss;
        oss << "[ERROR] runSessionWithOutputMemory(" << debugName
            << "): input name count (" << inputNames.size()
            << ") does not match tensor count (" << inputTensors.size() << ").";
        return oss.str();
    }

    if (outputNames.size() != outputMemoryInfos.size()) {
        std::ostringstream oss;
        oss << "[ERROR] runSessionWithOutputMemory(" << debugName
            << "): output name count (" << outputNames.size()
            << ") does not match memory binding count (" << outputMemoryInfos.size() << ").";
        return oss.str();
    }

    try {
        Ort::IoBinding ioBinding(*session);
        for (size_t i = 0; i < inputNames.size(); ++i) {
            ioBinding.BindInput(inputNames[i], inputTensors[i]);
        }

        for (size_t i = 0; i < outputNames.size(); ++i) {
            const Ort::MemoryInfo* memoryInfo = outputMemoryInfos[i] ? outputMemoryInfos[i] : &m_memoryInfo;
            ioBinding.BindOutput(outputNames[i], *memoryInfo);
        }

        ioBinding.SynchronizeInputs();
        session->Run(Ort::RunOptions{nullptr}, ioBinding);
        ioBinding.SynchronizeOutputs();
        return ioBinding.GetOutputValues();
    }
    catch (const std::exception &e) {
        std::ostringstream oss;
        oss << "[ERROR] runSessionWithOutputMemory(" << debugName << ") => " << e.what();
        return oss.str();
    }
}

std::variant<std::vector<Ort::Value>, std::string>
SAM2::runImageEncoderSession(const std::vector<Ort::Value> &inputTensors)
{
    if (m_useCudaOutputBinding) {
        std::vector<const Ort::MemoryInfo*> outputMemoryInfos(
            m_imgEncoderOutputNames.size(),
            &m_cudaMemoryInfo);
        return runSessionWithOutputMemory(
            m_imgEncoderSession.get(),
            m_imgEncoderInputNames,
            m_imgEncoderOutputNames,
            inputTensors,
            outputMemoryInfos,
            "imgEncoderSession(cuda-bound)");
    }

    return runSession(
        m_imgEncoderSession.get(),
        m_imgEncoderInputNames,
        m_imgEncoderOutputNames,
        inputTensors,
        "imgEncoderSession");
}

std::variant<std::vector<Ort::Value>, std::string>
SAM2::runImageDecoderSession(const std::vector<Ort::Value> &inputTensors)
{
    if (m_useCudaOutputBinding) {
        std::vector<const Ort::MemoryInfo*> outputMemoryInfos;
        outputMemoryInfos.reserve(m_imgDecoderOutputNames.size());
        for (const char* outputName : m_imgDecoderOutputNames) {
            const std::string lowered = outputName ? lowerCopy(outputName) : std::string();
            outputMemoryInfos.push_back(
                lowered.find("mask_for_mem") != std::string::npos ? &m_cudaMemoryInfo : &m_memoryInfo);
        }

        return runSessionWithOutputMemory(
            m_imgDecoderSession.get(),
            m_imgDecoderInputNames,
            m_imgDecoderOutputNames,
            inputTensors,
            outputMemoryInfos,
            "imgDecoderSession(cuda-bound)");
    }

    return runSession(
        m_imgDecoderSession.get(),
        m_imgDecoderInputNames,
        m_imgDecoderOutputNames,
        inputTensors,
        "imgDecoderSession");
}

std::variant<std::vector<Ort::Value>, std::string>
SAM2::runVideoPropDecoderSession(const std::vector<Ort::Value> &inputTensors)
{
    if (m_useCudaOutputBinding) {
        const auto &outputNames = getPropDecoderOutputNames();
        std::vector<const Ort::MemoryInfo*> outputMemoryInfos;
        outputMemoryInfos.reserve(outputNames.size());
        for (const char* outputName : outputNames) {
            const std::string lowered = outputName ? lowerCopy(outputName) : std::string();
            outputMemoryInfos.push_back(
                lowered.find("mask_for_mem") != std::string::npos ? &m_cudaMemoryInfo : &m_memoryInfo);
        }

        return runSessionWithOutputMemory(
            getPropDecoderSession(),
            getPropDecoderInputNames(),
            outputNames,
            inputTensors,
            outputMemoryInfos,
            "videoPropDecoderSession(cuda-bound)");
    }

    return runSession(
        getPropDecoderSession(),
        getPropDecoderInputNames(),
        getPropDecoderOutputNames(),
        inputTensors,
        "videoPropDecoderSession");
}

std::variant<std::vector<Ort::Value>, std::string>
SAM2::runMemAttentionSession(const std::vector<Ort::Value> &inputTensors)
{
    if (m_useCudaOutputBinding) {
        std::vector<const Ort::MemoryInfo*> outputMemoryInfos(
            m_memAttentionOutputNames.size(),
            &m_cudaMemoryInfo);
        return runSessionWithOutputMemory(
            m_memAttentionSession.get(),
            m_memAttentionInputNames,
            m_memAttentionOutputNames,
            inputTensors,
            outputMemoryInfos,
            "memAttentionSession(cuda-bound)");
    }

    return runSession(
        m_memAttentionSession.get(),
        m_memAttentionInputNames,
        m_memAttentionOutputNames,
        inputTensors,
        "memAttentionSession");
}

std::variant<std::vector<Ort::Value>, std::string>
SAM2::runMemEncoderSession(const std::vector<Ort::Value> &inputTensors)
{
    return runSession(
        m_memEncoderSession.get(),
        m_memEncoderInputNames,
        m_memEncoderOutputNames,
        inputTensors,
        "memEncoderSession");
}

bool SAM2::hasCudaDriver()
{
    static int cached = -1;
    if (cached != -1) {
        return cached != 0;
    }

#if defined(_WIN32)
    static HMODULE hCUDART =
        LoadLibraryExW(L"cudart64_12.dll", nullptr, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!hCUDART) {
        cached = 0;
        return false;
    }

    using GetCnt = int (__cdecl*)(int*);
    auto fn = reinterpret_cast<GetCnt>(GetProcAddress(hCUDART, "cudaGetDeviceCount"));
    if (!fn) {
        cached = 0;
        return false;
    }

    int n = 0;
    const int err = fn(&n);
    cached = (err == 0 && n > 0) ? 1 : 0;
    return cached != 0;

#elif defined(__linux__)
    void* hCUDART = dlopen("libcudart.so.12", RTLD_LAZY | RTLD_LOCAL);
    if (!hCUDART) {
        cached = 0;
        return false;
    }

    using GetCnt = int (*)(int*);
    auto fn = reinterpret_cast<GetCnt>(dlsym(hCUDART, "cudaGetDeviceCount"));
    if (!fn) {
        cached = 0;
        return false;
    }

    int n = 0;
    const int err = fn(&n);
    cached = (err == 0 && n > 0) ? 1 : 0;
    return cached != 0;

#else
    cached = 0;
    return false;
#endif
}
