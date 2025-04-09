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

Size SAM2::getInputSize()
{
    // Typically [1,3,1024,1024] => shape[2]=1024 (H), shape[3]=1024 (W)
    if (m_inputShapeEncoder.size() >= 4) {
        return Size(
            static_cast<int>(m_inputShapeEncoder[3]),
            static_cast<int>(m_inputShapeEncoder[2])
        );
    }
    return Size(0, 0);
}

void SAM2::setupSessionOptions(Ort::SessionOptions &options,
                               int threadsNumber,
                               GraphOptimizationLevel optLevel,
                               const std::string &device)
{
    options.SetIntraOpNumThreads(threadsNumber);
    options.SetGraphOptimizationLevel(optLevel);

    if (device == "cpu") {
        std::cout << "[DEBUG] Using CPU execution provider." << std::endl;
        int use_arena = 1;
        OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CPU(options, use_arena);
        if (status != nullptr) {
            const char* error_message = Ort::GetApi().GetErrorMessage(status);
            Ort::GetApi().ReleaseStatus(status);
            throw std::runtime_error(std::string("Error appending CPU execution provider: ") + error_message);
        }
    }
    else if (device.rfind("cuda:", 0) == 0) {
        std::cout << "[DEBUG] Using CUDA execution provider." << std::endl;
        int gpuId = std::stoi(device.substr(5));
        OrtCUDAProviderOptions cudaOpts;
        cudaOpts.device_id = gpuId;
        options.AppendExecutionProvider_CUDA(cudaOpts);
    }
    else if (device.rfind("coreml", 0) == 0) {
#ifdef __APPLE__
        std::cout << "[DEBUG] Using CoreML execution provider." << std::endl;
        // For CoreML, you need to provide a uint32_t flag.
        // Here we use the default flag (COREML_FLAG_USE_NONE). You can modify coreml_flags as needed.
        uint32_t coreml_flags = COREML_FLAG_USE_NONE;
        OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CoreML(options, coreml_flags);
        if (status != nullptr) {
            const char* error_message = Ort::GetApi().GetErrorMessage(status);
            Ort::GetApi().ReleaseStatus(status);
            throw std::runtime_error(std::string("Error appending CoreML execution provider: ") + error_message);
        }
#else
        std::cout << "[WARN] CoreML requested but not supported on this platform. Defaulting to CPU." << std::endl;
        int use_arena = 1;
        OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CPU(options, use_arena);
        if (status != nullptr) {
            const char* error_message = Ort::GetApi().GetErrorMessage(status);
            Ort::GetApi().ReleaseStatus(status);
            throw std::runtime_error(std::string("Error appending CPU execution provider: ") + error_message);
        }
#endif
    }
    else {
        std::cout << "[DEBUG] Unknown device type. Defaulting to CPU execution provider." << std::endl;
        int use_arena = 1;
        OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CPU(options, use_arena);
        if (status != nullptr) {
            const char* error_message = Ort::GetApi().GetErrorMessage(status);
            Ort::GetApi().ReleaseStatus(status);
            throw std::runtime_error(std::string("Error appending default CPU execution provider: ") + error_message);
        }
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

    // Configure session options
    setupSessionOptions(m_encoderOptions, threadsNumber, GraphOptimizationLevel::ORT_ENABLE_ALL, device);
    setupSessionOptions(m_decoderOptions, threadsNumber, GraphOptimizationLevel::ORT_ENABLE_ALL, device);

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

        // Query shapes for the encoder's 3 main outputs
        {
            auto encInputInfo = m_imgEncoderSession->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
            m_inputShapeEncoder = encInputInfo.GetShape();

            auto out0Info = m_imgEncoderSession->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();
            m_outputShapeEncoder = out0Info.GetShape();

            auto out1Info = m_imgEncoderSession->GetOutputTypeInfo(1).GetTensorTypeAndShapeInfo();
            m_highResFeatures1Shape = out1Info.GetShape();

            auto out2Info = m_imgEncoderSession->GetOutputTypeInfo(2).GetTensorTypeAndShapeInfo();
            m_highResFeatures2Shape = out2Info.GetShape();
        }

        // Gather input/output node info for the encoder
        {
            Ort::AllocatorWithDefaultOptions alloc;
            m_imgEncoderInputNodes.clear();
            size_t inCountEnc = m_imgEncoderSession->GetInputCount();
            for (size_t i = 0; i < inCountEnc; i++) {
                Node node;
                auto inNamePtr = m_imgEncoderSession->GetInputNameAllocated(i, alloc);
                node.name = std::string(inNamePtr.get());
                auto shape = m_imgEncoderSession->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
                node.dim.assign(shape.begin(), shape.end());
                m_imgEncoderInputNodes.push_back(std::move(node));
            }

            m_imgEncoderOutputNodes.clear();
            size_t outCountEnc = m_imgEncoderSession->GetOutputCount();
            for (size_t i = 0; i < outCountEnc; i++) {
                Node node;
                auto outNamePtr = m_imgEncoderSession->GetOutputNameAllocated(i, alloc);
                node.name = std::string(outNamePtr.get());
                auto shape = m_imgEncoderSession->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
                node.dim.assign(shape.begin(), shape.end());
                m_imgEncoderOutputNodes.push_back(std::move(node));
            }
        }

        // Gather input/output node info for the decoder
        {
            Ort::AllocatorWithDefaultOptions alloc;
            m_imgDecoderInputNodes.clear();
            size_t inCountDec = m_imgDecoderSession->GetInputCount();
            for (size_t i = 0; i < inCountDec; i++) {
                Node node;
                auto inNamePtr = m_imgDecoderSession->GetInputNameAllocated(i, alloc);
                node.name = std::string(inNamePtr.get());
                auto shape = m_imgDecoderSession->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
                node.dim.assign(shape.begin(), shape.end());
                m_imgDecoderInputNodes.push_back(std::move(node));
            }

            m_imgDecoderOutputNodes.clear();
            size_t outCountDec = m_imgDecoderSession->GetOutputCount();
            for (size_t i = 0; i < outCountDec; i++) {
                Node node;
                auto outNamePtr = m_imgDecoderSession->GetOutputNameAllocated(i, alloc);
                node.name = std::string(outNamePtr.get());
                auto shape = m_imgDecoderSession->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
                node.dim.assign(shape.begin(), shape.end());
                m_imgDecoderOutputNodes.push_back(std::move(node));
            }
        }

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

    setupSessionOptions(m_memAttentionOptions, threadsNumber, GraphOptimizationLevel::ORT_ENABLE_ALL, device);
    setupSessionOptions(m_memEncoderOptions, threadsNumber, GraphOptimizationLevel::ORT_ENABLE_ALL, device);

    try {
    #ifdef _WIN32
        std::wstring wAttn = strToWstr(memAttentionPath);
        std::wstring wEnc2 = strToWstr(memEncoderPath);

        m_memAttentionSession = std::make_unique<Ort::Session>(m_memAttentionEnv, wAttn.c_str(), m_memAttentionOptions);
        m_memEncoderSession  = std::make_unique<Ort::Session>(m_memEncoderEnv, wEnc2.c_str(), m_memEncoderOptions);
    #else
        m_memAttentionSession = std::make_unique<Ort::Session>(m_memAttentionEnv, memAttentionPath.c_str(), m_memAttentionOptions);
        m_memEncoderSession  = std::make_unique<Ort::Session>(m_memEncoderEnv, memEncoderPath.c_str(), m_memEncoderOptions);
    #endif

        // gather node info for mem_attention
        {
            Ort::AllocatorWithDefaultOptions alloc;
            m_memAttentionInputNodes.clear();
            size_t inCount= m_memAttentionSession->GetInputCount();
            for(size_t i=0; i<inCount; i++){
                Node node;
                auto inName= m_memAttentionSession->GetInputNameAllocated(i,alloc);
                node.name= std::string(inName.get());
                auto shape= m_memAttentionSession->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
                node.dim.assign(shape.begin(), shape.end());
                m_memAttentionInputNodes.push_back(std::move(node));
            }
            m_memAttentionOutputNodes.clear();
            size_t outCount= m_memAttentionSession->GetOutputCount();
            for(size_t i=0; i<outCount; i++){
                Node node;
                auto outName= m_memAttentionSession->GetOutputNameAllocated(i,alloc);
                node.name= std::string(outName.get());
                auto shape= m_memAttentionSession->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
                node.dim.assign(shape.begin(), shape.end());
                m_memAttentionOutputNodes.push_back(std::move(node));
            }
        }

        // gather node info for mem_encoder
        {
            Ort::AllocatorWithDefaultOptions alloc;
            m_memEncoderInputNodes.clear();
            size_t inCount= m_memEncoderSession->GetInputCount();
            for(size_t i=0; i<inCount; i++){
                Node node;
                auto inName= m_memEncoderSession->GetInputNameAllocated(i,alloc);
                node.name= std::string(inName.get());
                auto shape= m_memEncoderSession->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
                node.dim.assign(shape.begin(), shape.end());
                m_memEncoderInputNodes.push_back(std::move(node));
            }
            m_memEncoderOutputNodes.clear();
            size_t outCount= m_memEncoderSession->GetOutputCount();
            for(size_t i=0; i<outCount; i++){
                Node node;
                auto outName= m_memEncoderSession->GetOutputNameAllocated(i,alloc);
                node.name= std::string(outName.get());
                auto shape= m_memEncoderSession->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
                node.dim.assign(shape.begin(), shape.end());
                m_memEncoderOutputNodes.push_back(std::move(node));
            }
        }

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
                 const std::vector<Node> &inputNodes,
                 const std::vector<Node> &outputNodes,
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
// 4 pipeline-step methods
// --------------------
std::variant<std::vector<Ort::Value>, std::string>
SAM2::runImageEncoder(const std::vector<Ort::Value> &inputTensors)
{
    return runSession(m_imgEncoderSession.get(),
                      m_imgEncoderInputNodes,
                      m_imgEncoderOutputNodes,
                      inputTensors,
                      "imgEncoderSession");
}

std::variant<std::vector<Ort::Value>, std::string>
SAM2::runImageDecoder(const std::vector<Ort::Value> &inputTensors)
{
    return runSession(m_imgDecoderSession.get(),
                      m_imgDecoderInputNodes,
                      m_imgDecoderOutputNodes,
                      inputTensors,
                      "imgDecoderSession");
}

std::variant<std::vector<Ort::Value>, std::string>
SAM2::runMemAttention(const std::vector<Ort::Value> &inputTensors)
{
    return runSession(m_memAttentionSession.get(),
                      m_memAttentionInputNodes,
                      m_memAttentionOutputNodes,
                      inputTensors,
                      "memAttentionSession");
}

std::variant<std::vector<Ort::Value>, std::string>
SAM2::runMemEncoder(const std::vector<Ort::Value> &inputTensors)
{
    return runSession(m_memEncoderSession.get(),
                      m_memEncoderInputNodes,
                      m_memEncoderOutputNodes,
                      inputTensors,
                      "memEncoderSession");
}
