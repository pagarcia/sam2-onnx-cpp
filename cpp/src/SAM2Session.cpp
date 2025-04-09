#include "SAM2Session.h"

#include <fstream>
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <cstring> // for std::memcpy on some platforms

#ifdef _WIN32
#include <windows.h>
// Helper: convert std::string to wstring for ONNX on Windows
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
SAM2Session::SAM2Session()
{
    // Could init stuff here if needed
}

SAM2Session::~SAM2Session()
{
    clearAllSessions();
}

// --------------------
// Clear / reset
// --------------------
void SAM2Session::clearAllSessions()
{
    // Reset all session pointers
    m_imgEncoderSession.reset();
    m_imgDecoderSession.reset();
    m_memAttentionSession.reset();
    m_memEncoderSession.reset();

    // Clear node vectors
    m_imgEncoderInputNodes.clear();
    m_imgEncoderOutputNodes.clear();
    m_imgDecoderInputNodes.clear();
    m_imgDecoderOutputNodes.clear();
    m_memAttentionInputNodes.clear();
    m_memAttentionOutputNodes.clear();
    m_memEncoderInputNodes.clear();
    m_memEncoderOutputNodes.clear();

    // Clear shapes
    m_encoderInputShape.clear();
}

// --------------------
// Check file existence
// --------------------
bool SAM2Session::modelFileExists(const std::string &modelPath)
{
    std::ifstream f(modelPath.c_str());
    return f.good();
}

// --------------------
// Get encoder input size
// e.g. if shape is [1,3,1024,1024], returns (1024,1024)
// --------------------
cv::Size SAM2Session::getEncoderInputSize() const
{
    if (m_encoderInputShape.size() >= 4) {
        return cv::Size(
            static_cast<int>(m_encoderInputShape[3]),
            static_cast<int>(m_encoderInputShape[2])
        );
    }
    return cv::Size(0,0);
}

// --------------------
// Initialize image (encoder+decoder)
// --------------------
bool SAM2Session::initializeImage(const std::string &encoderPath,
                                   const std::string &decoderPath,
                                   int threadsNumber,
                                   const std::string &device)
{
    clearAllSessions();
    if (!modelFileExists(encoderPath) || !modelFileExists(decoderPath)) {
        std::cerr << "[ERROR] SAM2_Session::initializeImage => file not found.\n";
        return false;
    }

    // Configure session options
    setupSessionOptions(m_encoderOptions, threadsNumber, GraphOptimizationLevel::ORT_ENABLE_ALL, device);
    setupSessionOptions(m_decoderOptions, threadsNumber, GraphOptimizationLevel::ORT_ENABLE_ALL, device);

    try {
#ifdef _WIN32
        std::wstring wEnc = strToWstr(encoderPath);
        std::wstring wDec = strToWstr(decoderPath);

        m_imgEncoderSession = std::make_unique<Ort::Session>(m_envEncoder, wEnc.c_str(), m_encoderOptions);
        m_imgDecoderSession = std::make_unique<Ort::Session>(m_envDecoder, wDec.c_str(), m_decoderOptions);
#else
        m_imgEncoderSession_ = std::make_unique<Ort::Session>(m_envEncoder, encoderPath.c_str(), m_encoderOptions);
        m_imgDecoderSession_ = std::make_unique<Ort::Session>(m_envDecoder, decoderPath.c_str(), m_decoderOptions);
#endif

        // ---------------------
        // Query shapes for the encoder's input + output(s)
        // e.g. [1,3,1024,1024] => input; outputs => [1,256,64,64] plus other high-res feats
        // ---------------------
        {
            // The encoder might have multiple inputs, but typically 1 for your scenario
            // We'll just do the first input:
            auto inInfo = m_imgEncoderSession->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
            m_encoderInputShape = inInfo.GetShape(); // store for later reference
            // We can parse output shapes similarly, or we can do it on-demand
        }

        // Build img_encoder_input_nodes_ / output_nodes_
        {
            Ort::AllocatorWithDefaultOptions alloc;
            size_t inCount = m_imgEncoderSession->GetInputCount();
            m_imgEncoderInputNodes.clear();
            m_imgEncoderInputNodes.reserve(inCount);
            for (size_t i = 0; i < inCount; i++) {
                Node node;
                auto inName = m_imgEncoderSession->GetInputNameAllocated(i, alloc);
                node.name = std::string(inName.get());

                auto shape = m_imgEncoderSession->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
                node.dim.assign(shape.begin(), shape.end());

                m_imgEncoderInputNodes.push_back(std::move(node));
            }
            size_t outCount = m_imgEncoderSession->GetOutputCount();
            m_imgEncoderOutputNodes.clear();
            m_imgEncoderOutputNodes.reserve(outCount);
            for (size_t i = 0; i < outCount; i++) {
                Node node;
                auto outName = m_imgEncoderSession->GetOutputNameAllocated(i, alloc);
                node.name = std::string(outName.get());

                auto shape = m_imgEncoderSession->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
                node.dim.assign(shape.begin(), shape.end());

                m_imgEncoderOutputNodes.push_back(std::move(node));
            }
        }

        // Build img_decoder_input_nodes_ / output_nodes_ similarly
        {
            Ort::AllocatorWithDefaultOptions alloc;
            size_t inCount = m_imgDecoderSession->GetInputCount();
            m_imgDecoderInputNodes.clear();
            m_imgDecoderInputNodes.reserve(inCount);
            for (size_t i = 0; i < inCount; i++) {
                Node node;
                auto inName = m_imgDecoderSession->GetInputNameAllocated(i, alloc);
                node.name = std::string(inName.get());

                auto shape = m_imgDecoderSession->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
                node.dim.assign(shape.begin(), shape.end());
                m_imgDecoderInputNodes.push_back(std::move(node));
            }
            size_t outCount = m_imgDecoderSession->GetOutputCount();
            m_imgDecoderOutputNodes.clear();
            m_imgDecoderOutputNodes.reserve(outCount);
            for (size_t i = 0; i < outCount; i++) {
                Node node;
                auto outName = m_imgDecoderSession->GetOutputNameAllocated(i, alloc);
                node.name = std::string(outName.get());

                auto shape = m_imgDecoderSession->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
                node.dim.assign(shape.begin(), shape.end());
                m_imgDecoderOutputNodes.push_back(std::move(node));
            }
        }

        std::cout << "[DEBUG] SAM2_Session::initializeImage => success.\n";
    }
    catch(const std::exception &e) {
        std::cerr << "[ERROR] initializeImage => " << e.what() << std::endl;
        return false;
    }

    return true;
}

// --------------------
// Initialize memory-based models
// --------------------
bool SAM2Session::initializeVideo(const std::string &memAttentionPath,
                                   const std::string &memEncoderPath,
                                   int threadsNumber,
                                   const std::string &device)
{
    // We assume the image sessions are already loaded or not, depending on your usage
    if(!modelFileExists(memAttentionPath) || !modelFileExists(memEncoderPath)) {
        std::cerr << "[ERROR] memory model files not found.\n";
        return false;
    }

    // Setup
    setupSessionOptions(m_memAttentionOptions, threadsNumber, GraphOptimizationLevel::ORT_ENABLE_ALL, device);
    setupSessionOptions(m_memEncoderOptions, threadsNumber, GraphOptimizationLevel::ORT_ENABLE_ALL, device);

    try {
#ifdef _WIN32
        std::wstring wAttn = strToWstr(memAttentionPath);
        std::wstring wEnc2 = strToWstr(memEncoderPath);

        m_memAttentionSession = std::make_unique<Ort::Session>(m_envMemAttention, wAttn.c_str(), m_memAttentionOptions);
        m_memEncoderSession = std::make_unique<Ort::Session>(m_envMemEncoder, wEnc2.c_str(), m_memEncoderOptions);
#else
        m_memAttentionSession = std::make_unique<Ort::Session>(m_envMemAttention, memAttentionPath.c_str(), m_memAttentionOptions);
        m_memEncoderSession = std::make_unique<Ort::Session>(m_envMemEncoder, memEncoderPath.c_str(), m_memEncoderOptions);
#endif

        // Build mem_attention_input_nodes_, etc.
        {
            Ort::AllocatorWithDefaultOptions alloc;
            size_t inCount = m_memAttentionSession->GetInputCount();
            m_memAttentionInputNodes.clear();
            m_memAttentionInputNodes.reserve(inCount);
            for (size_t i=0; i<inCount; i++) {
                Node node;
                auto inName = m_memAttentionSession->GetInputNameAllocated(i, alloc);
                node.name = std::string(inName.get());
                auto shape = m_memAttentionSession->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
                node.dim.assign(shape.begin(), shape.end());
                m_memAttentionInputNodes.push_back(std::move(node));
            }

            size_t outCount = m_memAttentionSession->GetOutputCount();
            m_memAttentionOutputNodes.clear();
            m_memAttentionOutputNodes.reserve(outCount);
            for (size_t i=0; i<outCount; i++) {
                Node node;
                auto outName = m_memAttentionSession->GetOutputNameAllocated(i, alloc);
                node.name = std::string(outName.get());
                auto shape = m_memAttentionSession->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
                node.dim.assign(shape.begin(), shape.end());
                m_memAttentionOutputNodes.push_back(std::move(node));
            }
        }
        // Build mem_encoder_input_nodes_, etc.
        {
            Ort::AllocatorWithDefaultOptions alloc;
            size_t inCount = m_memEncoderSession->GetInputCount();
            m_memEncoderInputNodes.clear();
            m_memEncoderInputNodes.reserve(inCount);
            for (size_t i=0; i<inCount; i++){
                Node node;
                auto inName = m_memEncoderSession->GetInputNameAllocated(i,alloc);
                node.name = std::string(inName.get());
                auto shape = m_memEncoderSession->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
                node.dim.assign(shape.begin(), shape.end());
                m_memEncoderInputNodes.push_back(std::move(node));
            }

            size_t outCount = m_memEncoderSession->GetOutputCount();
            m_memEncoderOutputNodes.clear();
            m_memEncoderOutputNodes.reserve(outCount);
            for (size_t i=0; i<outCount; i++){
                Node node;
                auto outName = m_memEncoderSession->GetOutputNameAllocated(i,alloc);
                node.name = std::string(outName.get());
                auto shape = m_memEncoderSession->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
                node.dim.assign(shape.begin(), shape.end());
                m_memEncoderOutputNodes.push_back(std::move(node));
            }
        }

        std::cout << "[DEBUG] SAM2_Session::initializeVideo => success.\n";
    }
    catch(const std::exception &e) {
        std::cerr << "[ERROR] initializeVideo => " << e.what() << std::endl;
        return false;
    }
    return true;
}

// --------------------
// Setup session options
// (CPU, CUDA, CoreML, etc.)
// --------------------
void SAM2Session::setupSessionOptions(Ort::SessionOptions &options,
                                       int threadsNumber,
                                       GraphOptimizationLevel optLevel,
                                       const std::string &device)
{
    options.SetIntraOpNumThreads(threadsNumber);
    options.SetGraphOptimizationLevel(optLevel);

    if(device == "cpu") {
        std::cout << "[DEBUG] Using CPU EP.\n";
        int use_arena = 1;
        OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CPU(options, use_arena);
        if(status != nullptr) {
            const char* err = Ort::GetApi().GetErrorMessage(status);
            Ort::GetApi().ReleaseStatus(status);
            throw std::runtime_error(std::string("Error appending CPU provider: ")+err);
        }
    } 
    else if(device.rfind("cuda:", 0) == 0) {
        std::cout << "[DEBUG] Using CUDA EP.\n";
        int gpuId = std::stoi(device.substr(5));
        OrtCUDAProviderOptions cudaOpts;
        cudaOpts.device_id = gpuId;
        // If needed, set other cudaOpts fields
        options.AppendExecutionProvider_CUDA(cudaOpts);
    }
#ifdef __APPLE__
    else if(device.rfind("coreml", 0) == 0) {
        std::cout << "[DEBUG] Using CoreML EP.\n";
        uint32_t coreml_flags = COREML_FLAG_USE_NONE;
        OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CoreML(options, coreml_flags);
        if(status != nullptr) {
            const char* err = Ort::GetApi().GetErrorMessage(status);
            Ort::GetApi().ReleaseStatus(status);
            throw std::runtime_error(std::string("Error appending CoreML provider: ")+err);
        }
    }
#endif
    else {
        // Fallback
        std::cout << "[DEBUG] Unknown device => fallback to CPU.\n";
        int use_arena = 1;
        OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CPU(options, use_arena);
        if(status != nullptr) {
            const char* err = Ort::GetApi().GetErrorMessage(status);
            Ort::GetApi().ReleaseStatus(status);
            throw std::runtime_error(std::string("Error appending default CPU provider: ")+err);
        }
    }
}

// --------------------
// runSession helper
// --------------------
std::variant<std::vector<Ort::Value>, std::string>
SAM2Session::runSession(Ort::Session* session,
                         const std::vector<Node> &inputNodes,
                         const std::vector<Node> &outputNodes,
                         const std::vector<Ort::Value> &inputTensors,
                         const std::string &debugName)
{
    if(!session) {
        return std::string("[ERROR] runSession(" + debugName + "): session is null.\n");
    }
    std::vector<const char*> inNames;
    inNames.reserve(inputNodes.size());
    for(const auto &nd : inputNodes) {
        inNames.push_back(nd.name.c_str());
    }

    std::vector<const char*> outNames;
    outNames.reserve(outputNodes.size());
    for(const auto &nd : outputNodes) {
        outNames.push_back(nd.name.c_str());
    }

    try {
        auto outputs = session->Run(
            Ort::RunOptions{nullptr},
            inNames.data(),
            const_cast<Ort::Value*>(inputTensors.data()), // ORT expects non-const
            inputTensors.size(),
            outNames.data(),
            outNames.size()
        );
        return outputs; // success => vector<Ort::Value>
    }
    catch(const std::exception &e) {
        std::ostringstream oss;
        oss << "[ERROR] runSession(" << debugName << ") => " << e.what();
        return oss.str();
    }
}

// --------------------
// Model sub-runs
// --------------------
std::variant<std::vector<Ort::Value>, std::string>
SAM2Session::runImageEncoder(const std::vector<Ort::Value> &inputTensors)
{
    return runSession(m_imgEncoderSession.get(),
                      m_imgEncoderInputNodes,
                      m_imgEncoderOutputNodes,
                      inputTensors,
                      "img_encoder_session");
}

std::variant<std::vector<Ort::Value>, std::string>
SAM2Session::runImageDecoder(const std::vector<Ort::Value> &inputTensors)
{
    return runSession(m_imgDecoderSession.get(),
                      m_imgDecoderInputNodes,
                      m_imgDecoderOutputNodes,
                      inputTensors,
                      "img_decoder_session");
}

std::variant<std::vector<Ort::Value>, std::string>
SAM2Session::runMemAttention(const std::vector<Ort::Value> &inputTensors)
{
    return runSession(m_memAttentionSession.get(),
                      m_memAttentionInputNodes,
                      m_memAttentionOutputNodes,
                      inputTensors,
                      "mem_attention_session");
}

std::variant<std::vector<Ort::Value>, std::string>
SAM2Session::runMemEncoder(const std::vector<Ort::Value> &inputTensors)
{
    return runSession(m_memEncoderSession.get(),
                      m_memEncoderInputNodes,
                      m_memEncoderOutputNodes,
                      inputTensors,
                      "mem_encoder_session");
}
