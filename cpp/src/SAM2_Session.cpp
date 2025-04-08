#include "SAM2_Session.h"

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
    int sizeNeeded = MultiByteToWideChar(CP_UTF8, 0,
                                         str.c_str(),
                                         (int)str.size(),
                                         NULL, 0);
    std::wstring wstr(sizeNeeded, 0);
    MultiByteToWideChar(CP_UTF8, 0,
                        str.c_str(),
                        (int)str.size(),
                        &wstr[0], sizeNeeded);
    return wstr;
}
#endif

// --------------------
// Constructor / Destructor
// --------------------
SAM2_Session::SAM2_Session()
{
    // Could init stuff here if needed
}

SAM2_Session::~SAM2_Session()
{
    clearAllSessions();
}

// --------------------
// Clear / reset
// --------------------
void SAM2_Session::clearAllSessions()
{
    // Reset all session pointers
    img_encoder_session_.reset();
    img_decoder_session_.reset();
    mem_attention_session_.reset();
    mem_encoder_session_.reset();

    // Clear node vectors
    img_encoder_input_nodes_.clear();
    img_encoder_output_nodes_.clear();
    img_decoder_input_nodes_.clear();
    img_decoder_output_nodes_.clear();
    mem_attention_input_nodes_.clear();
    mem_attention_output_nodes_.clear();
    mem_encoder_input_nodes_.clear();
    mem_encoder_output_nodes_.clear();

    // Clear shapes
    encoderInputShape_.clear();
}

// --------------------
// Check file existence
// --------------------
bool SAM2_Session::modelFileExists(const std::string &modelPath)
{
    std::ifstream f(modelPath.c_str());
    return f.good();
}

// --------------------
// Get encoder input size
// e.g. if shape is [1,3,1024,1024], returns (1024,1024)
// --------------------
cv::Size SAM2_Session::getEncoderInputSize() const
{
    if (encoderInputShape_.size() >= 4) {
        return cv::Size(
            static_cast<int>(encoderInputShape_[3]),
            static_cast<int>(encoderInputShape_[2])
        );
    }
    return cv::Size(0,0);
}

// --------------------
// Initialize image (encoder+decoder)
// --------------------
bool SAM2_Session::initializeImage(const std::string &encoderPath,
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
    setupSessionOptions(encoderOptions_, threadsNumber, 
                        GraphOptimizationLevel::ORT_ENABLE_ALL, device);
    setupSessionOptions(decoderOptions_, threadsNumber,
                        GraphOptimizationLevel::ORT_ENABLE_ALL, device);

    try {
#ifdef _WIN32
        std::wstring wEnc = strToWstr(encoderPath);
        std::wstring wDec = strToWstr(decoderPath);

        img_encoder_session_ = std::make_unique<Ort::Session>(
            env_encoder_, wEnc.c_str(), encoderOptions_
        );
        img_decoder_session_ = std::make_unique<Ort::Session>(
            env_decoder_, wDec.c_str(), decoderOptions_
        );
#else
        img_encoder_session_ = std::make_unique<Ort::Session>(
            env_encoder_, encoderPath.c_str(), encoderOptions_
        );
        img_decoder_session_ = std::make_unique<Ort::Session>(
            env_decoder_, decoderPath.c_str(), decoderOptions_
        );
#endif

        // ---------------------
        // Query shapes for the encoder's input + output(s)
        // e.g. [1,3,1024,1024] => input; outputs => [1,256,64,64] plus other high-res feats
        // ---------------------
        {
            // The encoder might have multiple inputs, but typically 1 for your scenario
            // We'll just do the first input:
            auto inInfo = img_encoder_session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
            encoderInputShape_ = inInfo.GetShape(); // store for later reference

            // We can parse output shapes similarly, or we can do it on-demand
        }

        // Build img_encoder_input_nodes_ / output_nodes_
        {
            Ort::AllocatorWithDefaultOptions alloc;
            size_t inCount = img_encoder_session_->GetInputCount();
            img_encoder_input_nodes_.clear();
            img_encoder_input_nodes_.reserve(inCount);
            for (size_t i = 0; i < inCount; i++) {
                Node node;
                auto inName = img_encoder_session_->GetInputNameAllocated(i, alloc);
                node.name = std::string(inName.get());

                auto shape = img_encoder_session_
                                ->GetInputTypeInfo(i)
                                .GetTensorTypeAndShapeInfo()
                                .GetShape();
                node.dim.assign(shape.begin(), shape.end());

                img_encoder_input_nodes_.push_back(std::move(node));
            }
            size_t outCount = img_encoder_session_->GetOutputCount();
            img_encoder_output_nodes_.clear();
            img_encoder_output_nodes_.reserve(outCount);
            for (size_t i = 0; i < outCount; i++) {
                Node node;
                auto outName = img_encoder_session_->GetOutputNameAllocated(i, alloc);
                node.name = std::string(outName.get());

                auto shape = img_encoder_session_
                                ->GetOutputTypeInfo(i)
                                .GetTensorTypeAndShapeInfo()
                                .GetShape();
                node.dim.assign(shape.begin(), shape.end());

                img_encoder_output_nodes_.push_back(std::move(node));
            }
        }

        // Build img_decoder_input_nodes_ / output_nodes_ similarly
        {
            Ort::AllocatorWithDefaultOptions alloc;
            size_t inCount = img_decoder_session_->GetInputCount();
            img_decoder_input_nodes_.clear();
            img_decoder_input_nodes_.reserve(inCount);
            for (size_t i = 0; i < inCount; i++) {
                Node node;
                auto inName = img_decoder_session_->GetInputNameAllocated(i, alloc);
                node.name = std::string(inName.get());

                auto shape = img_decoder_session_
                                ->GetInputTypeInfo(i)
                                .GetTensorTypeAndShapeInfo()
                                .GetShape();
                node.dim.assign(shape.begin(), shape.end());
                img_decoder_input_nodes_.push_back(std::move(node));
            }
            size_t outCount = img_decoder_session_->GetOutputCount();
            img_decoder_output_nodes_.clear();
            img_decoder_output_nodes_.reserve(outCount);
            for (size_t i = 0; i < outCount; i++) {
                Node node;
                auto outName = img_decoder_session_->GetOutputNameAllocated(i, alloc);
                node.name = std::string(outName.get());

                auto shape = img_decoder_session_
                                ->GetOutputTypeInfo(i)
                                .GetTensorTypeAndShapeInfo()
                                .GetShape();
                node.dim.assign(shape.begin(), shape.end());
                img_decoder_output_nodes_.push_back(std::move(node));
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
bool SAM2_Session::initializeVideo(const std::string &memAttentionPath,
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
    setupSessionOptions(memAttentionOptions_, threadsNumber, 
                        GraphOptimizationLevel::ORT_ENABLE_ALL, device);
    setupSessionOptions(memEncoderOptions_, threadsNumber,
                        GraphOptimizationLevel::ORT_ENABLE_ALL, device);

    try {
#ifdef _WIN32
        std::wstring wAttn = strToWstr(memAttentionPath);
        std::wstring wEnc2 = strToWstr(memEncoderPath);

        mem_attention_session_ = std::make_unique<Ort::Session>(
            env_mem_attention_, wAttn.c_str(), memAttentionOptions_
        );
        mem_encoder_session_ = std::make_unique<Ort::Session>(
            env_mem_encoder_, wEnc2.c_str(), memEncoderOptions_
        );
#else
        mem_attention_session_ = std::make_unique<Ort::Session>(
            env_mem_attention_, memAttentionPath.c_str(), memAttentionOptions_
        );
        mem_encoder_session_ = std::make_unique<Ort::Session>(
            env_mem_encoder_, memEncoderPath.c_str(), memEncoderOptions_
        );
#endif

        // Build mem_attention_input_nodes_, etc.
        {
            Ort::AllocatorWithDefaultOptions alloc;
            size_t inCount = mem_attention_session_->GetInputCount();
            mem_attention_input_nodes_.clear();
            mem_attention_input_nodes_.reserve(inCount);
            for (size_t i=0; i<inCount; i++) {
                Node node;
                auto inName = mem_attention_session_->GetInputNameAllocated(i, alloc);
                node.name = std::string(inName.get());
                auto shape = mem_attention_session_
                                ->GetInputTypeInfo(i)
                                .GetTensorTypeAndShapeInfo()
                                .GetShape();
                node.dim.assign(shape.begin(), shape.end());
                mem_attention_input_nodes_.push_back(std::move(node));
            }

            size_t outCount = mem_attention_session_->GetOutputCount();
            mem_attention_output_nodes_.clear();
            mem_attention_output_nodes_.reserve(outCount);
            for (size_t i=0; i<outCount; i++) {
                Node node;
                auto outName = mem_attention_session_->GetOutputNameAllocated(i, alloc);
                node.name = std::string(outName.get());
                auto shape = mem_attention_session_
                                ->GetOutputTypeInfo(i)
                                .GetTensorTypeAndShapeInfo()
                                .GetShape();
                node.dim.assign(shape.begin(), shape.end());
                mem_attention_output_nodes_.push_back(std::move(node));
            }
        }
        // Build mem_encoder_input_nodes_, etc.
        {
            Ort::AllocatorWithDefaultOptions alloc;
            size_t inCount = mem_encoder_session_->GetInputCount();
            mem_encoder_input_nodes_.clear();
            mem_encoder_input_nodes_.reserve(inCount);
            for (size_t i=0; i<inCount; i++){
                Node node;
                auto inName = mem_encoder_session_->GetInputNameAllocated(i,alloc);
                node.name = std::string(inName.get());
                auto shape = mem_encoder_session_
                                ->GetInputTypeInfo(i)
                                .GetTensorTypeAndShapeInfo()
                                .GetShape();
                node.dim.assign(shape.begin(), shape.end());
                mem_encoder_input_nodes_.push_back(std::move(node));
            }

            size_t outCount = mem_encoder_session_->GetOutputCount();
            mem_encoder_output_nodes_.clear();
            mem_encoder_output_nodes_.reserve(outCount);
            for (size_t i=0; i<outCount; i++){
                Node node;
                auto outName = mem_encoder_session_->GetOutputNameAllocated(i,alloc);
                node.name = std::string(outName.get());
                auto shape = mem_encoder_session_
                                ->GetOutputTypeInfo(i)
                                .GetTensorTypeAndShapeInfo()
                                .GetShape();
                node.dim.assign(shape.begin(), shape.end());
                mem_encoder_output_nodes_.push_back(std::move(node));
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
void SAM2_Session::setupSessionOptions(Ort::SessionOptions &options,
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
SAM2_Session::runSession(Ort::Session* session,
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
SAM2_Session::runImageEncoder(const std::vector<Ort::Value> &inputTensors)
{
    return runSession(img_encoder_session_.get(),
                      img_encoder_input_nodes_,
                      img_encoder_output_nodes_,
                      inputTensors,
                      "img_encoder_session");
}

std::variant<std::vector<Ort::Value>, std::string>
SAM2_Session::runImageDecoder(const std::vector<Ort::Value> &inputTensors)
{
    return runSession(img_decoder_session_.get(),
                      img_decoder_input_nodes_,
                      img_decoder_output_nodes_,
                      inputTensors,
                      "img_decoder_session");
}

std::variant<std::vector<Ort::Value>, std::string>
SAM2_Session::runMemAttention(const std::vector<Ort::Value> &inputTensors)
{
    return runSession(mem_attention_session_.get(),
                      mem_attention_input_nodes_,
                      mem_attention_output_nodes_,
                      inputTensors,
                      "mem_attention_session");
}

std::variant<std::vector<Ort::Value>, std::string>
SAM2_Session::runMemEncoder(const std::vector<Ort::Value> &inputTensors)
{
    return runSession(mem_encoder_session_.get(),
                      mem_encoder_input_nodes_,
                      mem_encoder_output_nodes_,
                      inputTensors,
                      "mem_encoder_session");
}
