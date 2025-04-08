#include "SAM2_Session.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cpu_provider_factory.h>

#ifdef _WIN32
#include <windows.h>
// Utility function to convert std::string to std::wstring on Windows
static std::wstring strToWstr(const std::string &str) {
    int sizeNeeded = MultiByteToWideChar(CP_UTF8, 0,
                                           str.c_str(),
                                           static_cast<int>(str.size()),
                                           NULL, 0);
    std::wstring wstr(sizeNeeded, 0);
    MultiByteToWideChar(CP_UTF8, 0,
                        str.c_str(),
                        static_cast<int>(str.size()),
                        &wstr[0], sizeNeeded);
    return wstr;
}
#endif

// Create static environments for each session type.
// Ensuring that the Ort::Env objects live at least as long as the sessions.
static Ort::Env encoderEnv(ORT_LOGGING_LEVEL_WARNING, "img_encoder");
static Ort::Env decoderEnv(ORT_LOGGING_LEVEL_WARNING, "img_decoder");
static Ort::Env memAttentionEnv(ORT_LOGGING_LEVEL_WARNING, "mem_attention");
static Ort::Env memEncoderEnv(ORT_LOGGING_LEVEL_WARNING, "mem_encoder");

SAM2_Session::SAM2_Session() {
    // Constructor – nothing to do here, sessions are created via init methods.
}

SAM2_Session::~SAM2_Session() {
    // Destructor – unique_ptr will automatically free sessions.
}

bool SAM2_Session::initImageEncoder(const std::string &encoderPath, int threadsNumber, const std::string &device) {
    // Check if the model file exists.
    std::ifstream f(encoderPath.c_str());
    if (!f.good()) {
        std::cerr << "[ERROR] Encoder model file not found: " << encoderPath << "\n";
        return false;
    }

    Ort::SessionOptions options;
    setupSessionOptions(options, threadsNumber, GraphOptimizationLevel::ORT_ENABLE_ALL, device);

#ifdef _WIN32
    std::wstring wPath = strToWstr(encoderPath);
    try {
        img_encoder_session_ = std::make_unique<Ort::Session>(encoderEnv, wPath.c_str(), options);
    } catch (const std::exception &e) {
        std::cerr << "[ERROR] initImageEncoder: " << e.what() << "\n";
        return false;
    }
#else
    try {
        img_encoder_session_ = std::make_unique<Ort::Session>(encoderEnv, encoderPath.c_str(), options);
    } catch (const std::exception &e) {
        std::cerr << "[ERROR] initImageEncoder: " << e.what() << "\n";
        return false;
    }
#endif

    return true;
}

bool SAM2_Session::initImageDecoder(const std::string &decoderPath, int threadsNumber, const std::string &device) {
    std::ifstream f(decoderPath.c_str());
    if (!f.good()) {
        std::cerr << "[ERROR] Decoder model file not found: " << decoderPath << "\n";
        return false;
    }
    Ort::SessionOptions options;
    setupSessionOptions(options, threadsNumber, GraphOptimizationLevel::ORT_ENABLE_ALL, device);

#ifdef _WIN32
    std::wstring wPath = strToWstr(decoderPath);
    try {
        img_decoder_session_ = std::make_unique<Ort::Session>(decoderEnv, wPath.c_str(), options);
    } catch (const std::exception &e) {
        std::cerr << "[ERROR] initImageDecoder: " << e.what() << "\n";
        return false;
    }
#else
    try {
        img_decoder_session_ = std::make_unique<Ort::Session>(decoderEnv, decoderPath.c_str(), options);
    } catch (const std::exception &e) {
        std::cerr << "[ERROR] initImageDecoder: " << e.what() << "\n";
        return false;
    }
#endif

    return true;
}

bool SAM2_Session::initMemAttention(const std::string &memAttentionPath, int threadsNumber, const std::string &device) {
    std::ifstream f(memAttentionPath.c_str());
    if (!f.good()) {
        std::cerr << "[ERROR] Memory attention model file not found: " << memAttentionPath << "\n";
        return false;
    }
    Ort::SessionOptions options;
    setupSessionOptions(options, threadsNumber, GraphOptimizationLevel::ORT_ENABLE_ALL, device);

#ifdef _WIN32
    std::wstring wPath = strToWstr(memAttentionPath);
    try {
        mem_attention_session_ = std::make_unique<Ort::Session>(memAttentionEnv, wPath.c_str(), options);
    } catch (const std::exception &e) {
        std::cerr << "[ERROR] initMemAttention: " << e.what() << "\n";
        return false;
    }
#else
    try {
        mem_attention_session_ = std::make_unique<Ort::Session>(memAttentionEnv, memAttentionPath.c_str(), options);
    } catch (const std::exception &e) {
        std::cerr << "[ERROR] initMemAttention: " << e.what() << "\n";
        return false;
    }
#endif

    return true;
}

bool SAM2_Session::initMemEncoder(const std::string &memEncoderPath, int threadsNumber, const std::string &device) {
    std::ifstream f(memEncoderPath.c_str());
    if (!f.good()) {
        std::cerr << "[ERROR] Memory encoder model file not found: " << memEncoderPath << "\n";
        return false;
    }
    Ort::SessionOptions options;
    setupSessionOptions(options, threadsNumber, GraphOptimizationLevel::ORT_ENABLE_ALL, device);

#ifdef _WIN32
    std::wstring wPath = strToWstr(memEncoderPath);
    try {
        mem_encoder_session_ = std::make_unique<Ort::Session>(memEncoderEnv, wPath.c_str(), options);
    } catch (const std::exception &e) {
        std::cerr << "[ERROR] initMemEncoder: " << e.what() << "\n";
        return false;
    }
#else
    try {
        mem_encoder_session_ = std::make_unique<Ort::Session>(memEncoderEnv, memEncoderPath.c_str(), options);
    } catch (const std::exception &e) {
        std::cerr << "[ERROR] initMemEncoder: " << e.what() << "\n";
        return false;
    }
#endif

    return true;
}

// Accessor methods
Ort::Session* SAM2_Session::getImgEncoderSession() {
    return img_encoder_session_ ? img_encoder_session_.get() : nullptr;
}

Ort::Session* SAM2_Session::getImgDecoderSession() {
    return img_decoder_session_ ? img_decoder_session_.get() : nullptr;
}

Ort::Session* SAM2_Session::getMemAttentionSession() {
    return mem_attention_session_ ? mem_attention_session_.get() : nullptr;
}

Ort::Session* SAM2_Session::getMemEncoderSession() {
    return mem_encoder_session_ ? mem_encoder_session_.get() : nullptr;
}

// runSession: run the provided session with given input and output names and tensors.
std::variant<std::vector<Ort::Value>, std::string>
SAM2_Session::runSession(Ort::Session* session,
                         const std::vector<const char*>& inputNames,
                         const std::vector<const char*>& outputNames,
                         const std::vector<Ort::Value>& inputTensors,
                         const std::string &debugName) {
    if (!session) {
        return std::string("[ERROR] runSession(" + debugName + "): session is null.");
    }
    try {
        auto outputs = session->Run(
            Ort::RunOptions{nullptr},
            inputNames.data(),
            const_cast<Ort::Value*>(inputTensors.data()),  // ORT requires non-const pointer
            inputTensors.size(),
            outputNames.data(),
            outputNames.size()
        );
        return outputs;
    }
    catch (const std::exception &e) {
        return std::string("[ERROR] runSession(" + debugName + ") => ") + e.what();
    }
}

// Static helper to configure the session options.
void SAM2_Session::setupSessionOptions(Ort::SessionOptions &options, int threadsNumber,
                                       GraphOptimizationLevel optLevel, const std::string &device) {
    options.SetIntraOpNumThreads(threadsNumber);
    options.SetGraphOptimizationLevel(optLevel);

    if (device == "cpu") {
        std::cout << "[DEBUG] Using CPU execution provider." << std::endl;
        int use_arena = 1;
        OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CPU(options, use_arena);
        if (status != nullptr) {
            const char* error_message = Ort::GetApi().GetErrorMessage(status);
            Ort::GetApi().ReleaseStatus(status);
            throw std::runtime_error(std::string("Error appending CPU provider: ") + error_message);
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
        uint32_t coreml_flags = COREML_FLAG_USE_NONE;
        OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CoreML(options, coreml_flags);
        if (status != nullptr) {
            const char* error_message = Ort::GetApi().GetErrorMessage(status);
            Ort::GetApi().ReleaseStatus(status);
            throw std::runtime_error(std::string("Error appending CoreML provider: ") + error_message);
        }
#else
        std::cout << "[WARN] CoreML requested but not supported; defaulting to CPU." << std::endl;
        int use_arena = 1;
        OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CPU(options, use_arena);
        if (status != nullptr) {
            const char* error_message = Ort::GetApi().GetErrorMessage(status);
            Ort::GetApi().ReleaseStatus(status);
            throw std::runtime_error(std::string("Error appending CPU as default: ") + error_message);
        }
#endif
    }
    else {
        std::cout << "[DEBUG] Unknown device (" << device << "); defaulting to CPU." << std::endl;
        int use_arena = 1;
        OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CPU(options, use_arena);
        if (status != nullptr) {
            const char* error_message = Ort::GetApi().GetErrorMessage(status);
            Ort::GetApi().ReleaseStatus(status);
            throw std::runtime_error(std::string("Error appending default CPU provider: ") + error_message);
        }
    }
}
