#ifndef SAM2_SESSION_H
#define SAM2_SESSION_H

#include <onnxruntime_cxx_api.h>
#include <cpu_provider_factory.h>
#ifdef __APPLE__
#include <coreml_provider_factory.h>
#endif

#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <memory>
#include <variant>

// A small helper to hold node names & shapes.
struct Node {
    std::string name;
    std::vector<int64_t> dim;
};

/**
 *  SAM2_Session
 *
 *  A class that manages all the ONNX Runtime sessions (encoder, decoder,
 *  memory-attention, memory-encoder), including the environment, session 
 *  options, node names, shapes, and the actual session->Run(...) calls.
 */
class SAM2_Session {
public:
    SAM2_Session();
    ~SAM2_Session();

    /**
     *  Loads the image encoder + decoder models.
     *
     *  @param encoderPath Path to image-encoder .onnx file.
     *  @param decoderPath Path to image-decoder .onnx file.
     *  @param threadsNumber Number of CPU threads (if using CPU).
     *  @param device Device to run on ("cpu", "cuda:0", "coreml", etc.).
     *  @return True if successfully loaded, false otherwise.
     */
    bool initializeImage(const std::string &encoderPath,
                         const std::string &decoderPath,
                         int threadsNumber,
                         const std::string &device = "cpu");

    /**
     *  Loads the memory-based models (mem-attention + mem-encoder).
     *  This is optional if doing multi-frame tasks.
     */
    bool initializeVideo(const std::string &memAttentionPath,
                         const std::string &memEncoderPath,
                         int threadsNumber,
                         const std::string &device = "cpu");

    /**
     *  Reset / clear all session pointers and shapes.
     *  Automatically called in destructor.
     */
    void clearAllSessions();

    // -------------------
    // Wrappers around session->Run for each sub-model
    // -------------------
    // These return either vector<Ort::Value> on success, or string error message on failure.
    std::variant<std::vector<Ort::Value>, std::string>
        runImageEncoder(const std::vector<Ort::Value> &inputTensors);

    std::variant<std::vector<Ort::Value>, std::string>
        runImageDecoder(const std::vector<Ort::Value> &inputTensors);

    std::variant<std::vector<Ort::Value>, std::string>
        runMemAttention(const std::vector<Ort::Value> &inputTensors);

    std::variant<std::vector<Ort::Value>, std::string>
        runMemEncoder(const std::vector<Ort::Value> &inputTensors);

    /**
     *  Helper to configure session options (threads, device, GPU vs CPU, etc.).
     *  You can reuse your existing logic or adapt as needed.
     */
    static void setupSessionOptions(Ort::SessionOptions &options,
                                    int threadsNumber,
                                    GraphOptimizationLevel optLevel,
                                    const std::string &device);

    // -------------------
    // Optional: Provide a way to query the encoder input shape 
    //           (if needed by your higher-level logic)
    // -------------------
    cv::Size getEncoderInputSize() const;

    // Additional getters if you want to expose shapes for other model outputs:
    // e.g. getEncoderOutputShape(), etc.

public:
    // You may store this as public if needed by your external code,
    // or keep it private and only expose "createTensor" or run methods
    Ort::MemoryInfo memoryInfo_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

private:
    // Helper to do the actual session->Run
    std::variant<std::vector<Ort::Value>, std::string>
        runSession(Ort::Session* session,
                   const std::vector<Node> &inputNodes,
                   const std::vector<Node> &outputNodes,
                   const std::vector<Ort::Value> &inputTensors,
                   const std::string &debugName);

    // Check if a file exists
    bool modelFileExists(const std::string &modelPath);

private:
    // ---------------------
    // ONNXRuntime session pointers
    // ---------------------
    std::unique_ptr<Ort::Session> img_encoder_session_;
    std::unique_ptr<Ort::Session> img_decoder_session_;
    std::unique_ptr<Ort::Session> mem_attention_session_;
    std::unique_ptr<Ort::Session> mem_encoder_session_;

    // ---------------------
    // ORT environment & session options for each
    // (you can share one Env or keep them separate)
    // ---------------------
    Ort::Env env_encoder_{ORT_LOGGING_LEVEL_WARNING, "EncoderEnv"};
    Ort::Env env_decoder_{ORT_LOGGING_LEVEL_WARNING, "DecoderEnv"};
    Ort::Env env_mem_attention_{ORT_LOGGING_LEVEL_WARNING, "MemAttnEnv"};
    Ort::Env env_mem_encoder_{ORT_LOGGING_LEVEL_WARNING, "MemEncEnv"};

    Ort::SessionOptions encoderOptions_;
    Ort::SessionOptions decoderOptions_;
    Ort::SessionOptions memAttentionOptions_;
    Ort::SessionOptions memEncoderOptions_;

    // ---------------------
    // Node info: input & output, shapes, etc.
    // ---------------------
    std::vector<Node> img_encoder_input_nodes_;
    std::vector<Node> img_encoder_output_nodes_;
    std::vector<Node> img_decoder_input_nodes_;
    std::vector<Node> img_decoder_output_nodes_;
    std::vector<Node> mem_attention_input_nodes_;
    std::vector<Node> mem_attention_output_nodes_;
    std::vector<Node> mem_encoder_input_nodes_;
    std::vector<Node> mem_encoder_output_nodes_;

    // Example: store shape for the encoder's first input
    //          typically [1,3,1024,1024]
    std::vector<int64_t> encoderInputShape_;
};

#endif // SAM2_SESSION_H
