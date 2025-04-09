#ifndef SAM2SESSION_H
#define SAM2SESSION_H

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

class SAM2Session {
public:
    SAM2Session();
    ~SAM2Session();

    bool initializeImage(const std::string &encoderPath,
                         const std::string &decoderPath,
                         int threadsNumber,
                         const std::string &device = "cpu");

    bool initializeVideo(const std::string &memAttentionPath,
                         const std::string &memEncoderPath,
                         int threadsNumber,
                         const std::string &device = "cpu");

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

    static void setupSessionOptions(Ort::SessionOptions &options,
                                    int threadsNumber,
                                    GraphOptimizationLevel optLevel,
                                    const std::string &device);

    cv::Size getEncoderInputSize() const;

public:
    Ort::MemoryInfo m_memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

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
    std::unique_ptr<Ort::Session> m_imgEncoderSession;
    std::unique_ptr<Ort::Session> m_imgDecoderSession;
    std::unique_ptr<Ort::Session> m_memAttentionSession;
    std::unique_ptr<Ort::Session> m_memEncoderSession;

    // ---------------------
    // ORT environment & session options for each
    // (you can share one Env or keep them separate)
    // ---------------------
    Ort::Env m_envEncoder{ORT_LOGGING_LEVEL_WARNING, "EncoderEnv"};
    Ort::Env m_envDecoder{ORT_LOGGING_LEVEL_WARNING, "DecoderEnv"};
    Ort::Env m_envMemAttention{ORT_LOGGING_LEVEL_WARNING, "MemAttnEnv"};
    Ort::Env m_envMemEncoder{ORT_LOGGING_LEVEL_WARNING, "MemEncEnv"};

    Ort::SessionOptions m_encoderOptions;
    Ort::SessionOptions m_decoderOptions;
    Ort::SessionOptions m_memAttentionOptions;
    Ort::SessionOptions m_memEncoderOptions;

    // ---------------------
    // Node info: input & output, shapes, etc.
    // ---------------------
    std::vector<Node> m_imgEncoderInputNodes;
    std::vector<Node> m_imgEncoderOutputNodes;
    std::vector<Node> m_imgDecoderInputNodes;
    std::vector<Node> m_imgDecoderOutputNodes;
    std::vector<Node> m_memAttentionInputNodes;
    std::vector<Node> m_memAttentionOutputNodes;
    std::vector<Node> m_memEncoderInputNodes;
    std::vector<Node> m_memEncoderOutputNodes;

    // Example: store shape for the encoder's first input
    //          typically [1,3,1024,1024]
    std::vector<int64_t> m_encoderInputShape;
};

#endif // SAM2SESSION_H
