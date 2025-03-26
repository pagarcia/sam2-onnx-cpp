#ifndef SAMCPP__SAM_H_
#define SAMCPP__SAM_H_

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <list>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <variant>

template <typename T>
Ort::Value createTensor(const Ort::MemoryInfo &memoryInfo,
                        const std::vector<T> &data,
                        const std::vector<int64_t> &shape)
{
    return Ort::Value::CreateTensor<T>(
        memoryInfo,
        const_cast<T*>(data.data()),
        data.size(),
        shape.data(),
        shape.size()
    );
}

struct Prompts {
    std::vector<cv::Point> points;
    std::vector<int>       pointLabels;
    std::vector<cv::Rect>  rects;
};

struct Node {
    std::string name;
    std::vector<int64_t> dim;
};

class SAM2 {
public:
    SAM2();
    ~SAM2();

    // Loads ONLY the image encoder + decoder
    bool initialize(const std::string &encoderPath,
                    const std::string &decoderPath,
                    int threadsNumber,
                    std::string device = "cpu");

    // Loads all 4 modules (encoder, decoder, mem-attn, mem-enc)
    bool initializeVideo(const std::string &encoderPath,
                         const std::string &decoderPath,
                         const std::string &memAttentionPath,
                         const std::string &memEncoderPath,
                         int threadsNumber,
                         std::string device = "cpu");

    // For single-frame usage:
    bool preprocessImage(const cv::Mat &image);
    cv::Mat InferSingleFrame(const cv::Size &originalSize);

    // For multi-frame usage:
    void setPrompts(const Prompts &prompts, const cv::Size &originalImageSize);
    cv::Mat InferMultiFrame(const cv::Mat &resizedFrame,
                            const cv::Size &originalSize,
                            const Prompts &prompts);

    // Basic info
    cv::Size getInputSize();
    bool modelExists(const std::string &modelPath);

    // Optional label helpers
    void setRectsLabels(const std::list<cv::Rect> &rects,
                        std::vector<float> *inputPointValues,
                        std::vector<float> *inputLabelValues);
    void setPointsLabels(const std::list<cv::Point> &points,
                         int label,
                         std::vector<float> *inputPointValues,
                         std::vector<float> *inputLabelValues);

    // ORT session config
    static void setupSessionOptions(Ort::SessionOptions &options,
                                    int threadsNumber,
                                    GraphOptimizationLevel optLevel,
                                    const std::string &device);

    // Exposed pipeline-step methods (optional advanced usage):
    std::variant<std::vector<Ort::Value>, std::string>
        runImageEncoder(const std::vector<Ort::Value> &inputTensors);
    std::variant<std::vector<Ort::Value>, std::string>
        runImageDecoder(const std::vector<Ort::Value> &inputTensors);
    std::variant<std::vector<Ort::Value>, std::string>
        runMemAttention(const std::vector<Ort::Value> &inputTensors);
    std::variant<std::vector<Ort::Value>, std::string>
        runMemEncoder(const std::vector<Ort::Value> &inputTensors);

private:
    bool clearSessions();

    // The single helper that calls session->Run(...).
    std::variant<std::vector<Ort::Value>, std::string>
    runSession(Ort::Session* session,
               const std::vector<Node> &inputNodes,
               const std::vector<Node> &outputNodes,
               const std::vector<Ort::Value> &inputTensors,
               const std::string &debugName);

    // Helper for normalizing a BGR image to float[1,3,H,W].
    std::vector<float> normalizeBGR(const cv::Mat &bgrImg);

private:
    // 1) Single-frame sessions
    std::unique_ptr<Ort::Session> img_encoder_session;
    std::unique_ptr<Ort::Session> img_decoder_session;

    // 2) Multi-frame sessions
    std::unique_ptr<Ort::Session> mem_attention_session;
    std::unique_ptr<Ort::Session> mem_encoder_session;

    // Node info
    std::vector<Node> img_encoder_input_nodes;
    std::vector<Node> img_encoder_output_nodes;
    std::vector<Node> img_decoder_input_nodes;
    std::vector<Node> img_decoder_output_nodes;

    // Node info for memory
    std::vector<Node> mem_attention_input_nodes;
    std::vector<Node> mem_attention_output_nodes;
    std::vector<Node> mem_encoder_input_nodes;
    std::vector<Node> mem_encoder_output_nodes;

    // Shapes
    std::vector<int64_t> inputShapeEncoder;     // typically [1,3,1024,1024]
    std::vector<int64_t> outputShapeEncoder;    // e.g. [1,256,64,64]
    std::vector<int64_t> highResFeatures1Shape; // e.g. [1,32,256,256]
    std::vector<int64_t> highResFeatures2Shape; // e.g. [1,64,128,128]

    // Encoder outputs for single-frame usage
    std::vector<float> outputTensorValuesEncoder;  // [1,256,64,64]
    std::vector<float> highResFeatures1;           // [1,32,256,256]
    std::vector<float> highResFeatures2;           // [1,64,128,128]

    // Stored prompt data (for single-frame decode)
    std::vector<float> promptPointCoords_;
    std::vector<float> promptPointLabels_;

    // Multi-frame memory
    bool hasMemory_ = false;
    std::vector<float> maskMemFeatures_;     // e.g. [1,64,64,64]
    std::vector<int64_t> maskMemFeaturesShape_;
    std::vector<float> maskMemPosEnc_;
    std::vector<int64_t> maskMemPosEncShape_;
    std::vector<float> temporalCode_;
    std::vector<int64_t> temporalCodeShape_;

    // ORT environment & options
    Ort::Env encoderEnv{ORT_LOGGING_LEVEL_WARNING, "img_encoder"};
    Ort::Env decoderEnv{ORT_LOGGING_LEVEL_WARNING, "img_decoder"};
    Ort::Env memAttentionEnv{ORT_LOGGING_LEVEL_WARNING, "mem_attention"};
    Ort::Env memEncoderEnv{ORT_LOGGING_LEVEL_WARNING, "mem_encoder"};

    Ort::SessionOptions encoderOptions;
    Ort::SessionOptions decoderOptions;
    Ort::SessionOptions memAttentionOptions;
    Ort::SessionOptions memEncoderOptions;

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Normalization constants
    static constexpr float MEAN_R = 0.485f;
    static constexpr float MEAN_G = 0.456f;
    static constexpr float MEAN_B = 0.406f;
    static constexpr float STD_R  = 0.229f;
    static constexpr float STD_G  = 0.224f;
    static constexpr float STD_B  = 0.225f;
};

#endif // SAMCPP__SAM_H_