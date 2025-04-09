#ifndef SAMCPP__SAM_H_
#define SAMCPP__SAM_H_

#include <onnxruntime_cxx_api.h>
#include <cpu_provider_factory.h>
#ifdef __APPLE__
#include <coreml_provider_factory.h>
#endif
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

struct Point {
    int x;
    int y;
    Point(int xVal = 0, int yVal = 0) : x(xVal), y(yVal) {}
};

struct Rect {
    int x; // x coordinate of the top-left corner 
    int y; // y coordinate of the top-left corner
    int width; // width of the rectangle 
    int height; // height of the rectangle 
    Rect(int xVal = 0, int yVal = 0, int w = 0, int h = 0) : x(xVal), y(yVal), width(w), height(h) {}
    Point br() const { return Point(x + width, y + height); }
};

struct Prompts {
    std::vector<Point> points;
    std::vector<int>       pointLabels;
    std::vector<Rect>  rects;
};

struct Node {
    std::string name;
    std::vector<int64_t> dim;
};

struct Size {
    int width;
    int height;
    Size(int w = 0, int h = 0) : width(w), height(h) {}
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
    bool preprocessImage(const cv::Mat &originalImage);
    cv::Mat InferSingleFrame(const Size &originalImageSize);

    // For multi-frame usage:
    void setPrompts(const Prompts &prompts, const Size &originalImageSize);
    cv::Mat InferMultiFrame(const cv::Mat &originalImage,
                            const Prompts &prompts);

    // Basic info
    Size getInputSize();
    bool modelExists(const std::string &modelPath);

    // Optional label helpers
    void setRectsLabels(const std::list<Rect> &rects,
                        std::vector<float> *inputPointValues,
                        std::vector<float> *inputLabelValues);
    void setPointsLabels(const std::list<Point> &points,
                         int label,
                         std::vector<float> *inputPointValues,
                         std::vector<float> *inputLabelValues);

    // ORT session config
    static void setupSessionOptions(Ort::SessionOptions &options,
                                    int threadsNumber,
                                    GraphOptimizationLevel optLevel,
                                    const std::string &device);

    // Exposed pipeline-step methods (optional advanced usage):
    std::variant<std::vector<Ort::Value>, std::string> runImageEncoder(const std::vector<Ort::Value> &inputTensors);
    std::variant<std::vector<Ort::Value>, std::string> runImageDecoder(const std::vector<Ort::Value> &inputTensors);
    std::variant<std::vector<Ort::Value>, std::string> runMemAttention(const std::vector<Ort::Value> &inputTensors);
    std::variant<std::vector<Ort::Value>, std::string> runMemEncoder(const std::vector<Ort::Value> &inputTensors);

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
    std::unique_ptr<Ort::Session> m_imgEncoderSession;
    std::unique_ptr<Ort::Session> m_imgDecoderSession;

    // 2) Multi-frame sessions
    std::unique_ptr<Ort::Session> m_memAttentionSession;
    std::unique_ptr<Ort::Session> m_memEncoderSession;

    // Node info
    std::vector<Node> m_imgEncoderInputNodes;
    std::vector<Node> m_imgEncoderOutputNodes;
    std::vector<Node> m_imgDecoderInputNodes;
    std::vector<Node> m_imgDecoderOutputNodes;

    // Node info for memory
    std::vector<Node> m_memAttentionInputNodes;
    std::vector<Node> m_memAttentionOutputNodes;
    std::vector<Node> m_memEncoderInputNodes;
    std::vector<Node> m_memEncoderOutputNodes;

    // Shapes
    std::vector<int64_t> m_inputShapeEncoder;     // typically [1,3,1024,1024]
    std::vector<int64_t> m_outputShapeEncoder;    // e.g. [1,256,64,64]
    std::vector<int64_t> m_highResFeatures1Shape; // e.g. [1,32,256,256]
    std::vector<int64_t> m_highResFeatures2Shape; // e.g. [1,64,128,128]

    // Encoder outputs for single-frame usage
    std::vector<float> m_outputTensorValuesEncoder;  // [1,256,64,64]
    std::vector<float> m_highResFeatures1;           // [1,32,256,256]
    std::vector<float> m_highResFeatures2;           // [1,64,128,128]

    // Stored prompt data (for single-frame decode)
    std::vector<float> m_promptPointCoords;
    std::vector<float> m_promptPointLabels;

    // Multi-frame memory
    bool m_hasMemory = false;
    std::vector<float> m_maskMemFeatures;     // e.g. [1,64,64,64]
    std::vector<int64_t> m_maskMemFeaturesShape;
    std::vector<float> m_maskMemPosEnc;
    std::vector<int64_t> m_maskMemPosEncShape;
    std::vector<float> m_temporalCode;
    std::vector<int64_t> m_temporalCodeShape;

    // ORT environment & options
    Ort::Env m_encoderEnv{ORT_LOGGING_LEVEL_WARNING, "img_encoder"};
    Ort::Env m_decoderEnv{ORT_LOGGING_LEVEL_WARNING, "img_decoder"};
    Ort::Env m_memAttentionEnv{ORT_LOGGING_LEVEL_WARNING, "mem_attention"};
    Ort::Env m_memEncoderEnv{ORT_LOGGING_LEVEL_WARNING, "mem_encoder"};

    Ort::SessionOptions m_encoderOptions;
    Ort::SessionOptions m_decoderOptions;
    Ort::SessionOptions m_memAttentionOptions;
    Ort::SessionOptions m_memEncoderOptions;

    Ort::MemoryInfo m_memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Normalization constants
    static constexpr float m_MEAN_R = 0.485f;
    static constexpr float m_MEAN_G = 0.456f;
    static constexpr float m_MEAN_B = 0.406f;
    static constexpr float m_STD_R  = 0.229f;
    static constexpr float m_STD_G  = 0.224f;
    static constexpr float m_STD_B  = 0.225f;
};

#endif // SAMCPP__SAM_H_