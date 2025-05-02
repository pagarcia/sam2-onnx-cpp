#ifndef SAMCPP__SAM_H_
#define SAMCPP__SAM_H_

#include <onnxruntime_cxx_api.h>
#include <cpu_provider_factory.h>
#ifdef __APPLE__
#include <coreml_provider_factory.h>
#endif
#include <list>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <variant>
#include "Image.h"

inline size_t computeElementCount(const std::vector<int64_t>& shape)
{
    size_t count = 1;
    for (auto dim : shape) count *= static_cast<size_t>(dim);
    return count;
}

template <typename T>
inline Ort::Value createTensor(const Ort::MemoryInfo &memoryInfo,
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

template <typename T>
inline void extractTensorData(Ort::Value &tensor,
                              std::vector<T> &dataOut,
                              std::vector<int64_t> &shapeOut)
{
    T* p = tensor.GetTensorMutableData<T>();
    auto shape = tensor.GetTensorTypeAndShapeInfo().GetShape();
    size_t count = computeElementCount(shape);
    dataOut.assign(p, p + count);
    shapeOut.assign(shape.begin(), shape.end());
}

struct SAM2Point {
    int x;
    int y;
    SAM2Point(int xVal = 0, int yVal = 0) : x(xVal), y(yVal) {}
};

struct SAM2Rect {
    int x; // x coordinate of the top-left corner
    int y; // y coordinate of the top-left corner
    int width; // width of the rectangle
    int height; // height of the rectangle
    SAM2Rect(int xVal = 0, int yVal = 0, int w = 0, int h = 0) : x(xVal), y(yVal), width(w), height(h) {}
    SAM2Point br() const { return SAM2Point(x + width, y + height); }
};

struct SAM2Prompts {
    std::vector<SAM2Point> points;
    std::vector<int>   pointLabels;
    std::vector<SAM2Rect>  rects;
};

struct SAM2Node {
    std::string name;
    std::vector<int64_t> dim;
};

struct SAM2Size {
    int width;
    int height;
    SAM2Size(int w = 0, int h = 0) : width(w), height(h) {}
};

struct EncoderOutputs {
    std::vector<Ort::Value> outputs;  // All raw outputs from runImageEncoder
    std::vector<float> embedData;
    std::vector<int64_t> embedShape;
    std::vector<float> feats0Data;
    std::vector<int64_t> feats0Shape;
    std::vector<float> feats1Data;
    std::vector<int64_t> feats1Shape;
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

    /// Clear any accumulated memory
    void resetMemory();

    // For single-frame usage:
    EncoderOutputs getEncoderOutputsFromImage(const Image<float> &originalImage, SAM2Size targetImageSize);
    // General helper to prepare decoder inputs.
    // 'primaryFeature' and its shape can be either the encoder embed (frame0)
    // or the fused feature (frameN). 'additionalInputs' allows caller to
    // pass any extra tensors that should come after the first three.
    std::vector<Ort::Value> prepareDecoderInputs(
        const std::vector<float>& promptCoords,
        const std::vector<float>& promptLabels,
        const std::vector<float>& primaryFeature,
        const std::vector<int64_t>& primaryFeatureShape,
        std::vector<Ort::Value> additionalInputs = std::vector<Ort::Value>());

    bool preprocessImage(const Image<float> &originalImage);
    Image<float> inferSingleFrame(const SAM2Size &originalImageSize);

    // For multi-frame usage:
    Image<float> inferMultiFrame(const Image<float> &originalImage,
                                 const SAM2Prompts &prompts);

    // Basic info
    SAM2Size getInputSize();
    bool modelExists(const std::string &modelPath);

    // Prompt and helpers
    void setPrompts(const SAM2Prompts &prompts,
                    const SAM2Size &originalImageSize);
    void setRectsLabels(const std::list<SAM2Rect> &rects,
                        std::vector<float> *inputPointValues,
                        std::vector<float> *inputLabelValues);
    void setPointsLabels(const std::list<SAM2Point> &points,
                         int label,
                         std::vector<float> *inputPointValues,
                         std::vector<float> *inputLabelValues);
    static Image<float> createBinaryMask(const SAM2Size &targetSize,
                                         const SAM2Size &maskSize,
                                         float *maskData,
                                         float threshold = 0.f);
    static Image<float> extractAndCreateMask(Ort::Value &maskTensor, const SAM2Size &targetSize);

    // ORT session config
    static void setupSessionOptions(Ort::SessionOptions &options,
                                    int threadsNumber,
                                    GraphOptimizationLevel optLevel,
                                    const std::string &device);

    static std::vector<SAM2Node> getSessionNodes(Ort::Session* session, bool isInput);

    // Exposed pipeline-step methods (optional advanced usage):
    std::variant<std::vector<Ort::Value>, std::string> runImageEncoderSession(const std::vector<Ort::Value> &inputTensors);
    std::variant<std::vector<Ort::Value>, std::string> runImageDecoderSession(const std::vector<Ort::Value> &inputTensors);
    std::variant<std::vector<Ort::Value>, std::string> runMemAttentionSession(const std::vector<Ort::Value> &inputTensors);
    std::variant<std::vector<Ort::Value>, std::string> runMemEncoderSession(const std::vector<Ort::Value> &inputTensors);

private:
    bool clearSessions();

    // The single helper that calls session->Run(...).
    std::variant<std::vector<Ort::Value>, std::string>
    runSession(Ort::Session* session,
               const std::vector<SAM2Node> &inputNodes,
               const std::vector<SAM2Node> &outputNodes,
               const std::vector<Ort::Value> &inputTensors,
               const std::string &debugName);

private:
    // 1) Single-frame sessions
    std::unique_ptr<Ort::Session> m_imgEncoderSession;
    std::unique_ptr<Ort::Session> m_imgDecoderSession;

    // 2) Multi-frame sessions
    std::unique_ptr<Ort::Session> m_memAttentionSession;
    std::unique_ptr<Ort::Session> m_memEncoderSession;

    // Node info
    std::vector<SAM2Node> m_imgEncoderInputNodes;
    std::vector<SAM2Node> m_imgEncoderOutputNodes;
    std::vector<SAM2Node> m_imgDecoderInputNodes;
    std::vector<SAM2Node> m_imgDecoderOutputNodes;

    // Node info for memory
    std::vector<SAM2Node> m_memAttentionInputNodes;
    std::vector<SAM2Node> m_memAttentionOutputNodes;
    std::vector<SAM2Node> m_memEncoderInputNodes;
    std::vector<SAM2Node> m_memEncoderOutputNodes;

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
};

#endif // SAMCPP__SAM_H_
