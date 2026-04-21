// sam2-onnx-cpp/cpp/src/SAM2.h
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
#include <initializer_list>
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
inline Ort::Value createTensorView(const Ort::MemoryInfo &memoryInfo,
                                   T* data,
                                   const std::vector<int64_t> &shape)
{
    return Ort::Value::CreateTensor<T>(
        memoryInfo,
        data,
        computeElementCount(shape),
        shape.data(),
        shape.size()
        );
}

template <typename T>
inline Ort::Value createTensorView(const Ort::MemoryInfo &memoryInfo,
                                   Ort::Value &tensor)
{
    auto shape = tensor.GetTensorTypeAndShapeInfo().GetShape();
    return createTensorView<T>(memoryInfo, tensor.GetTensorMutableData<T>(), shape);
}

template <typename T>
inline void extractTensorData(const Ort::Value &tensor,
                              std::vector<T> &dataOut,
                              std::vector<int64_t> &shapeOut)
{
    const T* p = tensor.GetTensorData<T>();
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
    std::vector<Ort::Value> outputs;
};

struct MemoryFrameState {
    std::vector<float> features;
    std::vector<int64_t> featuresShape;
    std::vector<float> pos;
    std::vector<int64_t> posShape;
};

struct ObjectPointerFrameState {
    std::vector<float> value;
    int frameIndex = -1;
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

    // Loads video modules where frame-0 and propagation may use different decoders
    bool initializeVideo(const std::string &encoderPath,
                         const std::string &decoderInitPath,
                         const std::string &decoderPropPath,
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

    /// Helper to know if ther is NVIDIA driver + at least one CUDA device present
    static bool hasCudaDriver();

    // ORT session config
    static void setupSessionOptions(Ort::SessionOptions &options,
                                    int threadsNumber,
                                    GraphOptimizationLevel optLevel,
                                    const std::string &device);

    static std::vector<SAM2Node> getSessionNodes(Ort::Session* session, bool isInput);

    // Exposed pipeline-step methods (optional advanced usage):
    std::variant<std::vector<Ort::Value>, std::string> runImageEncoderSession(const std::vector<Ort::Value> &inputTensors);
    std::variant<std::vector<Ort::Value>, std::string> runImageDecoderSession(const std::vector<Ort::Value> &inputTensors);
    std::variant<std::vector<Ort::Value>, std::string> runVideoPropDecoderSession(const std::vector<Ort::Value> &inputTensors);
    std::variant<std::vector<Ort::Value>, std::string> runMemAttentionSession(const std::vector<Ort::Value> &inputTensors);
    std::variant<std::vector<Ort::Value>, std::string> runMemEncoderSession(const std::vector<Ort::Value> &inputTensors);

private:
    bool clearSessions();
    static std::vector<const char*> collectNodeNames(const std::vector<SAM2Node> &nodes);
    static std::vector<const char*> selectNodeNames(const std::vector<SAM2Node> &nodes,
                                                    std::initializer_list<const char*> preferredKeys);
    static int findNodeIndex(const std::vector<SAM2Node> &nodes, const std::string &key);
    static int findNameIndex(const std::vector<const char*> &names, const std::string &key);
    static Ort::MemoryInfo cloneMemoryInfo(const Ort::ConstMemoryInfo &memoryInfo);
    static Ort::MemoryInfo cloneTensorMemoryInfo(const Ort::Value &tensor);
    static bool isStaticOptimizableModel(const std::string &modelPath);
    static bool shouldUseCpuArena(const std::string &device);
    bool initializeNamedSession(std::unique_ptr<Ort::Session> *sessionOut,
                                const Ort::Env &env,
                                const std::string &modelPath,
                                const Ort::SessionOptions &options,
                                std::vector<SAM2Node> *inputNodes,
                                std::vector<SAM2Node> *outputNodes,
                                std::vector<const char*> *inputNames,
                                std::vector<const char*> *outputNames,
                                bool skipShapeMetadata = false);
    Ort::Session* getPropDecoderSession();
    const std::vector<SAM2Node>& getPropDecoderInputNodes() const;
    const std::vector<const char*>& getPropDecoderInputNames() const;
    const std::vector<const char*>& getPropDecoderOutputNames() const;
    const std::vector<const char*>& getPropDecoderVideoOutputNames() const;
    MemoryFrameState captureMemoryFrameState(const std::vector<Ort::Value> &memoryOutputs);
    void updateTemporalCode(const std::vector<Ort::Value> &memoryOutputs);
    void trimRecentMemoryFrames();
    void trimRecentObjectPointers();
    void storeConditioningMemory(const std::vector<Ort::Value> &memoryOutputs);
    void appendRecentMemory(const std::vector<Ort::Value> &memoryOutputs);
    void storeConditioningObjectPointer(const std::vector<Ort::Value> &decoderOutputs,
                                        const std::vector<const char*> &outputNames,
                                        int frameIndex);
    void appendRecentObjectPointer(const std::vector<Ort::Value> &decoderOutputs,
                                   const std::vector<const char*> &outputNames,
                                   int frameIndex);
    std::vector<Ort::Value> buildDecoderInputs(const std::vector<SAM2Node> &inputNodes,
                                               Ort::Value &primaryFeature,
                                               Ort::Value &highResFeatures0,
                                               Ort::Value &highResFeatures1,
                                               const std::vector<float> *promptCoords = nullptr,
                                               const std::vector<float> *promptLabels = nullptr);
    std::vector<Ort::Value> buildMemAttentionInputs(Ort::Value &currentVisionFeat,
                                                    Ort::Value *currentVisionPosEmbed,
                                                    const std::vector<float> &emptyMemory0);
    std::vector<Ort::Value> buildMemEncoderInputs(Ort::Value &maskForMem,
                                                  Ort::Value &pixFeat);

    // The single helper that calls session->Run(...).
    std::variant<std::vector<Ort::Value>, std::string>
    runSession(Ort::Session* session,
               const std::vector<const char*> &inputNames,
               const std::vector<const char*> &outputNames,
               const std::vector<Ort::Value> &inputTensors,
               const std::string &debugName);
    std::variant<std::vector<Ort::Value>, std::string>
    runSessionWithOutputMemory(Ort::Session* session,
                               const std::vector<const char*> &inputNames,
                               const std::vector<const char*> &outputNames,
                               const std::vector<Ort::Value> &inputTensors,
                               const std::vector<const Ort::MemoryInfo*> &outputMemoryInfos,
                               const std::string &debugName);

private:
    // 1) Single-frame sessions
    std::unique_ptr<Ort::Session> m_imgEncoderSession;
    std::unique_ptr<Ort::Session> m_imgDecoderSession;
    std::unique_ptr<Ort::Session> m_videoPropDecoderSession;

    // 2) Multi-frame sessions
    std::unique_ptr<Ort::Session> m_memAttentionSession;
    std::unique_ptr<Ort::Session> m_memEncoderSession;

    // Node info
    std::vector<SAM2Node> m_imgEncoderInputNodes;
    std::vector<SAM2Node> m_imgEncoderOutputNodes;
    std::vector<SAM2Node> m_imgDecoderInputNodes;
    std::vector<SAM2Node> m_imgDecoderOutputNodes;
    std::vector<SAM2Node> m_videoPropDecoderInputNodes;
    std::vector<SAM2Node> m_videoPropDecoderOutputNodes;

    // Node info for memory
    std::vector<SAM2Node> m_memAttentionInputNodes;
    std::vector<SAM2Node> m_memAttentionOutputNodes;
    std::vector<SAM2Node> m_memEncoderInputNodes;
    std::vector<SAM2Node> m_memEncoderOutputNodes;

    // Cached C-string names for ORT::Run
    std::vector<const char*> m_imgEncoderInputNames;
    std::vector<const char*> m_imgEncoderOutputNames;
    std::vector<const char*> m_imgDecoderInputNames;
    std::vector<const char*> m_imgDecoderOutputNames;
    std::vector<const char*> m_videoPropDecoderInputNames;
    std::vector<const char*> m_videoPropDecoderOutputNames;
    std::vector<const char*> m_memAttentionInputNames;
    std::vector<const char*> m_memAttentionOutputNames;
    std::vector<const char*> m_memEncoderInputNames;
    std::vector<const char*> m_memEncoderOutputNames;
    std::vector<const char*> m_imgDecoderImageOutputNames;
    std::vector<const char*> m_imgDecoderVideoOutputNames;
    std::vector<const char*> m_videoPropDecoderVideoOutputNames;
    std::vector<const char*> m_memEncoderStateOutputNames;

    // Shapes
    std::vector<int64_t> m_inputShapeEncoder;     // typically [1,3,1024,1024]
    int m_encoderEmbedIndex = -1;
    int m_encoderCurrentVisionFeatIndex = -1;
    int m_encoderHighRes0Index = -1;
    int m_encoderHighRes1Index = -1;
    int m_encoderVisionPosIndex = -1;

    // Stored prompt data (for single-frame decode)
    std::vector<float> m_promptPointCoords;
    std::vector<float> m_promptPointLabels;
    std::vector<float> m_promptPointCoordsScratch;
    std::vector<float> m_promptPointLabelsScratch;

    // Cached encoder outputs for repeated prompting on a single frame
    std::vector<Ort::Value> m_cachedEncoderOutputs;

    // Multi-frame memory
    bool m_hasMemory = false;
    bool m_hasDedicatedPropDecoder = false;
    bool m_memAttentionSingleFrameOnly = false;
    bool m_memAttentionUsesObjectPointers = false;
    size_t m_maxMemoryFrames = 7;
    size_t m_maxObjectPointers = 16;
    int m_videoFrameIndex = 0;
    std::vector<Ort::Value> m_memoryStateOutputs;
    std::vector<MemoryFrameState> m_conditioningMemoryFrames;
    std::vector<MemoryFrameState> m_recentMemoryFrames;
    std::vector<ObjectPointerFrameState> m_conditioningObjectPointers;
    std::vector<ObjectPointerFrameState> m_recentObjectPointers;
    std::vector<float> m_temporalCode;
    std::vector<int64_t> m_temporalCodeShape;
    std::vector<float> m_memoryFeaturesScratch;
    std::vector<int64_t> m_memoryFeaturesShapeScratch;
    std::vector<float> m_memoryPosScratch;
    std::vector<int64_t> m_memoryPosShapeScratch;
    std::vector<float> m_objectPointerScratch;
    std::vector<int64_t> m_objectPointerShapeScratch;
    std::vector<float> m_objectPointerOffsetsScratch;
    std::vector<int64_t> m_objectPointerOffsetsShapeScratch;
    std::vector<float> m_emptyMemory0Scratch;

    // ORT environments
    Ort::Env m_encoderEnv{ORT_LOGGING_LEVEL_WARNING, "img_encoder"};
    Ort::Env m_decoderEnv{ORT_LOGGING_LEVEL_WARNING, "img_decoder"};
    Ort::Env m_videoPropDecoderEnv{ORT_LOGGING_LEVEL_WARNING, "video_prop_decoder"};
    Ort::Env m_memAttentionEnv{ORT_LOGGING_LEVEL_WARNING, "mem_attention"};
    Ort::Env m_memEncoderEnv{ORT_LOGGING_LEVEL_WARNING, "mem_encoder"};

    Ort::MemoryInfo m_memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::MemoryInfo m_cudaMemoryInfo{nullptr};
    bool m_useCudaOutputBinding = false;
    int m_cudaDeviceId = 0;
    std::string m_device = "cpu";
};

#endif // SAMCPP__SAM_H_
