#include "SAM2.h"
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>
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

cv::Size SAM2::getInputSize()
{
    // Typically [1,3,1024,1024] => shape[2]=1024 (H), shape[3]=1024 (W)
    if (m_inputShapeEncoder.size() >= 4) {
        return cv::Size(
            static_cast<int>(m_inputShapeEncoder[3]),
            static_cast<int>(m_inputShapeEncoder[2])
        );
    }
    return cv::Size(0, 0);
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
// Normalization helper
// --------------------
std::vector<float> SAM2::normalizeBGR(const cv::Mat &bgrImg)
{
    // bgrImg must already match getInputSize() e.g. [1024,1024,3].
    const int H = bgrImg.rows;
    const int W = bgrImg.cols;
    const size_t total = (size_t)1 * 3 * H * W;

    std::vector<float> data(total, 0.f);

    // We'll fill plane [0..2], but in 'RR,GG,BB' order or we can store 'R=plane0,G=plane1,B=plane2'
    for(int r=0; r<H; r++){
        for(int c=0; c<W; c++){
            int idx = r*W + c;
            int planeSize = W * H;
            auto pix = bgrImg.at<cv::Vec3b>(r, c);

            float b = pix[0]/255.f;
            float g = pix[1]/255.f;
            float r = pix[2]/255.f;

            data[idx + planeSize*0] = (r - m_MEAN_R) / m_STD_R;  // R-plane
            data[idx + planeSize*1] = (g - m_MEAN_G) / m_STD_G;  // G-plane
            data[idx + planeSize*2] = (b - m_MEAN_B) / m_STD_B;  // B-plane
        }
    }
    return data;
}

// --------------------
// Single-frame usage
// --------------------
bool SAM2::preprocessImage(const cv::Mat &image)
{
    try {
        cv::Size expected = getInputSize();
        if(image.size() != expected || image.channels() != 3){
            std::cerr << "[WARN] mismatch in preprocessImage.\n";
            return false;
        }

        // Convert BGR to normalized float
        std::vector<float> data = normalizeBGR(image);

        // Create an input tensor
        Ort::Value inTensor = createTensor<float>(m_memoryInfo, data, m_inputShapeEncoder);

        std::vector<Ort::Value> encInputs;
        encInputs.reserve(1);
        encInputs.push_back(std::move(inTensor));

        auto encRes = runImageEncoder(encInputs);
        if(encRes.index() == 1){
            std::cerr << "[ERROR] preprocessImage => " << std::get<std::string>(encRes) << "\n";
            return false;
        }

        auto &encOuts = std::get<0>(encRes);
        if(encOuts.size() < 3){
            std::cerr << "[ERROR] encoder <3 outputs?\n";
            return false;
        }

        // store the 3 relevant outputs
        {
            float* p = encOuts[0].GetTensorMutableData<float>();
            size_t ct = 1;
            for(auto d : m_outputShapeEncoder) ct *= (size_t)d;
            m_outputTensorValuesEncoder.assign(p, p+ct);
        }
        {
            float* p = encOuts[1].GetTensorMutableData<float>();
            size_t ct = 1;
            for(auto d : m_highResFeatures1Shape) ct *= (size_t)d;
            m_highResFeatures1.assign(p, p+ct);
        }
        {
            float* p = encOuts[2].GetTensorMutableData<float>();
            size_t ct = 1;
            for(auto d : m_highResFeatures2Shape) ct *= (size_t)d;
            m_highResFeatures2.assign(p, p+ct);
        }
        return true;
    }
    catch(const std::exception &e){
        std::cerr << "[ERROR] preprocessImage => " << e.what() << "\n";
        return false;
    }
}

cv::Mat SAM2::InferSingleFrame(const cv::Size &originalSize)
{
    if(m_promptPointLabels.empty() || m_promptPointCoords.empty()){
        std::cerr << "[WARN] InferSingleFrame => no prompts.\n";
        return cv::Mat();
    }
    // Build 5 inputs => decode => upsample => final
    int numPoints = (int)m_promptPointLabels.size();
    std::vector<int64_t> shpPoints = {1, numPoints, 2};
    std::vector<int64_t> shpLabels = {1, numPoints};

    // We'll push_back carefully
    std::vector<Ort::Value> decInputs;
    decInputs.reserve(5);

    // 0) point_coords
    decInputs.push_back(createTensor<float>(m_memoryInfo, m_promptPointCoords, shpPoints));
    // 1) point_labels
    decInputs.push_back(createTensor<float>(m_memoryInfo, m_promptPointLabels, shpLabels));
    // 2) image_embed
    decInputs.push_back(createTensor<float>(m_memoryInfo, m_outputTensorValuesEncoder, m_outputShapeEncoder));
    // 3) feats_0
    decInputs.push_back(createTensor<float>(m_memoryInfo, m_highResFeatures1, m_highResFeatures1Shape));
    // 4) feats_1
    decInputs.push_back(createTensor<float>(m_memoryInfo, m_highResFeatures2, m_highResFeatures2Shape));

    auto decRes = runImageDecoder(decInputs);
    if(decRes.index() == 1){
        std::cerr << "[ERROR] InferSingleFrame => decode => " << std::get<std::string>(decRes) << "\n";
        return cv::Mat();
    }
    auto &decOuts = std::get<0>(decRes);
    if(decOuts.size() < 3){
        std::cerr << "[ERROR] decode returned <3 outputs.\n";
        return cv::Mat();
    }
    // decOuts[2] => pred_mask => [1,N,256,256]
    float* pmData = decOuts[2].GetTensorMutableData<float>();
    auto pmShape  = decOuts[2].GetTensorTypeAndShapeInfo().GetShape();
    if(pmShape.size() < 4){
        std::cerr << "[ERROR] pred_mask shape?\n";
        return cv::Mat();
    }
    int maskH = (int)pmShape[2];
    int maskW = (int)pmShape[3];

    cv::Mat lowRes(maskH, maskW, CV_32FC1, (void*)pmData);
    cv::Mat upFloat;
    cv::resize(lowRes, upFloat, originalSize, 0, 0, cv::INTER_LINEAR);

    cv::Mat finalMask(originalSize, CV_8UC1, cv::Scalar(0));
    for(int r=0; r<finalMask.rows; r++){
        const float* rowF = upFloat.ptr<float>(r);
        uchar* rowB      = finalMask.ptr<uchar>(r);
        for(int c=0; c<finalMask.cols; c++){
            rowB[c] = (rowF[c] > 0.f) ? 255 : 0;
        }
    }
    return finalMask;
}

// --------------------
// Multi-frame usage
// --------------------
void SAM2::setPrompts(const Prompts &prompts, const cv::Size &originalImageSize)
{
    m_promptPointCoords.clear();
    m_promptPointLabels.clear();

    cv::Size encSize = getInputSize();
    if(encSize.width <= 0 || encSize.height <= 0){
        std::cerr << "[WARN] setPrompts => invalid encoder size.\n";
        return;
    }

    // Rect => label=2,3
    for(const auto &rc : prompts.rects){
        float x1 = rc.x * (float)encSize.width / (float)originalImageSize.width;
        float y1 = rc.y * (float)encSize.height / (float)originalImageSize.height;
        m_promptPointCoords.push_back(x1);
        m_promptPointCoords.push_back(y1);
        m_promptPointLabels.push_back(2.f);

        float x2 = rc.br().x * (float)encSize.width / (float)originalImageSize.width;
        float y2 = rc.br().y * (float)encSize.height / (float)originalImageSize.height;
        m_promptPointCoords.push_back(x2);
        m_promptPointCoords.push_back(y2);
        m_promptPointLabels.push_back(3.f);
    }

    // Points => label=1,0,etc.
    for(size_t i=0; i<prompts.points.size(); i++){
        float x = prompts.points[i].x * (float)encSize.width / (float)originalImageSize.width;
        float y = prompts.points[i].y * (float)encSize.height/ (float)originalImageSize.height;
        m_promptPointCoords.push_back(x);
        m_promptPointCoords.push_back(y);
        m_promptPointLabels.push_back((float)prompts.pointLabels[i]);
    }
}

cv::Mat SAM2::InferMultiFrame(const cv::Mat &resizedFrame,
                              const cv::Size &originalSize,
                              const Prompts &prompts)
{
    if (!m_memAttentionSession || !m_memEncoderSession) {
        std::cerr << "[ERROR] mem sessions not loaded => did you call initializeVideo()?\n";
        return cv::Mat();
    }

    // We'll track times for logging
    double encTimeMs = 0.0, attnTimeMs = 0.0, decTimeMs = 0.0, memEncTimeMs = 0.0;

    // -----------
    // If no memory => "Frame 0" approach
    // -----------
    if (!m_hasMemory) {
        std::cout << "[INFO] InferMultiFrame => no memory => frame0.\n";
        // Similar to single-frame, but the encoder returns 5 outputs.
        auto t0 = std::chrono::steady_clock::now();

        // 1) normalize
        cv::Size expected = getInputSize();
        if(resizedFrame.size() != expected || resizedFrame.channels() != 3) {
            std::cerr << "[ERROR] frame0 => mismatch input.\n";
            return cv::Mat();
        }
        std::vector<float> encData = normalizeBGR(resizedFrame);

        std::vector<int64_t> shapeEnc = { 1, 3, (int64_t)expected.height, (int64_t)expected.width };
        Ort::Value encIn = createTensor<float>(m_memoryInfo, encData, shapeEnc);

        std::vector<Ort::Value> encInputs;
        encInputs.reserve(1);
        encInputs.push_back(std::move(encIn));

        // 2) runImageEncoder => 5 outputs
        auto encRes = runImageEncoder(encInputs);
        if(encRes.index() == 1) {
            std::cerr << "[ERROR] frame0 => runImageEncoder => " << std::get<std::string>(encRes) << "\n";
            return cv::Mat();
        }
        auto &encOuts = std::get<0>(encRes);
        if(encOuts.size() < 5) {
            std::cerr << "[ERROR] encoder returned <5 for frame0.\n";
            return cv::Mat();
        }
        auto t1 = std::chrono::steady_clock::now();
        encTimeMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // Grab the first 3 for decoding: [0] = image_embed, [1] = feats0, [2] = feats1
        float* embedPtr  = encOuts[0].GetTensorMutableData<float>();
        auto embedShape  = encOuts[0].GetTensorTypeAndShapeInfo().GetShape();

        float* feats0Ptr = encOuts[1].GetTensorMutableData<float>();
        auto feats0Shape = encOuts[1].GetTensorTypeAndShapeInfo().GetShape();

        float* feats1Ptr = encOuts[2].GetTensorMutableData<float>();
        auto feats1Shape = encOuts[2].GetTensorTypeAndShapeInfo().GetShape();

        // clone them for decode usage
        size_t embedCount = 1; for(auto d: embedShape)  embedCount  *= (size_t)d;
        size_t feats0Count=1; for(auto d: feats0Shape) feats0Count *= (size_t)d;
        size_t feats1Count=1; for(auto d: feats1Shape) feats1Count *= (size_t)d;

        std::vector<float> embedData(embedPtr,  embedPtr  + embedCount);
        std::vector<float> feats0Data(feats0Ptr, feats0Ptr+ feats0Count);
        std::vector<float> feats1Data(feats1Ptr, feats1Ptr+ feats1Count);

        // decode
        auto tDec0 = std::chrono::steady_clock::now();
        setPrompts(prompts, originalSize);

        int nPts = (int)m_promptPointLabels.size();
        std::vector<int64_t> shpPts = {1, nPts, 2};
        std::vector<int64_t> shpLbl = {1, nPts};

        std::vector<Ort::Value> decInputs;
        decInputs.reserve(5);
        decInputs.push_back(createTensor<float>(m_memoryInfo, m_promptPointCoords, shpPts));
        decInputs.push_back(createTensor<float>(m_memoryInfo, m_promptPointLabels, shpLbl));
        decInputs.push_back(createTensor<float>(m_memoryInfo, embedData, embedShape));
        decInputs.push_back(createTensor<float>(m_memoryInfo, feats0Data, feats0Shape));
        decInputs.push_back(createTensor<float>(m_memoryInfo, feats1Data, feats1Shape));

        auto decRes = runImageDecoder(decInputs);
        if(decRes.index() == 1) {
            std::cerr << "[ERROR] decode => " << std::get<std::string>(decRes) << "\n";
            return cv::Mat();
        }
        auto &decOuts = std::get<0>(decRes);
        if(decOuts.size() < 3) {
            std::cerr << "[ERROR] decode returned <3.\n";
            return cv::Mat();
        }
        auto tDec1 = std::chrono::steady_clock::now();
        decTimeMs = std::chrono::duration<double, std::milli>(tDec1 - tDec0).count();

        // decOuts[1] = mask_for_mem => [1,1,1024,1024]
        // decOuts[2] = pred_mask    => [1,N,256,256]
        // build final mask
        cv::Mat finalMask;
        {
            float* pm = decOuts[2].GetTensorMutableData<float>();
            auto pmShape = decOuts[2].GetTensorTypeAndShapeInfo().GetShape();
            if(pmShape.size() < 4){
                std::cerr << "[ERROR] pred_mask shape?\n";
                return cv::Mat();
            }
            int mh = (int)pmShape[2];
            int mw = (int)pmShape[3];
            cv::Mat lowRes(mh, mw, CV_32FC1, (void*)pm);
            cv::Mat upFloat;
            cv::resize(lowRes, upFloat, originalSize, 0, 0, cv::INTER_LINEAR);

            finalMask.create(originalSize, CV_8UC1);
            for(int r=0; r<finalMask.rows; r++){
                const float* rowF = upFloat.ptr<float>(r);
                uchar* rowB      = finalMask.ptr<uchar>(r);
                for(int c=0; c<finalMask.cols; c++){
                    rowB[c] = (rowF[c] > 0.f)?255:0;
                }
            }
        }

        // 3) mem-encode => pass decOuts[1] => mask_for_mem + embedData
        auto tMem0 = std::chrono::steady_clock::now();
        std::vector<Ort::Value> memEncInputs;
        memEncInputs.reserve(2);
        memEncInputs.push_back(std::move(decOuts[1])); // mask_for_mem
        memEncInputs.push_back(
            createTensor<float>(m_memoryInfo, embedData, embedShape)
        );

        auto memEncRes = runMemEncoder(memEncInputs);
        if (memEncRes.index() == 1) {
            std::cerr << "[ERROR] memEncoder => "
                      << std::get<std::string>(memEncRes) << "\n";
            return finalMask;
        }
        auto &memEncOuts = std::get<0>(memEncRes);
        if(memEncOuts.size() < 3){
            std::cerr << "[ERROR] memEncOuts <3.\n";
            return finalMask;
        }
        auto tMem1 = std::chrono::steady_clock::now();
        memEncTimeMs = std::chrono::duration<double, std::milli>(tMem1 - tMem0).count();

        // store memory
        {
            float* p = memEncOuts[0].GetTensorMutableData<float>();
            auto shape = memEncOuts[0].GetTensorTypeAndShapeInfo().GetShape();
            size_t ct=1; for(auto d: shape) ct*= (size_t)d;
            m_maskMemFeatures.assign(p, p+ct);
            m_maskMemFeaturesShape.assign(shape.begin(), shape.end());
        }
        {
            float* p = memEncOuts[1].GetTensorMutableData<float>();
            auto shape = memEncOuts[1].GetTensorTypeAndShapeInfo().GetShape();
            size_t ct=1; for(auto d: shape) ct*= (size_t)d;
            m_maskMemPosEnc.assign(p, p+ct);
            m_maskMemPosEncShape.assign(shape.begin(), shape.end());
        }
        {
            float* p = memEncOuts[2].GetTensorMutableData<float>();
            auto shape = memEncOuts[2].GetTensorTypeAndShapeInfo().GetShape();
            size_t ct=1; for(auto d: shape) ct*= (size_t)d;
            m_temporalCode.assign(p, p+ct);
            m_temporalCodeShape.assign(shape.begin(), shape.end());
        }

        m_hasMemory = true;

        std::cout << "[INFO] Frame0 times => "
                  << "Enc: " << encTimeMs << " ms, "
                  << "Dec: " << decTimeMs << " ms, "
                  << "MemEnc: " << memEncTimeMs << " ms\n";

        return finalMask;
    }
    else {
        // -----------
        // "Frame N" approach => mem-attention + decode + mem-encode
        // -----------
        std::cout << "[INFO] InferMultiFrame => we have memory => frameN.\n";
        auto tEnc0 = std::chrono::steady_clock::now();

        cv::Size expected = getInputSize();
        if(resizedFrame.size() != expected || resizedFrame.channels() != 3) {
            std::cerr << "[ERROR] frameN => mismatch input.\n";
            return cv::Mat();
        }
        std::vector<float> encData = normalizeBGR(resizedFrame);

        std::vector<int64_t> shapeEnc = { 1, 3, (int64_t)expected.height, (int64_t)expected.width };

        Ort::Value encIn = createTensor<float>(m_memoryInfo, encData, shapeEnc);

        std::vector<Ort::Value> encInputs;
        encInputs.reserve(1);
        encInputs.push_back(std::move(encIn));

        auto encRes = runImageEncoder(encInputs);
        if(encRes.index() == 1){
            std::cerr << "[ERROR] frameN => encoder => "
                      << std::get<std::string>(encRes) << "\n";
            return cv::Mat();
        }
        auto &encOuts = std::get<0>(encRes);
        if(encOuts.size() < 5){
            std::cerr << "[ERROR] frameN => encoder <5.\n";
            return cv::Mat();
        }
        auto tEnc1 = std::chrono::steady_clock::now();
        encTimeMs = std::chrono::duration<double,std::milli>(tEnc1 - tEnc0).count();

        // => [0] = image_embed [1,256,64,64]
        // => [4] = vision_pos_embed [4096,1,256]

        // Clone image_embed (used later by mem-encoder)
        float* embedPtr = encOuts[0].GetTensorMutableData<float>();
        auto embedShape = encOuts[0].GetTensorTypeAndShapeInfo().GetShape();
        size_t embedCount=1; for(auto d: embedShape) embedCount *= (size_t)d;
        std::vector<float> embedData(embedPtr, embedPtr + embedCount);

        // 2) mem-attention => fuse new embed + old memory
        auto tAttn0 = std::chrono::steady_clock::now();
        std::vector<Ort::Value> memAttnInputs;
        memAttnInputs.reserve(5);

        // current_vision_feat
        memAttnInputs.push_back(std::move(encOuts[0]));
        // vision_pos_embed
        memAttnInputs.push_back(std::move(encOuts[4]));

        // memory0 => empty object tokens
        {
            std::vector<float> mem0;
            std::vector<int64_t> mem0Shape = {0,256};
            memAttnInputs.push_back(createTensor<float>(m_memoryInfo, mem0, mem0Shape));
        }
        // memory1 => m_maskMemFeatures
        memAttnInputs.push_back(createTensor<float>(m_memoryInfo, m_maskMemFeatures, m_maskMemFeaturesShape));
        // memoryPosEmbed => m_maskMemPosEnc
        memAttnInputs.push_back(createTensor<float>(m_memoryInfo, m_maskMemPosEnc, m_maskMemPosEncShape));

        auto attnRes = runMemAttention(memAttnInputs);
        if(attnRes.index() == 1){
            std::cerr << "[ERROR] memAttn => " << std::get<std::string>(attnRes) << "\n";
            return cv::Mat();
        }
        auto &attnOuts = std::get<0>(attnRes);
        if(attnOuts.empty()){
            std::cerr << "[ERROR] memAttn returned empty.\n";
            return cv::Mat();
        }
        auto tAttn1 = std::chrono::steady_clock::now();
        attnTimeMs = std::chrono::duration<double,std::milli>(tAttn1 - tAttn0).count();

        // => fused_feat => [1,256,64,64]
        float* fusedData = attnOuts[0].GetTensorMutableData<float>();
        auto fusedShape  = attnOuts[0].GetTensorTypeAndShapeInfo().GetShape();
        size_t fusedCount = 1; 
        for(auto d : fusedShape) fusedCount *= (size_t)d;
        std::vector<float> fusedVec(fusedData, fusedData + fusedCount);

        // 3) decode => set prompts => final mask
        auto tDec0 = std::chrono::steady_clock::now();
        setPrompts(prompts, originalSize);

        int nPts = (int)m_promptPointLabels.size();
        std::vector<int64_t> shpPts = {1, nPts, 2};
        std::vector<int64_t> shpLbl = {1, nPts};

        std::vector<Ort::Value> decInputs;
        decInputs.reserve(5);

        decInputs.push_back(createTensor<float>(m_memoryInfo, m_promptPointCoords, shpPts));
        decInputs.push_back(createTensor<float>(m_memoryInfo, m_promptPointLabels, shpLbl));
        decInputs.push_back(createTensor<float>(m_memoryInfo, fusedVec, fusedShape));
        // feats0 => encOuts[1], feats1 => encOuts[2]
        decInputs.push_back(std::move(encOuts[1]));
        decInputs.push_back(std::move(encOuts[2]));

        auto decRes = runImageDecoder(decInputs);
        if(decRes.index() == 1){
            std::cerr << "[ERROR] decode => " << std::get<std::string>(decRes) << "\n";
            return cv::Mat();
        }
        auto &decOuts = std::get<0>(decRes);
        if(decOuts.size() < 3){
            std::cerr << "[ERROR] decode returned <3.\n";
            return cv::Mat();
        }
        auto tDec1 = std::chrono::steady_clock::now();
        decTimeMs = std::chrono::duration<double,std::milli>(tDec1 - tDec0).count();

        // => decOuts[1] = mask_for_mem => [1,1,1024,1024]
        // => decOuts[2] = pred_mask    => [1,N,256,256]
        cv::Mat finalMask;
        {
            float* pm = decOuts[2].GetTensorMutableData<float>();
            auto pmShape = decOuts[2].GetTensorTypeAndShapeInfo().GetShape();
            int mh = (int)pmShape[2];
            int mw = (int)pmShape[3];
            cv::Mat lowRes(mh, mw, CV_32FC1, (void*)pm);
            cv::Mat upFloat;
            cv::resize(lowRes, upFloat, originalSize, 0, 0, cv::INTER_LINEAR);

            finalMask.create(originalSize, CV_8UC1);
            for(int r=0; r<finalMask.rows; r++){
                const float* rowF = upFloat.ptr<float>(r);
                uchar* rowB      = finalMask.ptr<uchar>(r);
                for(int c=0; c<finalMask.cols; c++){
                    rowB[c] = (rowF[c] > 0.f)?255:0;
                }
            }
        }

        // 4) mem-encode => pass decOuts[1] => mask_for_mem + embedData
        auto tMem0 = std::chrono::steady_clock::now();
        std::vector<Ort::Value> memEncInputs;
        memEncInputs.reserve(2);
        memEncInputs.push_back(std::move(decOuts[1]));
        memEncInputs.push_back(
            createTensor<float>(m_memoryInfo, embedData, embedShape)
        );

        auto memEncRes = runMemEncoder(memEncInputs);
        if(memEncRes.index() == 1){
            std::cerr << "[ERROR] memEncoder => "
                      << std::get<std::string>(memEncRes) << "\n";
            return finalMask;
        }
        auto &memEncOuts = std::get<0>(memEncRes);
        if(memEncOuts.size() < 3){
            std::cerr << "[ERROR] memEncOuts <3.\n";
            return finalMask;
        }
        auto tMem1 = std::chrono::steady_clock::now();
        memEncTimeMs = std::chrono::duration<double,std::milli>(tMem1 - tMem0).count();

        // store memory
        {
            float* p = memEncOuts[0].GetTensorMutableData<float>();
            auto shape = memEncOuts[0].GetTensorTypeAndShapeInfo().GetShape();
            size_t ct=1; for(auto d: shape) ct*= (size_t)d;
            m_maskMemFeatures.assign(p, p+ct);
            m_maskMemFeaturesShape.assign(shape.begin(), shape.end());
        }
        {
            float* p = memEncOuts[1].GetTensorMutableData<float>();
            auto shape = memEncOuts[1].GetTensorTypeAndShapeInfo().GetShape();
            size_t ct=1; for(auto d: shape) ct*= (size_t)d;
            m_maskMemPosEnc.assign(p, p+ct);
            m_maskMemPosEncShape.assign(shape.begin(), shape.end());
        }
        {
            float* p = memEncOuts[2].GetTensorMutableData<float>();
            auto shape = memEncOuts[2].GetTensorTypeAndShapeInfo().GetShape();
            size_t ct=1; for(auto d: shape) ct*= (size_t)d;
            m_temporalCode.assign(p, p+ct);
            m_temporalCodeShape.assign(shape.begin(), shape.end());
        }

        std::cout << "[INFO] FrameN times => "
                  << "Enc: " << encTimeMs << " ms, "
                  << "Attn: " << attnTimeMs << " ms, "
                  << "Dec: " << decTimeMs << " ms, "
                  << "MemEnc: " << memEncTimeMs << " ms\n";

        return finalMask;
    }
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
                      "img_encoder_session");
}

std::variant<std::vector<Ort::Value>, std::string>
SAM2::runImageDecoder(const std::vector<Ort::Value> &inputTensors)
{
    return runSession(m_imgDecoderSession.get(),
                      m_imgDecoderInputNodes,
                      m_imgDecoderOutputNodes,
                      inputTensors,
                      "img_decoder_session");
}

std::variant<std::vector<Ort::Value>, std::string>
SAM2::runMemAttention(const std::vector<Ort::Value> &inputTensors)
{
    return runSession(m_memAttentionSession.get(),
                      m_memAttentionInputNodes,
                      m_memAttentionOutputNodes,
                      inputTensors,
                      "mem_attention_session");
}

std::variant<std::vector<Ort::Value>, std::string>
SAM2::runMemEncoder(const std::vector<Ort::Value> &inputTensors)
{
    return runSession(m_memEncoderSession.get(),
                      m_memEncoderInputNodes,
                      m_memEncoderOutputNodes,
                      inputTensors,
                      "mem_encoder_session");
}

// --------------------
// Optional label helpers
// --------------------
void SAM2::setRectsLabels(const std::list<cv::Rect> &rects,
                          std::vector<float> *inputPointValues,
                          std::vector<float> *inputLabelValues)
{
    for(const auto &rc : rects) {
        float x1 = (float)rc.x; 
        float y1 = (float)rc.y;
        inputPointValues->push_back(x1);
        inputPointValues->push_back(y1);
        inputLabelValues->push_back(2.f);

        float x2 = (float)rc.br().x; 
        float y2 = (float)rc.br().y;
        inputPointValues->push_back(x2);
        inputPointValues->push_back(y2);
        inputLabelValues->push_back(3.f);
    }
}

void SAM2::setPointsLabels(const std::list<cv::Point> &points,
                           int label,
                           std::vector<float> *inputPointValues,
                           std::vector<float> *inputLabelValues)
{
    for(const auto &pt : points) {
        inputPointValues->push_back((float)pt.x);
        inputPointValues->push_back((float)pt.y);
        inputLabelValues->push_back((float)label);
    }
}
