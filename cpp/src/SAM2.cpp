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
        img_encoder_session.reset();
        img_decoder_session.reset();
        mem_attention_session.reset();
        mem_encoder_session.reset();

        // Clear out shape/data vectors
        inputShapeEncoder.clear();
        outputShapeEncoder.clear();
        highResFeatures1Shape.clear();
        highResFeatures2Shape.clear();

        outputTensorValuesEncoder.clear();
        highResFeatures1.clear();
        highResFeatures2.clear();

        // Reset memory states for multi-frame usage
        hasMemory_ = false;
        maskMemFeatures_.clear();
        maskMemFeaturesShape_.clear();
        maskMemPosEnc_.clear();
        maskMemPosEncShape_.clear();
        temporalCode_.clear();
        temporalCodeShape_.clear();
    }
    catch(...) {
        return false;
    }
    return true;
}

cv::Size SAM2::getInputSize()
{
    // Typically [1,3,1024,1024] => shape[2]=1024 (H), shape[3]=1024 (W)
    if (inputShapeEncoder.size() >= 4) {
        return cv::Size(
            static_cast<int>(inputShapeEncoder[3]),
            static_cast<int>(inputShapeEncoder[2])
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

    if (device != "cpu" && device.rfind("cuda:", 0) == 0) {
        int gpuId = std::stoi(device.substr(5));
        OrtCUDAProviderOptions cudaOpts;
        cudaOpts.device_id = gpuId;
        options.AppendExecutionProvider_CUDA(cudaOpts);
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
    setupSessionOptions(encoderOptions, threadsNumber,
                        GraphOptimizationLevel::ORT_ENABLE_ALL, device);
    setupSessionOptions(decoderOptions, threadsNumber,
                        GraphOptimizationLevel::ORT_ENABLE_ALL, device);

    try {
    #ifdef _WIN32
        std::wstring wEnc = strToWstr(encoderPath);
        std::wstring wDec = strToWstr(decoderPath);

        img_encoder_session = std::make_unique<Ort::Session>(
            encoderEnv, wEnc.c_str(), encoderOptions
        );
        img_decoder_session = std::make_unique<Ort::Session>(
            decoderEnv, wDec.c_str(), decoderOptions
        );
    #else
        img_encoder_session = std::make_unique<Ort::Session>(
            encoderEnv, encoderPath.c_str(), encoderOptions
        );
        img_decoder_session = std::make_unique<Ort::Session>(
            decoderEnv, decoderPath.c_str(), decoderOptions
        );
    #endif

        // Query shapes for the encoder's 3 main outputs
        {
            auto encInputInfo = img_encoder_session
                                  ->GetInputTypeInfo(0)
                                  .GetTensorTypeAndShapeInfo();
            inputShapeEncoder = encInputInfo.GetShape();

            auto out0Info = img_encoder_session->GetOutputTypeInfo(0)
                              .GetTensorTypeAndShapeInfo();
            outputShapeEncoder = out0Info.GetShape();

            auto out1Info = img_encoder_session->GetOutputTypeInfo(1)
                              .GetTensorTypeAndShapeInfo();
            highResFeatures1Shape = out1Info.GetShape();

            auto out2Info = img_encoder_session->GetOutputTypeInfo(2)
                              .GetTensorTypeAndShapeInfo();
            highResFeatures2Shape = out2Info.GetShape();
        }

        // Gather input/output node info for the encoder
        {
            Ort::AllocatorWithDefaultOptions alloc;
            img_encoder_input_nodes.clear();
            size_t inCountEnc = img_encoder_session->GetInputCount();
            for (size_t i = 0; i < inCountEnc; i++) {
                Node node;
                auto inNamePtr = img_encoder_session->GetInputNameAllocated(i, alloc);
                node.name = std::string(inNamePtr.get());
                auto shape = img_encoder_session
                               ->GetInputTypeInfo(i)
                               .GetTensorTypeAndShapeInfo().GetShape();
                node.dim.assign(shape.begin(), shape.end());
                img_encoder_input_nodes.push_back(std::move(node));
            }

            img_encoder_output_nodes.clear();
            size_t outCountEnc = img_encoder_session->GetOutputCount();
            for (size_t i = 0; i < outCountEnc; i++) {
                Node node;
                auto outNamePtr = img_encoder_session->GetOutputNameAllocated(i, alloc);
                node.name = std::string(outNamePtr.get());
                auto shape = img_encoder_session
                               ->GetOutputTypeInfo(i)
                               .GetTensorTypeAndShapeInfo().GetShape();
                node.dim.assign(shape.begin(), shape.end());
                img_encoder_output_nodes.push_back(std::move(node));
            }
        }

        // Gather input/output node info for the decoder
        {
            Ort::AllocatorWithDefaultOptions alloc;
            img_decoder_input_nodes.clear();
            size_t inCountDec = img_decoder_session->GetInputCount();
            for (size_t i = 0; i < inCountDec; i++) {
                Node node;
                auto inNamePtr = img_decoder_session->GetInputNameAllocated(i, alloc);
                node.name = std::string(inNamePtr.get());
                auto shape = img_decoder_session
                               ->GetInputTypeInfo(i)
                               .GetTensorTypeAndShapeInfo().GetShape();
                node.dim.assign(shape.begin(), shape.end());
                img_decoder_input_nodes.push_back(std::move(node));
            }

            img_decoder_output_nodes.clear();
            size_t outCountDec = img_decoder_session->GetOutputCount();
            for (size_t i = 0; i < outCountDec; i++) {
                Node node;
                auto outNamePtr = img_decoder_session->GetOutputNameAllocated(i, alloc);
                node.name = std::string(outNamePtr.get());
                auto shape = img_decoder_session
                               ->GetOutputTypeInfo(i)
                               .GetTensorTypeAndShapeInfo().GetShape();
                node.dim.assign(shape.begin(), shape.end());
                img_decoder_output_nodes.push_back(std::move(node));
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

    setupSessionOptions(memAttentionOptions, threadsNumber,
                        GraphOptimizationLevel::ORT_ENABLE_ALL, device);
    setupSessionOptions(memEncoderOptions, threadsNumber,
                        GraphOptimizationLevel::ORT_ENABLE_ALL, device);

    try {
    #ifdef _WIN32
        std::wstring wAttn = strToWstr(memAttentionPath);
        std::wstring wEnc2 = strToWstr(memEncoderPath);

        mem_attention_session = std::make_unique<Ort::Session>(
            memAttentionEnv, wAttn.c_str(), memAttentionOptions
        );
        mem_encoder_session  = std::make_unique<Ort::Session>(
            memEncoderEnv, wEnc2.c_str(), memEncoderOptions
        );
    #else
        mem_attention_session = std::make_unique<Ort::Session>(
            memAttentionEnv, memAttentionPath.c_str(), memAttentionOptions
        );
        mem_encoder_session  = std::make_unique<Ort::Session>(
            memEncoderEnv, memEncoderPath.c_str(), memEncoderOptions
        );
    #endif

        // gather node info for mem_attention
        {
            Ort::AllocatorWithDefaultOptions alloc;
            mem_attention_input_nodes.clear();
            size_t inCount= mem_attention_session->GetInputCount();
            for(size_t i=0; i<inCount; i++){
                Node node;
                auto inName= mem_attention_session->GetInputNameAllocated(i,alloc);
                node.name= std::string(inName.get());
                auto shape= mem_attention_session
                              ->GetInputTypeInfo(i)
                              .GetTensorTypeAndShapeInfo().GetShape();
                node.dim.assign(shape.begin(), shape.end());
                mem_attention_input_nodes.push_back(std::move(node));
            }
            mem_attention_output_nodes.clear();
            size_t outCount= mem_attention_session->GetOutputCount();
            for(size_t i=0; i<outCount; i++){
                Node node;
                auto outName= mem_attention_session->GetOutputNameAllocated(i,alloc);
                node.name= std::string(outName.get());
                auto shape= mem_attention_session
                              ->GetOutputTypeInfo(i)
                              .GetTensorTypeAndShapeInfo().GetShape();
                node.dim.assign(shape.begin(), shape.end());
                mem_attention_output_nodes.push_back(std::move(node));
            }
        }

        // gather node info for mem_encoder
        {
            Ort::AllocatorWithDefaultOptions alloc;
            mem_encoder_input_nodes.clear();
            size_t inCount= mem_encoder_session->GetInputCount();
            for(size_t i=0; i<inCount; i++){
                Node node;
                auto inName= mem_encoder_session->GetInputNameAllocated(i,alloc);
                node.name= std::string(inName.get());
                auto shape= mem_encoder_session
                             ->GetInputTypeInfo(i)
                             .GetTensorTypeAndShapeInfo().GetShape();
                node.dim.assign(shape.begin(), shape.end());
                mem_encoder_input_nodes.push_back(std::move(node));
            }
            mem_encoder_output_nodes.clear();
            size_t outCount= mem_encoder_session->GetOutputCount();
            for(size_t i=0; i<outCount; i++){
                Node node;
                auto outName= mem_encoder_session->GetOutputNameAllocated(i,alloc);
                node.name= std::string(outName.get());
                auto shape= mem_encoder_session
                             ->GetOutputTypeInfo(i)
                             .GetTensorTypeAndShapeInfo().GetShape();
                node.dim.assign(shape.begin(), shape.end());
                mem_encoder_output_nodes.push_back(std::move(node));
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
            float r_ = pix[2]/255.f;

            data[idx + planeSize*0] = (r_ - MEAN_R) / STD_R;  // R-plane
            data[idx + planeSize*1] = (g  - MEAN_G) / STD_G;  // G-plane
            data[idx + planeSize*2] = (b  - MEAN_B) / STD_B;  // B-plane
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
        Ort::Value inTensor = createTensor<float>(memoryInfo, data, inputShapeEncoder);

        std::vector<Ort::Value> encInputs;
        encInputs.reserve(1);
        encInputs.push_back(std::move(inTensor));

        auto encRes = runImageEncoder(encInputs);
        if(encRes.index() == 1){
            std::cerr << "[ERROR] preprocessImage => "
                      << std::get<std::string>(encRes) << "\n";
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
            for(auto d : outputShapeEncoder) ct *= (size_t)d;
            outputTensorValuesEncoder.assign(p, p+ct);
        }
        {
            float* p = encOuts[1].GetTensorMutableData<float>();
            size_t ct = 1;
            for(auto d : highResFeatures1Shape) ct *= (size_t)d;
            highResFeatures1.assign(p, p+ct);
        }
        {
            float* p = encOuts[2].GetTensorMutableData<float>();
            size_t ct = 1;
            for(auto d : highResFeatures2Shape) ct *= (size_t)d;
            highResFeatures2.assign(p, p+ct);
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
    if(promptPointLabels_.empty() || promptPointCoords_.empty()){
        std::cerr << "[WARN] InferSingleFrame => no prompts.\n";
        return cv::Mat();
    }
    // Build 5 inputs => decode => upsample => final
    int numPoints = (int)promptPointLabels_.size();
    std::vector<int64_t> shpPoints = {1, numPoints, 2};
    std::vector<int64_t> shpLabels = {1, numPoints};

    // We'll push_back carefully
    std::vector<Ort::Value> decInputs;
    decInputs.reserve(5);

    // 0) point_coords
    decInputs.push_back(
        createTensor<float>(memoryInfo, promptPointCoords_, shpPoints)
    );
    // 1) point_labels
    decInputs.push_back(
        createTensor<float>(memoryInfo, promptPointLabels_, shpLabels)
    );
    // 2) image_embed
    decInputs.push_back(
        createTensor<float>(memoryInfo, outputTensorValuesEncoder, outputShapeEncoder)
    );
    // 3) feats_0
    decInputs.push_back(
        createTensor<float>(memoryInfo, highResFeatures1, highResFeatures1Shape)
    );
    // 4) feats_1
    decInputs.push_back(
        createTensor<float>(memoryInfo, highResFeatures2, highResFeatures2Shape)
    );

    auto decRes = runImageDecoder(decInputs);
    if(decRes.index() == 1){
        std::cerr << "[ERROR] InferSingleFrame => decode => "
                  << std::get<std::string>(decRes) << "\n";
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
    promptPointCoords_.clear();
    promptPointLabels_.clear();

    cv::Size encSize = getInputSize();
    if(encSize.width <= 0 || encSize.height <= 0){
        std::cerr << "[WARN] setPrompts => invalid encoder size.\n";
        return;
    }

    // Rect => label=2,3
    for(const auto &rc : prompts.rects){
        float x1 = rc.x * (float)encSize.width / (float)originalImageSize.width;
        float y1 = rc.y * (float)encSize.height / (float)originalImageSize.height;
        promptPointCoords_.push_back(x1);
        promptPointCoords_.push_back(y1);
        promptPointLabels_.push_back(2.f);

        float x2 = rc.br().x * (float)encSize.width / (float)originalImageSize.width;
        float y2 = rc.br().y * (float)encSize.height / (float)originalImageSize.height;
        promptPointCoords_.push_back(x2);
        promptPointCoords_.push_back(y2);
        promptPointLabels_.push_back(3.f);
    }

    // Points => label=1,0,etc.
    for(size_t i=0; i<prompts.points.size(); i++){
        float x = prompts.points[i].x * (float)encSize.width / (float)originalImageSize.width;
        float y = prompts.points[i].y * (float)encSize.height/ (float)originalImageSize.height;
        promptPointCoords_.push_back(x);
        promptPointCoords_.push_back(y);
        promptPointLabels_.push_back((float)prompts.pointLabels[i]);
    }
}

cv::Mat SAM2::InferMultiFrame(const cv::Mat &resizedFrame,
                              const cv::Size &originalSize,
                              const Prompts &prompts)
{
    if (!mem_attention_session || !mem_encoder_session) {
        std::cerr << "[ERROR] mem sessions not loaded => did you call initializeVideo()?\n";
        return cv::Mat();
    }

    // We'll track times for logging
    double encTimeMs = 0.0, attnTimeMs = 0.0, decTimeMs = 0.0, memEncTimeMs = 0.0;

    // -----------
    // If no memory => "Frame 0" approach
    // -----------
    if (!hasMemory_) {
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

        std::vector<int64_t> shapeEnc = {
            1, 3, (int64_t)expected.height, (int64_t)expected.width
        };
        Ort::Value encIn = createTensor<float>(memoryInfo, encData, shapeEnc);

        std::vector<Ort::Value> encInputs;
        encInputs.reserve(1);
        encInputs.push_back(std::move(encIn));

        // 2) runImageEncoder => 5 outputs
        auto encRes = runImageEncoder(encInputs);
        if(encRes.index() == 1) {
            std::cerr << "[ERROR] frame0 => runImageEncoder => "
                      << std::get<std::string>(encRes) << "\n";
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

        int nPts = (int)promptPointLabels_.size();
        std::vector<int64_t> shpPts = {1, nPts, 2};
        std::vector<int64_t> shpLbl = {1, nPts};

        std::vector<Ort::Value> decInputs;
        decInputs.reserve(5);
        decInputs.push_back(
            createTensor<float>(memoryInfo, promptPointCoords_, shpPts)
        );
        decInputs.push_back(
            createTensor<float>(memoryInfo, promptPointLabels_, shpLbl)
        );
        decInputs.push_back(
            createTensor<float>(memoryInfo, embedData, embedShape)
        );
        decInputs.push_back(
            createTensor<float>(memoryInfo, feats0Data, feats0Shape)
        );
        decInputs.push_back(
            createTensor<float>(memoryInfo, feats1Data, feats1Shape)
        );

        auto decRes = runImageDecoder(decInputs);
        if(decRes.index() == 1) {
            std::cerr << "[ERROR] decode => "
                      << std::get<std::string>(decRes) << "\n";
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
            createTensor<float>(memoryInfo, embedData, embedShape)
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
            maskMemFeatures_.assign(p, p+ct);
            maskMemFeaturesShape_.assign(shape.begin(), shape.end());
        }
        {
            float* p = memEncOuts[1].GetTensorMutableData<float>();
            auto shape = memEncOuts[1].GetTensorTypeAndShapeInfo().GetShape();
            size_t ct=1; for(auto d: shape) ct*= (size_t)d;
            maskMemPosEnc_.assign(p, p+ct);
            maskMemPosEncShape_.assign(shape.begin(), shape.end());
        }
        {
            float* p = memEncOuts[2].GetTensorMutableData<float>();
            auto shape = memEncOuts[2].GetTensorTypeAndShapeInfo().GetShape();
            size_t ct=1; for(auto d: shape) ct*= (size_t)d;
            temporalCode_.assign(p, p+ct);
            temporalCodeShape_.assign(shape.begin(), shape.end());
        }

        hasMemory_ = true;

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

        std::vector<int64_t> shapeEnc = {
            1, 3,
            (int64_t)expected.height, (int64_t)expected.width
        };

        Ort::Value encIn = createTensor<float>(memoryInfo, encData, shapeEnc);

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

        // memory_0 => empty object tokens
        {
            std::vector<float> mem0;
            std::vector<int64_t> mem0Shape = {0,256};
            memAttnInputs.push_back(
                createTensor<float>(memoryInfo, mem0, mem0Shape)
            );
        }
        // memory_1 => maskMemFeatures_
        memAttnInputs.push_back(
            createTensor<float>(memoryInfo, maskMemFeatures_, maskMemFeaturesShape_)
        );
        // memory_pos_embed => maskMemPosEnc_
        memAttnInputs.push_back(
            createTensor<float>(memoryInfo, maskMemPosEnc_, maskMemPosEncShape_)
        );

        auto attnRes = runMemAttention(memAttnInputs);
        if(attnRes.index() == 1){
            std::cerr << "[ERROR] memAttn => "
                      << std::get<std::string>(attnRes) << "\n";
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

        int nPts = (int)promptPointLabels_.size();
        std::vector<int64_t> shpPts = {1, nPts, 2};
        std::vector<int64_t> shpLbl = {1, nPts};

        std::vector<Ort::Value> decInputs;
        decInputs.reserve(5);

        decInputs.push_back(
            createTensor<float>(memoryInfo, promptPointCoords_, shpPts)
        );
        decInputs.push_back(
            createTensor<float>(memoryInfo, promptPointLabels_, shpLbl)
        );
        decInputs.push_back(
            createTensor<float>(memoryInfo, fusedVec, fusedShape)
        );
        // feats_0 => encOuts[1], feats_1 => encOuts[2]
        decInputs.push_back(std::move(encOuts[1]));
        decInputs.push_back(std::move(encOuts[2]));

        auto decRes = runImageDecoder(decInputs);
        if(decRes.index() == 1){
            std::cerr << "[ERROR] decode => "
                      << std::get<std::string>(decRes) << "\n";
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
            createTensor<float>(memoryInfo, embedData, embedShape)
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
            maskMemFeatures_.assign(p, p+ct);
            maskMemFeaturesShape_.assign(shape.begin(), shape.end());
        }
        {
            float* p = memEncOuts[1].GetTensorMutableData<float>();
            auto shape = memEncOuts[1].GetTensorTypeAndShapeInfo().GetShape();
            size_t ct=1; for(auto d: shape) ct*= (size_t)d;
            maskMemPosEnc_.assign(p, p+ct);
            maskMemPosEncShape_.assign(shape.begin(), shape.end());
        }
        {
            float* p = memEncOuts[2].GetTensorMutableData<float>();
            auto shape = memEncOuts[2].GetTensorTypeAndShapeInfo().GetShape();
            size_t ct=1; for(auto d: shape) ct*= (size_t)d;
            temporalCode_.assign(p, p+ct);
            temporalCodeShape_.assign(shape.begin(), shape.end());
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
    return runSession(img_encoder_session.get(),
                      img_encoder_input_nodes,
                      img_encoder_output_nodes,
                      inputTensors,
                      "img_encoder_session");
}

std::variant<std::vector<Ort::Value>, std::string>
SAM2::runImageDecoder(const std::vector<Ort::Value> &inputTensors)
{
    return runSession(img_decoder_session.get(),
                      img_decoder_input_nodes,
                      img_decoder_output_nodes,
                      inputTensors,
                      "img_decoder_session");
}

std::variant<std::vector<Ort::Value>, std::string>
SAM2::runMemAttention(const std::vector<Ort::Value> &inputTensors)
{
    return runSession(mem_attention_session.get(),
                      mem_attention_input_nodes,
                      mem_attention_output_nodes,
                      inputTensors,
                      "mem_attention_session");
}

std::variant<std::vector<Ort::Value>, std::string>
SAM2::runMemEncoder(const std::vector<Ort::Value> &inputTensors)
{
    return runSession(mem_encoder_session.get(),
                      mem_encoder_input_nodes,
                      mem_encoder_output_nodes,
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
