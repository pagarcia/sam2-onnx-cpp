#include "SAM2.h"
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <cstring> // for memcpy

// --------------------
// Multi-frame usage
// --------------------
cv::Mat SAM2::inferMultiFrame(const cv::Mat &originalImage,
                              const Prompts &prompts)
{
    if (!m_memAttentionSession || !m_memEncoderSession) {
        std::cerr << "[ERROR] mem sessions not loaded => did you call initializeVideo()?\n";
        return cv::Mat();
    }

    // We'll track times for logging
    double encTimeMs = 0.0, attnTimeMs = 0.0, decTimeMs = 0.0, memEncTimeMs = 0.0;

    Size originalImageSize(originalImage.size().width, originalImage.size().height);
    Size SAM2ImageSize = getInputSize();
    cv::Mat SAM2Image;
    cv::resize(originalImage, SAM2Image, cv::Size(SAM2ImageSize.width, SAM2ImageSize.height));
    std::vector<float> encData = normalizeBGR(SAM2Image);

    // -----------
    // If no memory => "Frame 0" approach
    // -----------
    if (!m_hasMemory) {
        std::cout << "[INFO] InferMultiFrame => no memory => frame0.\n";
        // Similar to single-frame, but the encoder returns 5 outputs.
        auto t0 = std::chrono::steady_clock::now();

        std::vector<int64_t> shapeEnc = { 1, 3, (int64_t)SAM2ImageSize.height, (int64_t)SAM2ImageSize.width };
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
        size_t embedCount = computeElementCount(embedShape);
        size_t feats0Count = computeElementCount(feats0Shape);
        size_t feats1Count = computeElementCount(feats1Shape);

        std::vector<float> embedData(embedPtr,  embedPtr  + embedCount);
        std::vector<float> feats0Data(feats0Ptr, feats0Ptr+ feats0Count);
        std::vector<float> feats1Data(feats1Ptr, feats1Ptr+ feats1Count);

        // decode
        auto tDec0 = std::chrono::steady_clock::now();
        setPrompts(prompts, originalImageSize);

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
        float* pm = decOuts[2].GetTensorMutableData<float>();
        auto pmShape = decOuts[2].GetTensorTypeAndShapeInfo().GetShape();
        int maskHeight = (int)pmShape[2];
        int maskWidth = (int)pmShape[3];
        cv::Mat originalImageSizeBinaryMask = createBinaryMask(originalImageSize, Size(maskWidth, maskHeight), pm);

        // 3) mem-encode => pass decOuts[1] => mask_for_mem + embedData
        auto tMem0 = std::chrono::steady_clock::now();
        std::vector<Ort::Value> memEncInputs;
        memEncInputs.reserve(2);
        memEncInputs.push_back(std::move(decOuts[1])); // mask_for_mem
        memEncInputs.push_back(createTensor<float>(m_memoryInfo, embedData, embedShape));

        auto memEncRes = runMemEncoder(memEncInputs);
        if (memEncRes.index() == 1) {
            std::cerr << "[ERROR] memEncoder => " << std::get<std::string>(memEncRes) << "\n";
            return originalImageSizeBinaryMask;
        }
        auto &memEncOuts = std::get<0>(memEncRes);
        if(memEncOuts.size() < 3){
            std::cerr << "[ERROR] memEncOuts <3.\n";
            return originalImageSizeBinaryMask;
        }
        auto tMem1 = std::chrono::steady_clock::now();
        memEncTimeMs = std::chrono::duration<double, std::milli>(tMem1 - tMem0).count();

        // store memory
        extractTensorData<float>(memEncOuts[0], m_maskMemFeatures, m_maskMemFeaturesShape);
        extractTensorData<float>(memEncOuts[1], m_maskMemPosEnc, m_maskMemPosEncShape);
        extractTensorData<float>(memEncOuts[2], m_temporalCode, m_temporalCodeShape);

        m_hasMemory = true;

        std::cout << "[INFO] Frame0 times => "
                  << "Enc: " << encTimeMs << " ms, "
                  << "Dec: " << decTimeMs << " ms, "
                  << "MemEnc: " << memEncTimeMs << " ms\n";

        return originalImageSizeBinaryMask;
    }
    else {
        // -----------
        // "Frame N" approach => mem-attention + decode + mem-encode
        // -----------
        std::cout << "[INFO] InferMultiFrame => we have memory => frameN.\n";
        auto tEnc0 = std::chrono::steady_clock::now();
        
        std::vector<int64_t> shapeEnc = { 1, 3, (int64_t)SAM2ImageSize.height, (int64_t)SAM2ImageSize.width };

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
        size_t embedCount = computeElementCount(embedShape);
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
        size_t fusedCount = computeElementCount(fusedShape);
        std::vector<float> fusedVec(fusedData, fusedData + fusedCount);

        // 3) decode => set prompts => final mask
        auto tDec0 = std::chrono::steady_clock::now();
        setPrompts(prompts, originalImageSize);

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
        float* pm = decOuts[2].GetTensorMutableData<float>();
        auto pmShape = decOuts[2].GetTensorTypeAndShapeInfo().GetShape();
        int mh = (int)pmShape[2];
        int mw = (int)pmShape[3];
        cv::Mat originalImageSizeBinaryMask = createBinaryMask(originalImageSize, Size(mw, mh), pm);

        // 4) mem-encode => pass decOuts[1] => mask_for_mem + embedData
        auto tMem0 = std::chrono::steady_clock::now();
        std::vector<Ort::Value> memEncInputs;
        memEncInputs.reserve(2);
        memEncInputs.push_back(std::move(decOuts[1]));
        memEncInputs.push_back(createTensor<float>(m_memoryInfo, embedData, embedShape));

        auto memEncRes = runMemEncoder(memEncInputs);
        if(memEncRes.index() == 1){
            std::cerr << "[ERROR] memEncoder => " << std::get<std::string>(memEncRes) << "\n";
            return originalImageSizeBinaryMask;
        }
        auto &memEncOuts = std::get<0>(memEncRes);
        if(memEncOuts.size() < 3){
            std::cerr << "[ERROR] memEncOuts <3.\n";
            return originalImageSizeBinaryMask;
        }
        auto tMem1 = std::chrono::steady_clock::now();
        memEncTimeMs = std::chrono::duration<double,std::milli>(tMem1 - tMem0).count();

        // store memory
        extractTensorData<float>(memEncOuts[0], m_maskMemFeatures, m_maskMemFeaturesShape);
        extractTensorData<float>(memEncOuts[1], m_maskMemPosEnc, m_maskMemPosEncShape);
        extractTensorData<float>(memEncOuts[2], m_temporalCode, m_temporalCodeShape);

        std::cout << "[INFO] FrameN times => "
                  << "Enc: " << encTimeMs << " ms, "
                  << "Attn: " << attnTimeMs << " ms, "
                  << "Dec: " << decTimeMs << " ms, "
                  << "MemEnc: " << memEncTimeMs << " ms\n";

        return originalImageSizeBinaryMask;
    }
}