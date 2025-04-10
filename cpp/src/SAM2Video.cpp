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
cv::Mat SAM2::inferMultiFrame(const cv::Mat &originalImage, const Prompts &prompts) {
    // Check that the memory sessions are loaded.
    if (!m_memAttentionSession || !m_memEncoderSession) {
        std::cerr << "[ERROR] Memory sessions not loaded => did you call initializeVideo()?\n";
        return cv::Mat();
    }

    // Timing variables.
    double encTimeMs = 0.0, attnTimeMs = 0.0, decTimeMs = 0.0, memEncTimeMs = 0.0;

    // Get the original and expected SAM2 sizes.
    Size origSize(originalImage.cols, originalImage.rows);
    Size SAM2Size = getInputSize();

    // Run the encoder for the current frame.
    EncoderOutputs encOutN;
    {
        auto tEncStart = std::chrono::steady_clock::now();
        try {
            encOutN = runEncoderForImage(originalImage, SAM2Size);
        } catch (const std::exception &e) {
            std::cerr << "[ERROR] Encoder failed: " << e.what() << "\n";
            return cv::Mat();
        }
        auto tEncEnd = std::chrono::steady_clock::now();
        encTimeMs = std::chrono::duration<double, std::milli>(tEncEnd - tEncStart).count();
    }

    // Set the prompts (this step is common to both branches).
    setPrompts(prompts, origSize);

    // --- Branch depending on whether memory is already built or not ---
    if (!m_hasMemory) {
        // ---------- Frame 0: No memory has been set yet ----------
        std::cout << "[INFO] InferMultiFrame => no memory => frame0.\n";

        // Prepare decoder input tensors.
        int nPts = static_cast<int>(m_promptPointLabels.size());
        std::vector<int64_t> shpPts = {1, nPts, 2};
        std::vector<int64_t> shpLbl = {1, nPts};
        std::vector<Ort::Value> decInputs;
        decInputs.reserve(5);
        decInputs.push_back(createTensor<float>(m_memoryInfo, m_promptPointCoords, shpPts));
        decInputs.push_back(createTensor<float>(m_memoryInfo, m_promptPointLabels, shpLbl));
        decInputs.push_back(createTensor<float>(m_memoryInfo, encOutN.embedData, encOutN.embedShape));
        decInputs.push_back(createTensor<float>(m_memoryInfo, encOutN.feats0Data, encOutN.feats0Shape));
        decInputs.push_back(createTensor<float>(m_memoryInfo, encOutN.feats1Data, encOutN.feats1Shape));

        // Run the decoder.
        auto tDecStart = std::chrono::steady_clock::now();
        auto decRes = runImageDecoder(decInputs);
        if (decRes.index() == 1) {
            std::cerr << "[ERROR] decode => " << std::get<std::string>(decRes) << "\n";
            return cv::Mat();
        }
        auto &decOuts = std::get<0>(decRes);
        if (decOuts.size() < 3) {
            std::cerr << "[ERROR] decode returned <3 outputs.\n";
            return cv::Mat();
        }
        auto tDecEnd = std::chrono::steady_clock::now();
        decTimeMs = std::chrono::duration<double, std::milli>(tDecEnd - tDecStart).count();

        // Build the final binary mask from the decoder output (decOuts[2] is pred_mask).
        float* pm = decOuts[2].GetTensorMutableData<float>();
        auto pmShape = decOuts[2].GetTensorTypeAndShapeInfo().GetShape();
        if (pmShape.size() < 4) {
            std::cerr << "[ERROR] pred_mask shape?\n";
            return cv::Mat();
        }
        int maskH = static_cast<int>(pmShape[2]);
        int maskW = static_cast<int>(pmShape[3]);
        cv::Mat finalMask = createBinaryMask(origSize, Size(maskW, maskH), pm);

        // Run the memory encoder to build the memory.
        auto tMemStart = std::chrono::steady_clock::now();
        std::vector<Ort::Value> memEncInputs;
        memEncInputs.reserve(2);
        // Use decOuts[1] as "mask_for_mem" plus the encoder embed data.
        memEncInputs.push_back(std::move(decOuts[1]));
        memEncInputs.push_back(createTensor<float>(m_memoryInfo, encOutN.embedData, encOutN.embedShape));
        auto memEncRes = runMemEncoder(memEncInputs);
        if (memEncRes.index() == 1) {
            std::cerr << "[ERROR] memEncoder => " << std::get<std::string>(memEncRes) << "\n";
            return finalMask;
        }
        auto &memEncOuts = std::get<0>(memEncRes);
        if (memEncOuts.size() < 3) {
            std::cerr << "[ERROR] memEncOuts <3.\n";
            return finalMask;
        }
        auto tMemEnd = std::chrono::steady_clock::now();
        memEncTimeMs = std::chrono::duration<double, std::milli>(tMemEnd - tMemStart).count();

        // Update memory buffers.
        extractTensorData<float>(memEncOuts[0], m_maskMemFeatures, m_maskMemFeaturesShape);
        extractTensorData<float>(memEncOuts[1], m_maskMemPosEnc, m_maskMemPosEncShape);
        extractTensorData<float>(memEncOuts[2], m_temporalCode, m_temporalCodeShape);

        m_hasMemory = true;
        std::cout << "[INFO] Frame0 times => Enc: " << encTimeMs << " ms, Dec: " 
                  << decTimeMs << " ms, MemEnc: " << memEncTimeMs << " ms\n";

        return finalMask;
    } else {
        // ---------- Frame N: Memory has been built already ----------
        std::cout << "[INFO] InferMultiFrame => we have memory => frameN.\n";

        // For frame N, run mem-attention on the current frame's encoder outputs.
        std::vector<Ort::Value> memAttnInputs;
        memAttnInputs.reserve(5);
        // Assume:
        //    outputs[0] is the current vision feature,
        //    outputs[4] is the positional embedding.
        // (Make sure that your encoder always returns at least 5 outputs.)
        memAttnInputs.push_back(std::move(encOutN.outputs[0]));
        memAttnInputs.push_back(std::move(encOutN.outputs[4]));
        // Provide an empty tensor for memory0.
        {
            std::vector<float> emptyMem;
            std::vector<int64_t> emptyShape = {0, 256};
            memAttnInputs.push_back(createTensor<float>(m_memoryInfo, emptyMem, emptyShape));
        }
        // Append stored memory for memory1 and memory positional embeddings.
        memAttnInputs.push_back(createTensor<float>(m_memoryInfo, m_maskMemFeatures, m_maskMemFeaturesShape));
        memAttnInputs.push_back(createTensor<float>(m_memoryInfo, m_maskMemPosEnc, m_maskMemPosEncShape));

        auto tAttnStart = std::chrono::steady_clock::now();
        auto attnRes = runMemAttention(memAttnInputs);
        if (attnRes.index() == 1) {
            std::cerr << "[ERROR] memAttn => " << std::get<std::string>(attnRes) << "\n";
            return cv::Mat();
        }
        auto tAttnEnd = std::chrono::steady_clock::now();
        attnTimeMs = std::chrono::duration<double, std::milli>(tAttnEnd - tAttnStart).count();

        auto &attnOuts = std::get<0>(attnRes);
        if (attnOuts.empty()) {
            std::cerr << "[ERROR] memAttn returned empty.\n";
            return cv::Mat();
        }

        // Build decoder inputs.
        // Use the fused feature from mem-attention as the third input.
        float* fusedData = attnOuts[0].GetTensorMutableData<float>();
        auto fusedShape = attnOuts[0].GetTensorTypeAndShapeInfo().GetShape();
        size_t fusedCount = computeElementCount(fusedShape);
        std::vector<float> fusedVec(fusedData, fusedData + fusedCount);

        int nPts = static_cast<int>(m_promptPointLabels.size());
        std::vector<int64_t> shpPts = {1, nPts, 2};
        std::vector<int64_t> shpLbl = {1, nPts};

        std::vector<Ort::Value> decInputs;
        decInputs.reserve(5);
        decInputs.push_back(createTensor<float>(m_memoryInfo, m_promptPointCoords, shpPts));
        decInputs.push_back(createTensor<float>(m_memoryInfo, m_promptPointLabels, shpLbl));
        decInputs.push_back(createTensor<float>(m_memoryInfo, fusedVec, fusedShape));
        // Use decoder inputs from the encoder’s raw outputs (outputs[1] and outputs[2]).
        decInputs.push_back(std::move(encOutN.outputs[1]));
        decInputs.push_back(std::move(encOutN.outputs[2]));

        auto tDecStart = std::chrono::steady_clock::now();
        auto decRes = runImageDecoder(decInputs);
        if (decRes.index() == 1) {
            std::cerr << "[ERROR] decode => " << std::get<std::string>(decRes) << "\n";
            return cv::Mat();
        }
        auto tDecEnd = std::chrono::steady_clock::now();
        decTimeMs = std::chrono::duration<double, std::milli>(tDecEnd - tDecStart).count();

        auto &decOuts = std::get<0>(decRes);
        if (decOuts.size() < 3) {
            std::cerr << "[ERROR] decode returned <3 outputs.\n";
            return cv::Mat();
        }

        // Create final binary mask from decoder output[2].
        float* pm = decOuts[2].GetTensorMutableData<float>();
        auto pmShape = decOuts[2].GetTensorTypeAndShapeInfo().GetShape();
        int maskH = static_cast<int>(pmShape[2]);
        int maskW = static_cast<int>(pmShape[3]);
        cv::Mat finalMask = createBinaryMask(origSize, Size(maskW, maskH), pm);

        // Run memory encoder to update stored memory.
        auto tMemStart = std::chrono::steady_clock::now();
        std::vector<Ort::Value> memEncInputs;
        memEncInputs.reserve(2);
        // Use decoder output[1] (mask_for_mem) and the encoder embed data.
        memEncInputs.push_back(std::move(decOuts[1]));
        memEncInputs.push_back(createTensor<float>(m_memoryInfo, encOutN.embedData, encOutN.embedShape));
        auto memEncRes = runMemEncoder(memEncInputs);
        if (memEncRes.index() == 1) {
            std::cerr << "[ERROR] memEncoder => " << std::get<std::string>(memEncRes) << "\n";
            return finalMask;
        }
        auto tMemEnd = std::chrono::steady_clock::now();
        memEncTimeMs = std::chrono::duration<double, std::milli>(tMemEnd - tMemStart).count();

        auto &memEncOuts = std::get<0>(memEncRes);
        if (memEncOuts.size() < 3) {
            std::cerr << "[ERROR] memEncOuts <3.\n";
            return finalMask;
        }
        // Update stored memory.
        extractTensorData<float>(memEncOuts[0], m_maskMemFeatures, m_maskMemFeaturesShape);
        extractTensorData<float>(memEncOuts[1], m_maskMemPosEnc, m_maskMemPosEncShape);
        extractTensorData<float>(memEncOuts[2], m_temporalCode, m_temporalCodeShape);

        std::cout << "[INFO] FrameN times => Enc: " << encTimeMs << " ms, "
                  << "Attn: " << attnTimeMs << " ms, Dec: " << decTimeMs << " ms, "
                  << "MemEnc: " << memEncTimeMs << " ms\n";

        return finalMask;
    }
}
