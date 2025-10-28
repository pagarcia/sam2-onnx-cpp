// sam2-onnx-cpp/cpp/src/SAM2Video.cpp
#include "SAM2.h"
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <cstring>

// --------------------
// Multi-frame usage
// --------------------
Image<float> SAM2::inferMultiFrame(const Image<float> &originalImage, const SAM2Prompts &prompts) {
    // Check that the memory sessions are loaded.
    if (!m_memAttentionSession || !m_memEncoderSession) {
        std::cerr << "[ERROR] Memory sessions not loaded => did you call initializeVideo()?\n";
        return Image<float>();
    }

    // Timing variables.
    double encTimeMs = 0.0, attnTimeMs = 0.0, decTimeMs = 0.0, memEncTimeMs = 0.0;

    // Get the original and expected SAM2 sizes.
    SAM2Size origSize(originalImage.getWidth(), originalImage.getHeight());
    SAM2Size targetSize = getInputSize();

    // Run the encoder for the current frame.
    EncoderOutputs encOutN;
    {
        auto tEncStart = std::chrono::steady_clock::now();
        try {
            encOutN = getEncoderOutputsFromImage(originalImage, targetSize);
        } catch (const std::exception &e) {
            std::cerr << "[ERROR] Encoder failed: " << e.what() << "\n";
            return Image<float>();
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

        // For frame0: create additional inputs for feats0 and feats1.
        std::vector<Ort::Value> additionalInputsFrame0;
        additionalInputsFrame0.push_back(std::move(createTensor<float>(m_memoryInfo, encOutN.feats0Data, encOutN.feats0Shape)));
        additionalInputsFrame0.push_back(std::move(createTensor<float>(m_memoryInfo, encOutN.feats1Data, encOutN.feats1Shape)));

        // Now call the helper; here primary feature is the encoder embed.
        std::vector<Ort::Value> decInputs = prepareDecoderInputs(
            m_promptPointCoords,
            m_promptPointLabels,
            encOutN.embedData,      // primary feature for frame0
            encOutN.embedShape,
            std::move(additionalInputsFrame0)  // additional inputs (feats0 and feats1)
            );

        // Run the decoder.
        auto tDecStart = std::chrono::steady_clock::now();
        auto decRes = runImageDecoderSession(decInputs);
        if (decRes.index() == 1) {
            std::cerr << "[ERROR] decode => " << std::get<std::string>(decRes) << "\n";
            return Image<float>();
        }
        auto &decOuts = std::get<0>(decRes);
        if (decOuts.size() < 3) {
            std::cerr << "[ERROR] decode returned <3 outputs.\n";
            return Image<float>();
        }
        auto tDecEnd = std::chrono::steady_clock::now();
        decTimeMs = std::chrono::duration<double, std::milli>(tDecEnd - tDecStart).count();

        // Build the final binary mask from the decoder output (decOuts[2] is pred_mask).
        Image<float> finalMask = extractAndCreateMask(decOuts[2], origSize);

        // Run the memory encoder to build the memory.
        auto tMemStart = std::chrono::steady_clock::now();
        std::vector<Ort::Value> memEncInputs;
        memEncInputs.reserve(2);
        // Use decOuts[1] as "mask_for_mem" plus the encoder embed data.
        memEncInputs.push_back(std::move(decOuts[1]));
        memEncInputs.push_back(createTensor<float>(m_memoryInfo, encOutN.embedData, encOutN.embedShape));
        auto memEncRes = runMemEncoderSession(memEncInputs);
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
        auto attnRes = runMemAttentionSession(memAttnInputs);
        if (attnRes.index() == 1) {
            std::cerr << "[ERROR] memAttn => " << std::get<std::string>(attnRes) << "\n";
            return Image<float>();
        }
        auto tAttnEnd = std::chrono::steady_clock::now();
        attnTimeMs = std::chrono::duration<double, std::milli>(tAttnEnd - tAttnStart).count();

        auto &attnOuts = std::get<0>(attnRes);
        if (attnOuts.empty()) {
            std::cerr << "[ERROR] memAttn returned empty.\n";
            return Image<float>();
        }

        // Build decoder inputs.
        // Use the fused feature from mem-attention as the third input.
        float* fusedData = attnOuts[0].GetTensorMutableData<float>();
        auto fusedShape = attnOuts[0].GetTensorTypeAndShapeInfo().GetShape();
        size_t fusedCount = computeElementCount(fusedShape);
        std::vector<float> fusedVec(fusedData, fusedData + fusedCount);

        std::vector<Ort::Value> additionalInputsFrameN;
        // For frameN, the remaining inputs are provided directly (without re-wrapping them via createTensor).
        additionalInputsFrameN.push_back(std::move(encOutN.outputs[1]));
        additionalInputsFrameN.push_back(std::move(encOutN.outputs[2]));

        // Now call the helper; here primary feature is the fused feature.
        std::vector<Ort::Value> decInputs = prepareDecoderInputs(
            m_promptPointCoords,
            m_promptPointLabels,
            fusedVec,         // primary feature for frameN (fused feature)
            fusedShape,
            std::move(additionalInputsFrameN)  // additional inputs (moved values from encoder outputs)
            );

        auto tDecStart = std::chrono::steady_clock::now();
        auto decRes = runImageDecoderSession(decInputs);
        if (decRes.index() == 1) {
            std::cerr << "[ERROR] decode => " << std::get<std::string>(decRes) << "\n";
            return Image<float>();
        }
        auto tDecEnd = std::chrono::steady_clock::now();
        decTimeMs = std::chrono::duration<double, std::milli>(tDecEnd - tDecStart).count();

        auto &decOuts = std::get<0>(decRes);
        if (decOuts.size() < 3) {
            std::cerr << "[ERROR] decode returned <3 outputs.\n";
            return Image<float>();
        }

        // Create final binary mask from decoder output[2].
        Image<float> finalMask = extractAndCreateMask(decOuts[2], origSize);

        // Run memory encoder to update stored memory.
        auto tMemStart = std::chrono::steady_clock::now();
        std::vector<Ort::Value> memEncInputs;
        memEncInputs.reserve(2);
        // Use decoder output[1] (mask_for_mem) and the encoder embed data.
        memEncInputs.push_back(std::move(decOuts[1]));
        memEncInputs.push_back(createTensor<float>(m_memoryInfo, encOutN.embedData, encOutN.embedShape));
        auto memEncRes = runMemEncoderSession(memEncInputs);
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