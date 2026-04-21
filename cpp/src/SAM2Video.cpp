#include "SAM2.h"

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <vector>

Image<float> SAM2::inferMultiFrameWithEncoderOutputs(std::vector<Ort::Value> &encoderOutputs,
                                                     const SAM2Size &originalSize,
                                                     const SAM2Prompts &prompts,
                                                     double encTimeMs)
{
    if (!m_memAttentionSession || !m_memEncoderSession) {
        std::cerr << "[ERROR] Memory sessions not loaded => did you call initializeVideo()?\n";
        return Image<float>();
    }

    Image<float> blankMask(originalSize.width, originalSize.height, 1);
    double attnTimeMs = 0.0;
    double decTimeMs = 0.0;
    double memEncTimeMs = 0.0;

    if (encoderOutputs.size() <= static_cast<size_t>(std::max({
            m_encoderEmbedIndex,
            m_encoderCurrentVisionFeatIndex,
            m_encoderHighRes0Index,
            m_encoderHighRes1Index,
        }))) {
        std::cerr << "[ERROR] Encoder outputs are missing expected tensors.\n";
        return blankMask;
    }

    setPrompts(prompts, originalSize);

    Ort::Value &encEmbed = encoderOutputs[static_cast<size_t>(m_encoderEmbedIndex)];
    Ort::Value &currVisionFeat = encoderOutputs[static_cast<size_t>(m_encoderCurrentVisionFeatIndex)];
    Ort::Value &feat0 = encoderOutputs[static_cast<size_t>(m_encoderHighRes0Index)];
    Ort::Value &feat1 = encoderOutputs[static_cast<size_t>(m_encoderHighRes1Index)];
    Ort::Value *visionPos = nullptr;
    if (m_encoderVisionPosIndex >= 0
        && encoderOutputs.size() > static_cast<size_t>(m_encoderVisionPosIndex)) {
        visionPos = &encoderOutputs[static_cast<size_t>(m_encoderVisionPosIndex)];
    }

    try {
        if (!m_hasMemory) {
            const int currentFrameIndex = m_videoFrameIndex;
            if (m_promptPointLabels.empty() || m_promptPointCoords.empty()) {
                std::cerr << "[WARN] inferMultiFrame => frame0 has no prompts, returning empty mask.\n";
                return blankMask;
            }

            const auto decStart = std::chrono::steady_clock::now();
            std::vector<Ort::Value> decInputs = buildDecoderInputs(
                m_imgDecoderInputNodes,
                encEmbed,
                feat0,
                feat1,
                &m_promptPointCoords,
                &m_promptPointLabels);
            auto decRes = runImageDecoderSession(decInputs);
            decTimeMs = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - decStart)
                            .count();
            if (decRes.index() == 1) {
                std::cerr << "[ERROR] videoDecoderInit => " << std::get<std::string>(decRes) << '\n';
                return blankMask;
            }

            auto &decOuts = std::get<0>(decRes);
            const auto &initOutputNames = m_imgDecoderOutputNames;
            const int initMaskForMemIndex = findNameIndex(initOutputNames, "mask_for_mem");
            const int initPredMaskIndex = findNameIndex(initOutputNames, "pred_mask");
            if (initMaskForMemIndex < 0 || initPredMaskIndex < 0
                || decOuts.size() <= static_cast<size_t>(std::max(initMaskForMemIndex, initPredMaskIndex))) {
                std::cerr << "[ERROR] videoDecoderInit returned insufficient outputs.\n";
                return blankMask;
            }

            Image<float> finalMask = extractAndCreateMask(
                decOuts[static_cast<size_t>(initPredMaskIndex)],
                originalSize);

            const auto memStart = std::chrono::steady_clock::now();
            std::vector<Ort::Value> memEncInputs = buildMemEncoderInputs(
                decOuts[static_cast<size_t>(initMaskForMemIndex)],
                encEmbed);
            auto memEncRes = runSession(
                m_memEncoderSession.get(),
                m_memEncoderInputNames,
                m_memEncoderStateOutputNames,
                memEncInputs,
                "memoryEncoderFrame0");
            memEncTimeMs = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - memStart)
                               .count();
            if (memEncRes.index() == 1) {
                std::cerr << "[ERROR] memoryEncoderFrame0 => " << std::get<std::string>(memEncRes) << '\n';
                return finalMask;
            }

            m_memoryStateOutputs = std::move(std::get<0>(memEncRes));
            storeConditioningMemory(m_memoryStateOutputs);
            storeConditioningObjectPointer(decOuts, initOutputNames, currentFrameIndex);
            ++m_videoFrameIndex;

            std::cout << "[INFO] Frame0 times => Enc: " << encTimeMs
                      << " ms, Dec: " << decTimeMs
                      << " ms, MemEnc: " << memEncTimeMs << " ms\n";
            return finalMask;
        }

        const int currentFrameIndex = m_videoFrameIndex;
        const auto attnStart = std::chrono::steady_clock::now();
        std::vector<float> emptyMemory0;
        std::vector<Ort::Value> memAttnInputs = buildMemAttentionInputs(currVisionFeat, visionPos, emptyMemory0);
        auto attnRes = runMemAttentionSession(memAttnInputs);
        attnTimeMs = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - attnStart)
                         .count();
        if (attnRes.index() == 1) {
            std::cerr << "[ERROR] memoryAttention => " << std::get<std::string>(attnRes) << '\n';
            return blankMask;
        }

        auto &attnOuts = std::get<0>(attnRes);
        if (attnOuts.empty()) {
            std::cerr << "[ERROR] memoryAttention returned no outputs.\n";
            return blankMask;
        }

        Ort::Value &fusedEmbed = attnOuts[0];

        const auto decStart = std::chrono::steady_clock::now();
        std::vector<Ort::Value> decInputs = buildDecoderInputs(
            getPropDecoderInputNodes(),
            fusedEmbed,
            feat0,
            feat1);
        auto decRes = runVideoPropDecoderSession(decInputs);
        decTimeMs = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - decStart)
                        .count();
        if (decRes.index() == 1) {
            std::cerr << "[ERROR] videoDecoderPropagate => " << std::get<std::string>(decRes) << '\n';
            return blankMask;
        }

        auto &decOuts = std::get<0>(decRes);
        const auto &propOutputNames = getPropDecoderOutputNames();
        const int propMaskForMemIndex = findNameIndex(propOutputNames, "mask_for_mem");
        const int propPredMaskIndex = findNameIndex(propOutputNames, "pred_mask");
        if (propMaskForMemIndex < 0 || propPredMaskIndex < 0
            || decOuts.size() <= static_cast<size_t>(std::max(propMaskForMemIndex, propPredMaskIndex))) {
            std::cerr << "[ERROR] videoDecoderPropagate returned insufficient outputs.\n";
            return blankMask;
        }

        Image<float> finalMask = extractAndCreateMask(
            decOuts[static_cast<size_t>(propPredMaskIndex)],
            originalSize);

        const auto memStart = std::chrono::steady_clock::now();
        std::vector<Ort::Value> memEncInputs = buildMemEncoderInputs(
            decOuts[static_cast<size_t>(propMaskForMemIndex)],
            fusedEmbed);
        auto memEncRes = runSession(
            m_memEncoderSession.get(),
            m_memEncoderInputNames,
            m_memEncoderStateOutputNames,
            memEncInputs,
            "memoryEncoderPropagate");
        memEncTimeMs = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - memStart)
                           .count();
        if (memEncRes.index() == 1) {
            std::cerr << "[ERROR] memoryEncoderPropagate => " << std::get<std::string>(memEncRes) << '\n';
            return finalMask;
        }

        m_memoryStateOutputs = std::move(std::get<0>(memEncRes));
        appendRecentMemory(m_memoryStateOutputs);
        appendRecentObjectPointer(decOuts, propOutputNames, currentFrameIndex);
        ++m_videoFrameIndex;

        std::cout << "[INFO] FrameN times => Enc: " << encTimeMs
                  << " ms, Attn: " << attnTimeMs
                  << " ms, Dec: " << decTimeMs
                  << " ms, MemEnc: " << memEncTimeMs << " ms\n";
        return finalMask;
    }
    catch (const std::exception &e) {
        std::cerr << "[ERROR] inferMultiFrame => " << e.what() << '\n';
        return blankMask;
    }
}

Image<float> SAM2::inferMultiFrame(const Image<float> &originalImage,
                                   const SAM2Prompts &prompts)
{
    const SAM2Size originalSize(originalImage.getWidth(), originalImage.getHeight());
    const SAM2Size targetSize = getInputSize();
    Image<float> blankMask(originalSize.width, originalSize.height, 1);

    EncoderOutputs encOutN;
    double encTimeMs = 0.0;
    {
        const auto start = std::chrono::steady_clock::now();
        try {
            encOutN = getEncoderOutputsFromImage(originalImage, targetSize);
        }
        catch (const std::exception &e) {
            std::cerr << "[ERROR] Encoder failed: " << e.what() << '\n';
            return blankMask;
        }
        encTimeMs = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - start)
                        .count();
    }

    return inferMultiFrameWithEncoderOutputs(encOutN.outputs, originalSize, prompts, encTimeMs);
}

Image<float> SAM2::inferMultiFrameCached(const SAM2Size &originalImageSize,
                                         const SAM2Prompts &prompts)
{
    if (m_cachedEncoderOutputs.empty()) {
        std::cerr << "[ERROR] inferMultiFrameCached => encoder outputs are not cached.\n";
        return Image<float>();
    }

    return inferMultiFrameWithEncoderOutputs(m_cachedEncoderOutputs, originalImageSize, prompts, 0.0);
}
