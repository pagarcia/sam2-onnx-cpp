#include "SAM2.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "CVHelpers.h"

EncoderOutputs SAM2::getEncoderOutputsFromImage(const Image<float> &originalImage, SAM2Size targetImageSize)
{
    EncoderOutputs outputs;

    const std::vector<float> encData =
        CVHelpers::resizeImageToPlanarTensor(originalImage, targetImageSize.width, targetImageSize.height);

    Ort::Value inTensor = createTensor<float>(m_memoryInfo, encData, m_inputShapeEncoder);
    std::vector<Ort::Value> encInputs;
    encInputs.reserve(1);
    encInputs.push_back(std::move(inTensor));

    auto encRes = runImageEncoderSession(encInputs);
    if (encRes.index() == 1) {
        throw std::runtime_error(std::get<std::string>(encRes));
    }

    outputs.outputs = std::move(std::get<0>(encRes));
    if (outputs.outputs.size() < 3) {
        throw std::runtime_error("Encoder returned insufficient outputs.");
    }

    return outputs;
}

std::vector<Ort::Value> SAM2::prepareDecoderInputs(
    const std::vector<float>& promptCoords,
    const std::vector<float>& promptLabels,
    const std::vector<float>& primaryFeature,
    const std::vector<int64_t>& primaryFeatureShape,
    std::vector<Ort::Value> additionalInputs)
{
    if (promptCoords.size() != promptLabels.size() * 2) {
        throw std::runtime_error("prepareDecoderInputs: point coordinate and label sizes do not match.");
    }

    if (computeElementCount(primaryFeatureShape) != primaryFeature.size()) {
        throw std::runtime_error("prepareDecoderInputs: primary feature shape does not match data size.");
    }

    std::vector<Ort::Value> inputs;
    const int numPoints = static_cast<int>(promptLabels.size());
    const std::vector<int64_t> shapeCoords = {1, numPoints, 2};
    const std::vector<int64_t> shapeLabels = {1, numPoints};

    inputs.push_back(createTensor<float>(m_memoryInfo, promptCoords, shapeCoords));
    inputs.push_back(createTensor<float>(m_memoryInfo, promptLabels, shapeLabels));
    inputs.push_back(createTensor<float>(m_memoryInfo, primaryFeature, primaryFeatureShape));

    inputs.insert(
        inputs.end(),
        std::make_move_iterator(additionalInputs.begin()),
        std::make_move_iterator(additionalInputs.end()));

    return inputs;
}

bool SAM2::preprocessImage(const Image<float> &originalImage)
{
    try {
        const SAM2Size targetSize = getInputSize();
        EncoderOutputs encOut = getEncoderOutputsFromImage(originalImage, targetSize);
        m_cachedEncoderOutputs = std::move(encOut.outputs);
        return true;
    }
    catch (const std::exception &e) {
        m_cachedEncoderOutputs.clear();
        std::cerr << "[ERROR] preprocessImage => " << e.what() << '\n';
        return false;
    }
}

Image<float> SAM2::inferSingleFrame(const SAM2Size &originalImageSize)
{
    if (m_promptPointLabels.empty() || m_promptPointCoords.empty()) {
        std::cerr << "[WARN] inferSingleFrame => no prompts.\n";
        return Image<float>();
    }

    if (m_cachedEncoderOutputs.size() <= static_cast<size_t>(std::max({m_encoderEmbedIndex, m_encoderHighRes0Index, m_encoderHighRes1Index}))) {
        std::cerr << "[ERROR] inferSingleFrame => encoder outputs are not cached.\n";
        return Image<float>();
    }

    try {
        Ort::Value &embed = m_cachedEncoderOutputs[static_cast<size_t>(m_encoderEmbedIndex)];
        Ort::Value &highRes0 = m_cachedEncoderOutputs[static_cast<size_t>(m_encoderHighRes0Index)];
        Ort::Value &highRes1 = m_cachedEncoderOutputs[static_cast<size_t>(m_encoderHighRes1Index)];

        std::vector<Ort::Value> decInputs = buildDecoderInputs(
            m_imgDecoderInputNodes,
            embed,
            highRes0,
            highRes1,
            &m_promptPointCoords,
            &m_promptPointLabels);

        auto decRes = runSession(
            m_imgDecoderSession.get(),
            m_imgDecoderInputNames,
            m_imgDecoderImageOutputNames,
            decInputs,
            "imgDecoderImage");
        if (decRes.index() == 1) {
            std::cerr << "[ERROR] inferSingleFrame => decode => " << std::get<std::string>(decRes) << '\n';
            return Image<float>();
        }

        auto &decOuts = std::get<0>(decRes);
        if (decOuts.empty()) {
            std::cerr << "[ERROR] inferSingleFrame => decoder returned no outputs.\n";
            return Image<float>();
        }

        return extractAndCreateMask(decOuts.back(), originalImageSize);
    }
    catch (const std::exception &e) {
        std::cerr << "[ERROR] inferSingleFrame => " << e.what() << '\n';
        return Image<float>();
    }
}

void SAM2::setPrompts(const SAM2Prompts &prompts,
                      const SAM2Size &originalImageSize)
{
    m_promptPointCoords.clear();
    m_promptPointLabels.clear();

    const SAM2Size encSize = getInputSize();
    if (encSize.width <= 0 || encSize.height <= 0) {
        std::cerr << "[WARN] setPrompts => invalid encoder size.\n";
        return;
    }

    if (originalImageSize.width <= 0 || originalImageSize.height <= 0) {
        std::cerr << "[WARN] setPrompts => invalid original image size.\n";
        return;
    }

    for (const auto &rawRect : prompts.rects) {
        SAM2Rect rect = rawRect;
        if (rect.width < 0) {
            rect.x += rect.width;
            rect.width = -rect.width;
        }
        if (rect.height < 0) {
            rect.y += rect.height;
            rect.height = -rect.height;
        }

        const float x1 = rect.x * static_cast<float>(encSize.width) / originalImageSize.width;
        const float y1 = rect.y * static_cast<float>(encSize.height) / originalImageSize.height;
        const float x2 = (rect.x + rect.width) * static_cast<float>(encSize.width) / originalImageSize.width;
        const float y2 = (rect.y + rect.height) * static_cast<float>(encSize.height) / originalImageSize.height;

        m_promptPointCoords.push_back(x1);
        m_promptPointCoords.push_back(y1);
        m_promptPointLabels.push_back(2.0f);

        m_promptPointCoords.push_back(x2);
        m_promptPointCoords.push_back(y2);
        m_promptPointLabels.push_back(3.0f);
    }

    if (prompts.points.size() != prompts.pointLabels.size()) {
        std::cerr << "[WARN] setPrompts => points (" << prompts.points.size()
                  << ") and labels (" << prompts.pointLabels.size()
                  << ") differ. Truncating to the shortest length.\n";
    }

    const size_t pointCount = std::min(prompts.points.size(), prompts.pointLabels.size());
    for (size_t i = 0; i < pointCount; ++i) {
        const float x = prompts.points[i].x * static_cast<float>(encSize.width) / originalImageSize.width;
        const float y = prompts.points[i].y * static_cast<float>(encSize.height) / originalImageSize.height;
        m_promptPointCoords.push_back(x);
        m_promptPointCoords.push_back(y);
        m_promptPointLabels.push_back(static_cast<float>(prompts.pointLabels[i]));
    }
}

void SAM2::setRectsLabels(const std::list<SAM2Rect> &rects,
                          std::vector<float> *inputPointValues,
                          std::vector<float> *inputLabelValues)
{
    for (const auto &rc : rects) {
        inputPointValues->push_back(static_cast<float>(rc.x));
        inputPointValues->push_back(static_cast<float>(rc.y));
        inputLabelValues->push_back(2.0f);

        inputPointValues->push_back(static_cast<float>(rc.br().x));
        inputPointValues->push_back(static_cast<float>(rc.br().y));
        inputLabelValues->push_back(3.0f);
    }
}

void SAM2::setPointsLabels(const std::list<SAM2Point> &points,
                           int label,
                           std::vector<float> *inputPointValues,
                           std::vector<float> *inputLabelValues)
{
    for (const auto &pt : points) {
        inputPointValues->push_back(static_cast<float>(pt.x));
        inputPointValues->push_back(static_cast<float>(pt.y));
        inputLabelValues->push_back(static_cast<float>(label));
    }
}

Image<float> SAM2::createBinaryMask(const SAM2Size &targetSize,
                                    const SAM2Size &maskSize,
                                    float *maskData,
                                    float threshold)
{
    return CVHelpers::resizeAndThresholdMask(
        maskData,
        maskSize.width,
        maskSize.height,
        targetSize.width,
        targetSize.height,
        threshold);
}

Image<float> SAM2::extractAndCreateMask(Ort::Value &maskTensor, const SAM2Size &targetSize)
{
    float* maskData = maskTensor.GetTensorMutableData<float>();
    const auto maskShape = maskTensor.GetTensorTypeAndShapeInfo().GetShape();

    if (maskShape.size() < 4) {
        std::cerr << "[ERROR] extractAndCreateMask => unexpected mask shape.\n";
        return Image<float>();
    }

    const int maskH = static_cast<int>(maskShape[2]);
    const int maskW = static_cast<int>(maskShape[3]);
    return createBinaryMask(targetSize, SAM2Size(maskW, maskH), maskData);
}
