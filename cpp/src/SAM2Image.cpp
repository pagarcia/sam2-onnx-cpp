// sam2-onnx-cpp/cpp/src/SAM2Image.cpp
#include "SAM2.h"
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <cstring>

// --------------------
// Single-frame usage
// --------------------
EncoderOutputs SAM2::getEncoderOutputsFromImage(const Image<float> &originalImage, SAM2Size targetImageSize) {
    EncoderOutputs outputs;

    Image<float> resizedImage = originalImage.resize(targetImageSize.width, targetImageSize.height);

    // Extract image in planar format (all r values first, then g values then b values)
    // SAM2 apparently needs them in this order
    std::vector<float> encData = resizedImage.getDataPlanarFormat();

    // Create an input tensor using the expected encoder input shape.
    Ort::Value inTensor = createTensor<float>(m_memoryInfo, encData, m_inputShapeEncoder);
    std::vector<Ort::Value> encInputs;
    encInputs.reserve(1);
    encInputs.push_back(std::move(inTensor));

    // Run the encoder.
    auto encRes = runImageEncoderSession(encInputs);
    if(encRes.index() == 1)
        throw std::runtime_error(std::get<std::string>(encRes));
    outputs.outputs = std::move(std::get<0>(encRes));

    if(outputs.outputs.size() < 3)
        throw std::runtime_error("Encoder returned insufficient outputs.");

    // Extract the first three outputs.
    extractTensorData<float>(outputs.outputs[0], outputs.embedData, outputs.embedShape);
    extractTensorData<float>(outputs.outputs[1], outputs.feats0Data, outputs.feats0Shape);
    extractTensorData<float>(outputs.outputs[2], outputs.feats1Data, outputs.feats1Shape);

    return outputs;
}

std::vector<Ort::Value> SAM2::prepareDecoderInputs(
    const std::vector<float>& promptCoords,
    const std::vector<float>& promptLabels,
    const std::vector<float>& primaryFeature,
    const std::vector<int64_t>& primaryFeatureShape,
    std::vector<Ort::Value> additionalInputs)
{
    std::vector<Ort::Value> inputs;
    // Compute shapes for the prompt inputs.
    int numPoints = static_cast<int>(promptLabels.size());
    std::vector<int64_t> shapeCoords = {1, numPoints, 2};
    std::vector<int64_t> shapeLabels = {1, numPoints};

    // Create prompt coordinate tensor.
    inputs.push_back(createTensor<float>(m_memoryInfo, promptCoords, shapeCoords));
    // Create prompt label tensor.
    inputs.push_back(createTensor<float>(m_memoryInfo, promptLabels, shapeLabels));
    // Create primary feature tensor (encoder embed or fused feature).
    inputs.push_back(createTensor<float>(m_memoryInfo, primaryFeature, primaryFeatureShape));

    // Instead of a range-for loop, use move iterators to move the additional inputs.
    inputs.insert(inputs.end(),
                  std::make_move_iterator(additionalInputs.begin()),
                  std::make_move_iterator(additionalInputs.end()));

    return inputs;
}

bool SAM2::preprocessImage(const Image<float> &originalImage)
{
    try {
        SAM2Size SAM2ImageSize = getInputSize();
        EncoderOutputs encOut = getEncoderOutputsFromImage(originalImage, SAM2ImageSize);

        // Store the extracted outputs in the appropriate member variables.
        m_outputTensorValuesEncoder = encOut.embedData;
        m_outputShapeEncoder = encOut.embedShape;
        m_highResFeatures1 = encOut.feats0Data;
        m_highResFeatures1Shape = encOut.feats0Shape;
        m_highResFeatures2 = encOut.feats1Data;
        m_highResFeatures2Shape = encOut.feats1Shape;

        return true;
    }
    catch(const std::exception &e){
        std::cerr << "[ERROR] preprocessImage => " << e.what() << "\n";
        return false;
    }
}

Image<float> SAM2::inferSingleFrame(const SAM2Size &originalImageSize)
{
    if(m_promptPointLabels.empty() || m_promptPointCoords.empty()){
        std::cerr << "[WARN] InferSingleFrame => no prompts.\n";
        return Image<float>();
    }

    // Build additional inputs for feats0 and feats1.
    std::vector<Ort::Value> additionalInputs;
    additionalInputs.push_back(std::move(createTensor<float>(m_memoryInfo, m_highResFeatures1, m_highResFeatures1Shape)));
    additionalInputs.push_back(std::move(createTensor<float>(m_memoryInfo, m_highResFeatures2, m_highResFeatures2Shape)));

    // Use m_outputTensorValuesEncoder and m_outputShapeEncoder as the primary feature.
    // The helper builds the point coordinate tensor, point label tensor, and primary feature tensor.
    std::vector<Ort::Value> decInputs = prepareDecoderInputs(
        m_promptPointCoords,
        m_promptPointLabels,
        m_outputTensorValuesEncoder,
        m_outputShapeEncoder,
        std::move(additionalInputs)
        );

    // Run the decoder
    auto decRes = runImageDecoderSession(decInputs);
    if(decRes.index() == 1){
        std::cerr << "[ERROR] InferSingleFrame => decode => " << std::get<std::string>(decRes) << "\n";
        return Image<float>();
    }
    auto &decOuts = std::get<0>(decRes);
    if(decOuts.size() < 3){
        std::cerr << "[ERROR] decode returned <3 outputs.\n";
        return Image<float>();
    }
    // decOuts[2] => pred_mask => [1,N,256,256]
    // Create a low-resolution floar mask image from the decoder output and convert it to binary uchar mask
    Image<float> originalImageSizeBinaryMask = extractAndCreateMask(decOuts[2], originalImageSize);
    return originalImageSizeBinaryMask;
}

// ------------------
// Prompt and helpers
// ------------------
void SAM2::setPrompts(const SAM2Prompts &prompts,
                      const SAM2Size    &originalImageSize)
{
    // ── clear previous prompt buffers ─────────────────────────────
    m_promptPointCoords.clear();
    m_promptPointLabels.clear();

    // encoder input size in pixels (e.g. 1024×1024)
    SAM2Size encSize = getInputSize();
    if (encSize.width <= 0 || encSize.height <= 0) {
        std::cerr << "[WARN] setPrompts => invalid encoder size.\n";
        return;
    }

    /*───────────────────────────────────────────────────────────────
      1.  Rectangles  →  two corner points  (labels 2 & 3)
    ───────────────────────────────────────────────────────────────*/
    for (const auto &rcRaw : prompts.rects)
    {
        // make a mutable copy so we can normalise it
        SAM2Rect rc = rcRaw;

        // normalise: ensure (x,y) is top-left; width/height positive
        if (rc.width  < 0) { rc.x += rc.width;  rc.width  = -rc.width; }
        if (rc.height < 0) { rc.y += rc.height; rc.height = -rc.height; }

        // scale TL corner → encoder space, push with label 2
        float x1 = rc.x *  (float)encSize.width  / originalImageSize.width;
        float y1 = rc.y *  (float)encSize.height / originalImageSize.height;
        m_promptPointCoords.push_back(x1);
        m_promptPointCoords.push_back(y1);
        m_promptPointLabels.push_back(2.f);                 // label-2 (TL)

        // scale BR corner → encoder space, push with label 3
        float x2 = (rc.x + rc.width)  * (float)encSize.width  / originalImageSize.width;
        float y2 = (rc.y + rc.height) * (float)encSize.height / originalImageSize.height;
        m_promptPointCoords.push_back(x2);
        m_promptPointCoords.push_back(y2);
        m_promptPointLabels.push_back(3.f);                 // label-3 (BR)
    }

    /*───────────────────────────────────────────────────────────────
      2.  Individual points (seed clicks)
          label: 1 = positive (FG), 0 = negative (BG)
    ───────────────────────────────────────────────────────────────*/
    for (size_t i = 0; i < prompts.points.size(); ++i)
    {
        float x = prompts.points[i].x * (float)encSize.width  / originalImageSize.width;
        float y = prompts.points[i].y * (float)encSize.height / originalImageSize.height;
        m_promptPointCoords.push_back(x);
        m_promptPointCoords.push_back(y);
        m_promptPointLabels.push_back((float)prompts.pointLabels[i]);   // 0 or 1
    }
}

void SAM2::setRectsLabels(const std::list<SAM2Rect> &rects,
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

void SAM2::setPointsLabels(const std::list<SAM2Point> &points,
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

Image<float> SAM2::createBinaryMask(const SAM2Size &targetSize,
                                    const SAM2Size &maskSize,
                                    float *maskData,
                                    float threshold)
{
    // Create an Image<float> for the low-resolution mask directly.
    // (Assuming the mask is single-channel.)
    Image<float> lowResImg(maskSize.width, maskSize.height, 1);
    std::memcpy(lowResImg.getData().data(), maskData, sizeof(float) * maskSize.width * maskSize.height);

    // Resize the low-resolution image to the target size using our custom resize method.
    Image<float> resizedImg = lowResImg.resize(targetSize.width, targetSize.height);

    // Create a new Image<float> for the binary mask (single channel) with the same dimensions.
    Image<float> binaryMask(targetSize.width, targetSize.height, 1);

    // Calculate the total number of pixels.
    size_t totalPixels = static_cast<size_t>(targetSize.width * targetSize.height);

    // Get references to the underlying data arrays.
    const std::vector<float>& srcData = resizedImg.getData();
    std::vector<float>& dstData = binaryMask.getData();

    // Flattened loop over all pixels.
    for (size_t idx = 0; idx < totalPixels; ++idx) {
        dstData[idx] = (srcData[idx] > threshold) ? 1.0f : 0.0f;
    }

    return binaryMask;
}

Image<float> SAM2::extractAndCreateMask(Ort::Value &maskTensor, const SAM2Size &targetSize)
{
    // Retrieve the raw float pointer from the tensor.
    float* maskData = maskTensor.GetTensorMutableData<float>();

    // Get the tensor shape.
    auto maskShape = maskTensor.GetTensorTypeAndShapeInfo().GetShape();

    // Verify that the shape meets the expected 4 dimensions.
    if(maskShape.size() < 4){
        std::cerr << "[ERROR] extractAndCreateMask => unexpected mask shape." << std::endl;
        return Image<float>();
    }

    // Extract the mask dimensions from the shape.
    int maskH = static_cast<int>(maskShape[2]);
    int maskW = static_cast<int>(maskShape[3]);

    // Call the existing helper to create a binary mask.
    return createBinaryMask(targetSize, SAM2Size(maskW, maskH), maskData);
}
