#include "SAM2.h"
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <cstring> // for memcpy

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
EncoderOutputs SAM2::getEncoderOutputsFromImage(const cv::Mat &originalImage, Size targetImageSize) {
    EncoderOutputs outputs;

    cv::Mat targetImage;
    cv::resize(originalImage, targetImage, cv::Size(targetImageSize.width, targetImageSize.height));

    // Assume 'frame' is already resized to the expected SAM2ImageSize.
    // (If not, you could resize here or let the caller do it.)
    std::vector<float> encData = normalizeBGR(targetImage);

    // Create an input tensor using the input shape (m_inputShapeEncoder)
    Ort::Value inTensor = createTensor<float>(m_memoryInfo, encData, m_inputShapeEncoder);
    std::vector<Ort::Value> encInputs;
    encInputs.reserve(1);
    encInputs.push_back(std::move(inTensor));

    auto encRes = runImageEncoderSession(encInputs);
    if(encRes.index() == 1) throw std::runtime_error(std::get<std::string>(encRes));
    outputs.outputs = std::move(std::get<0>(encRes));

    if(outputs.outputs.size() < 3) throw std::runtime_error("Encoder returned insufficient outputs.");

    // Extract the first three outputs:
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

bool SAM2::preprocessImage(const cv::Mat &originalImage)
{
    try {
        Size SAM2ImageSize = getInputSize();
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

cv::Mat SAM2::inferSingleFrame(const Size &originalImageSize)
{
    if(m_promptPointLabels.empty() || m_promptPointCoords.empty()){
        std::cerr << "[WARN] InferSingleFrame => no prompts.\n";
        return cv::Mat();
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
        return cv::Mat();
    }
    auto &decOuts = std::get<0>(decRes);
    if(decOuts.size() < 3){
        std::cerr << "[ERROR] decode returned <3 outputs.\n";
        return cv::Mat();
    }
    // decOuts[2] => pred_mask => [1,N,256,256]
    // Create a low-resolution floar mask image from the decoder output and convert it to binary uchar mask
    cv::Mat originalImageSizeBinaryMask = extractAndCreateMask(decOuts[2], originalImageSize);
    return originalImageSizeBinaryMask;
}

// ------------------
// Prompt and helpers
// ------------------
void SAM2::setPrompts(const Prompts &prompts, const Size &originalImageSize)
{
    m_promptPointCoords.clear();
    m_promptPointLabels.clear();

    Size SAM2ImageSize = getInputSize();
    if(SAM2ImageSize.width <= 0 || SAM2ImageSize.height <= 0){
        std::cerr << "[WARN] setPrompts => invalid encoder size.\n";
        return;
    }

    // Rect => label=2,3
    for(const auto &rc : prompts.rects){
        float x1 = rc.x * (float)SAM2ImageSize.width / (float)originalImageSize.width;
        float y1 = rc.y * (float)SAM2ImageSize.height / (float)originalImageSize.height;
        m_promptPointCoords.push_back(x1);
        m_promptPointCoords.push_back(y1);
        m_promptPointLabels.push_back(2.f);

        float x2 = rc.br().x * (float)SAM2ImageSize.width / (float)originalImageSize.width;
        float y2 = rc.br().y * (float)SAM2ImageSize.height / (float)originalImageSize.height;
        m_promptPointCoords.push_back(x2);
        m_promptPointCoords.push_back(y2);
        m_promptPointLabels.push_back(3.f);
    }

    // Points => label=1,0,etc.
    for(size_t i=0; i<prompts.points.size(); i++){
        float x = prompts.points[i].x * (float)SAM2ImageSize.width / (float)originalImageSize.width;
        float y = prompts.points[i].y * (float)SAM2ImageSize.height/ (float)originalImageSize.height;
        m_promptPointCoords.push_back(x);
        m_promptPointCoords.push_back(y);
        m_promptPointLabels.push_back((float)prompts.pointLabels[i]);
    }
}

void SAM2::setRectsLabels(const std::list<Rect> &rects,
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

void SAM2::setPointsLabels(const std::list<Point> &points,
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

cv::Mat SAM2::createBinaryMask(const Size &targetSize, 
                               const Size &maskSize, 
                               float *maskData, 
                               float threshold)
{
    // Create a low-resolution float mask from the raw data.
    cv::Mat lowResFloatMask(maskSize.height, maskSize.width, CV_32FC1, static_cast<void*>(maskData));
    
    // Resize to the target size (convert from SAM2::Size to cv::Size).
    cv::Mat resizedFloatMask;
    cv::resize(lowResFloatMask, resizedFloatMask,
               cv::Size(targetSize.width, targetSize.height), 0, 0, cv::INTER_LINEAR);
    
    // Create a binary mask with the target size.
    cv::Mat binaryMask(cv::Size(targetSize.width, targetSize.height), CV_8UC1, cv::Scalar(0));
    
    // Manual loop to convert the upscaled float mask to a binary mask.
    for (int r = 0; r < binaryMask.rows; ++r) {
        const float* srcRow = resizedFloatMask.ptr<float>(r);
        uchar* dstRow = binaryMask.ptr<uchar>(r);
        for (int c = 0; c < binaryMask.cols; ++c) {
            dstRow[c] = (srcRow[c] > threshold) ? 255 : 0;
        }
    }
    return binaryMask;
}

cv::Mat SAM2::extractAndCreateMask(Ort::Value &maskTensor, const Size &targetSize)
{
    // Retrieve the raw float pointer from the tensor.
    float* maskData = maskTensor.GetTensorMutableData<float>();

    // Get the tensor shape.
    auto maskShape = maskTensor.GetTensorTypeAndShapeInfo().GetShape();

    // Verify that the shape meets the expected 4 dimensions.
    if(maskShape.size() < 4){
        std::cerr << "[ERROR] extractAndCreateMask => unexpected mask shape." << std::endl;
        return cv::Mat();
    }

    // Extract the mask dimensions from the shape.
    int maskH = static_cast<int>(maskShape[2]);
    int maskW = static_cast<int>(maskShape[3]);

    // Call the existing helper to create a binary mask.
    return createBinaryMask(targetSize, Size(maskW, maskH), maskData);
}
