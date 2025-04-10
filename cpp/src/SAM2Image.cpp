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
bool SAM2::preprocessImage(const cv::Mat &originalImage)
{
    try {
        Size SAM2ImageSize = getInputSize();
        cv::Mat SAM2Image;
        cv::resize(originalImage, SAM2Image, cv::Size(SAM2ImageSize.width, SAM2ImageSize.height));

        // Convert BGR to normalized float
        std::vector<float> data = normalizeBGR(SAM2Image);

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

cv::Mat SAM2::inferSingleFrame(const Size &originalImageSize)
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

    // Run the decoder
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

    // Create a low-resolution floar mask image from the decoder output and convert it to binary uchar mask
    cv::Mat originalImageSizeBinaryMask = createBinaryMask(originalImageSize, Size(maskW, maskH), pmData);
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
