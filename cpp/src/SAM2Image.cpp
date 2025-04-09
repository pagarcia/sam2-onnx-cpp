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

cv::Mat SAM2::InferSingleFrame(const Size &originalImageSize)
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
    cv::resize(lowRes, upFloat, cv::Size(originalImageSize.width, originalImageSize.height), 0, 0, cv::INTER_LINEAR);

    cv::Mat finalMask(cv::Size(originalImageSize.width, originalImageSize.height), CV_8UC1, cv::Scalar(0));
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
// Optional label helpers
// --------------------
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
