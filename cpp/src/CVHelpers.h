#ifndef CVHELPERS_H
#define CVHELPERS_H

#include <opencv2/core.hpp>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include "Image.h"

namespace CVHelpers {

// Convert a cv::Mat to an Image<T>
// This function computes the expected type as CV_MAKETYPE(cv::DataType<T>::depth, channels)
// so that multi-channel images are handled correctly.
template <typename T>
Image<T> cvMatToImage(const cv::Mat &mat) {
    int channels = mat.channels();
    int expectedDepth = cv::DataType<T>::depth;
    int expectedType = CV_MAKETYPE(expectedDepth, channels);
    if(mat.type() != expectedType) {
        throw std::runtime_error("cvMatToImage: matrix type does not match template type");
    }
    
    // Create an Image with the same width, height, and number of channels.
    Image<T> img(mat.cols, mat.rows, channels);
    
    // Copy data: note that total number of elements is mat.total() * channels.
    if(mat.isContinuous()) {
        std::memcpy(img.getData().data(), mat.ptr<T>(), mat.total() * channels * sizeof(T));
    } else {
        for (int r = 0; r < mat.rows; r++) {
            const T* rowPtr = mat.ptr<T>(r);
            std::copy(rowPtr, rowPtr + mat.cols * channels,
                      img.getData().begin() + r * (mat.cols * channels));
        }
    }
    return img;
}

// Convert an Image<T> to cv::Mat.
// The image type is computed as CV_MAKETYPE(cv::DataType<T>::depth, channels)
template <typename T>
cv::Mat imageToCvMat(const Image<T> &img) {
    int channels = img.getChannels();
    int depth = cv::DataType<T>::depth;
    int type = CV_MAKETYPE(depth, channels);
    
    cv::Mat mat(img.getHeight(), img.getWidth(), type);
    if(mat.isContinuous()) {
        std::memcpy(mat.ptr<T>(), img.getData().data(), img.getData().size() * sizeof(T));
    } else {
        for (int r = 0; r < mat.rows; r++) {
            T* rowPtr = mat.ptr<T>(r);
            std::copy(img.getData().begin() + r * img.getWidth() * channels,
                      img.getData().begin() + (r + 1) * img.getWidth() * channels,
                      rowPtr);
        }
    }
    return mat;
}

// Normalize a BGR cv::Mat to a float vector in R, G, B order.
// The input cv::Mat is expected to be an 8-bit, 3-channel image (CV_8UC3).
// This function returns a vector of floats with normalized values.
inline std::vector<float> normalizeBGR(const cv::Mat &bgrImg,
                                       float meanR = 0.485f,
                                       float meanG = 0.456f,
                                       float meanB = 0.406f,
                                       float stdR  = 0.229f,
                                       float stdG  = 0.224f,
                                       float stdB  = 0.225f)
{
    // Ensure the input image has 3 channels.
    if(bgrImg.channels() != 3) {
        throw std::runtime_error("normalizeBGR: input image must have 3 channels (BGR).");
    }
    
    const int H = bgrImg.rows;
    const int W = bgrImg.cols;
    const size_t total = static_cast<size_t>(3 * H * W);
    std::vector<float> data(total, 0.f);
    const int planeSize = W * H;
    
    // The input is in BGR order; output will be in R, G, B order.
    for (int r = 0; r < H; r++) {
        for (int c = 0; c < W; c++) {
            int idx = r * W + c;
            cv::Vec3b pixel = bgrImg.at<cv::Vec3b>(r, c);
            float b = pixel[0] / 255.f;
            float g = pixel[1] / 255.f;
            float rVal = pixel[2] / 255.f;
            
            data[idx + 0 * planeSize] = (rVal - meanR) / stdR;  // R channel
            data[idx + 1 * planeSize] = (g - meanG) / stdG;     // G channel
            data[idx + 2 * planeSize] = (b - meanB) / stdB;     // B channel
        }
    }
    return data;
}

inline Image<float> normalizeRGB(const cv::Mat &bgrImg,
                                 float meanR = 0.485f,
                                 float meanG = 0.456f,
                                 float meanB = 0.406f,
                                 float stdR  = 0.229f,
                                 float stdG  = 0.224f,
                                 float stdB  = 0.225f)
{
    if(bgrImg.channels() != 3) {
        throw std::runtime_error("standardizeRGB: input image must have 3 channels (BGR).");
    }
    
    const int H = bgrImg.rows;
    const int W = bgrImg.cols;
    
    // Create an Image<float> with the same width and height as the input and 3 channels.
    // We will store the data in interleaved format: for each pixel, channels appear consecutively.
    Image<float> img(W, H, 3);
    
    // Loop over each pixel.
    for (int r = 0; r < H; r++) {
        for (int c = 0; c < W; c++) {
            cv::Vec3b pixel = bgrImg.at<cv::Vec3b>(r, c);
            // By default, OpenCV stores 8-bit color images in BGR order
            // So the first element is blue, the second green, and the third red
            // Convert channels to float in [0,1].
            float bVal = pixel[0] / 255.f;
            float gVal = pixel[1] / 255.f;
            float rVal = pixel[2] / 255.f;
            
            // Store in interleaved order as R, G, B.
            img.at(c, r, 0) = (rVal - meanR) / stdR;
            img.at(c, r, 1) = (gVal - meanG) / stdG;
            img.at(c, r, 2) = (bVal - meanB) / stdB;
        }
    }
    
    return img;
}

} // namespace CVHelpers

#endif // CVHELPERS_H
