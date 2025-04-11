#ifndef CVHELPERS_H
#define CVHELPERS_H

#include <opencv2/core.hpp>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include "Image.h"

namespace CVHelpers {

// Convert a cv::Mat to an Image<T>
// This function checks that the cv::Mat data type matches T by using cv::DataType<T>::type.
template <typename T>
Image<T> cvMatToImage(const cv::Mat &mat) {
    int expectedType = cv::DataType<T>::type;
    if(mat.type() != expectedType) {
        throw std::runtime_error("cvMatToImage: matrix type does not match template type");
    }

    // Create an Image with the same width and height as the cv::Mat.
    Image<T> img(mat.cols, mat.rows);

    // If cv::Mat is stored continuously in memory, use memcpy for efficient copy.
    if(mat.isContinuous()) {
        std::memcpy(img.getData().data(), mat.ptr<T>(), mat.total() * sizeof(T));
    } else {
        // Otherwise, copy row by row.
        for (int r = 0; r < mat.rows; r++) {
            const T* rowPtr = mat.ptr<T>(r);
            std::copy(rowPtr, rowPtr + mat.cols, img.getData().begin() + r * mat.cols);
        }
    }
    return img;
}

// Convert an Image<T> to cv::Mat
// This function allocates a new cv::Mat and copies the image data from our container.
template <typename T>
cv::Mat imageToCvMat(const Image<T> &img) {
    int type = cv::DataType<T>::type;
    // Create a cv::Mat with the same dimensions.
    cv::Mat mat(img.getHeight(), img.getWidth(), type);
    
    if(mat.isContinuous()) {
        std::memcpy(mat.ptr<T>(), img.getData().data(), img.getData().size() * sizeof(T));
    } else {
        // Copy row by row in case the cv::Mat is not stored continuously.
        for (int r = 0; r < mat.rows; r++) {
            T* rowPtr = mat.ptr<T>(r);
            std::copy(img.getData().begin() + r * img.getWidth(),
                      img.getData().begin() + (r + 1) * img.getWidth(),
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

} // namespace CVHelpers

#endif // CVHELPERS_H
