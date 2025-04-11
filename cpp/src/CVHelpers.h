#ifndef CVHELPERS_H
#define CVHELPERS_H

#include <opencv2/core.hpp>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include "Image.h"

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

#endif // CVHELPERS_H
