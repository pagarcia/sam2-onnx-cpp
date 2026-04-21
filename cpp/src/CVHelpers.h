#ifndef CVHELPERS_H
#define CVHELPERS_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "Image.h"

namespace CVHelpers {

template <typename T>
Image<T> cvMatToImage(const cv::Mat &mat)
{
    const int channels = mat.channels();
    const int expectedDepth = cv::DataType<T>::depth;
    const int expectedType = CV_MAKETYPE(expectedDepth, channels);
    if (mat.type() != expectedType) {
        throw std::runtime_error("cvMatToImage: matrix type does not match template type");
    }

    Image<T> img(mat.cols, mat.rows, channels);
    if (mat.isContinuous()) {
        std::memcpy(img.getData().data(), mat.ptr<T>(), mat.total() * channels * sizeof(T));
    } else {
        for (int r = 0; r < mat.rows; ++r) {
            const T* rowPtr = mat.ptr<T>(r);
            std::copy(
                rowPtr,
                rowPtr + mat.cols * channels,
                img.getData().begin() + static_cast<ptrdiff_t>(r) * mat.cols * channels);
        }
    }
    return img;
}

template <typename T>
cv::Mat imageToCvMat(const Image<T> &img)
{
    const int channels = img.getChannels();
    const int depth = cv::DataType<T>::depth;
    const int type = CV_MAKETYPE(depth, channels);

    cv::Mat mat(img.getHeight(), img.getWidth(), type);
    if (mat.isContinuous()) {
        std::memcpy(mat.ptr<T>(), img.getData().data(), img.getData().size() * sizeof(T));
    } else {
        for (int r = 0; r < mat.rows; ++r) {
            T* rowPtr = mat.ptr<T>(r);
            std::copy(
                img.getData().begin() + static_cast<ptrdiff_t>(r) * img.getWidth() * channels,
                img.getData().begin() + static_cast<ptrdiff_t>(r + 1) * img.getWidth() * channels,
                rowPtr);
        }
    }
    return mat;
}

template <typename T>
cv::Mat imageToCvMatView(const Image<T> &img)
{
    const int channels = img.getChannels();
    const int depth = cv::DataType<T>::depth;
    const int type = CV_MAKETYPE(depth, channels);
    return cv::Mat(
        img.getHeight(),
        img.getWidth(),
        type,
        const_cast<T*>(img.getData().data()));
}

template <typename T>
cv::Mat imageToCvMatWithType(const Image<T>& img, int targetType = CV_8UC1, double scaleFactor = 255.0)
{
    cv::Mat mat = imageToCvMat(img);
    cv::Mat converted;
    mat.convertTo(converted, targetType, scaleFactor);
    return converted;
}

inline Image<float> normalizeRGB(const cv::Mat &bgrImg,
                                 float scaleFactor = 255.f,
                                 float meanR = 0.485f,
                                 float meanG = 0.456f,
                                 float meanB = 0.406f,
                                 float stdR  = 0.229f,
                                 float stdG  = 0.224f,
                                 float stdB  = 0.225f)
{
    if (bgrImg.channels() != 3) {
        throw std::runtime_error("normalizeRGB: input image must have 3 channels (BGR).");
    }

    cv::Mat rgb;
    cv::cvtColor(bgrImg, rgb, cv::COLOR_BGR2RGB);

    cv::Mat rgbFloat;
    rgb.convertTo(rgbFloat, CV_32FC3, 1.0 / static_cast<double>(scaleFactor));

    std::vector<cv::Mat> channels(3);
    cv::split(rgbFloat, channels);
    channels[0] = (channels[0] - meanR) / stdR;
    channels[1] = (channels[1] - meanG) / stdG;
    channels[2] = (channels[2] - meanB) / stdB;
    cv::merge(channels, rgbFloat);

    return cvMatToImage<float>(rgbFloat);
}

inline std::vector<float> resizeImageToPlanarTensor(const Image<float> &img,
                                                    int targetWidth,
                                                    int targetHeight)
{
    if (targetWidth <= 0 || targetHeight <= 0) {
        throw std::runtime_error("resizeImageToPlanarTensor: invalid target size");
    }

    cv::Mat source = imageToCvMatView(img);
    cv::Mat resized;
    if (img.getWidth() == targetWidth && img.getHeight() == targetHeight) {
        resized = source;
    } else {
        cv::resize(source, resized, cv::Size(targetWidth, targetHeight), 0.0, 0.0, cv::INTER_LINEAR);
    }

    const int channels = resized.channels();
    if (channels == 1) {
        std::vector<float> planar(static_cast<size_t>(targetWidth) * targetHeight);
        if (resized.isContinuous()) {
            std::memcpy(planar.data(), resized.ptr<float>(), planar.size() * sizeof(float));
        } else {
            for (int r = 0; r < resized.rows; ++r) {
                const float* rowPtr = resized.ptr<float>(r);
                std::copy(
                    rowPtr,
                    rowPtr + resized.cols,
                    planar.begin() + static_cast<ptrdiff_t>(r) * resized.cols);
            }
        }
        return planar;
    }

    std::vector<cv::Mat> splitChannels;
    cv::split(resized, splitChannels);

    const size_t planeSize = static_cast<size_t>(targetWidth) * targetHeight;
    std::vector<float> planar(planeSize * static_cast<size_t>(channels));
    for (int channel = 0; channel < channels; ++channel) {
        cv::Mat plane = splitChannels[channel];
        float* dst = planar.data() + planeSize * static_cast<size_t>(channel);
        if (plane.isContinuous()) {
            std::memcpy(dst, plane.ptr<float>(), planeSize * sizeof(float));
        } else {
            for (int r = 0; r < plane.rows; ++r) {
                const float* rowPtr = plane.ptr<float>(r);
                std::copy(
                    rowPtr,
                    rowPtr + plane.cols,
                    dst + static_cast<ptrdiff_t>(r) * plane.cols);
            }
        }
    }

    return planar;
}

inline Image<float> resizeAndThresholdMask(const float *maskData,
                                           int maskWidth,
                                           int maskHeight,
                                           int targetWidth,
                                           int targetHeight,
                                           float threshold = 0.0f)
{
    if (!maskData || maskWidth <= 0 || maskHeight <= 0 || targetWidth <= 0 || targetHeight <= 0) {
        throw std::runtime_error("resizeAndThresholdMask: invalid mask dimensions");
    }

    cv::Mat lowRes(maskHeight, maskWidth, CV_32FC1, const_cast<float*>(maskData));
    cv::Mat resized;
    cv::resize(lowRes, resized, cv::Size(targetWidth, targetHeight), 0.0, 0.0, cv::INTER_LINEAR);

    cv::Mat binary;
    cv::threshold(resized, binary, threshold, 1.0, cv::THRESH_BINARY);
    return cvMatToImage<float>(binary);
}

} // namespace CVHelpers

#endif // CVHELPERS_H
