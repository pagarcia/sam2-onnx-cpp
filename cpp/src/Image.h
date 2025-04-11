#ifndef IMAGE_H
#define IMAGE_H

#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <numeric>      // for std::iota
#include <execution>    // for std::execution::par (requires C++17)

template <typename T>
class Image {
public:
    // Default constructor creates an empty image with 1 channel.
    Image() : width(0), height(0), channels(1) {}

    // Constructor that creates an image of given width, height, and channel count.
    Image(int w, int h, int c = 1)
        : width(w), height(h), channels(c), data(w * h * c) {}

    // Constructor that creates an image from existing data.
    // Throws if the data size does not match the given dimensions and channels.
    Image(int w, int h, int c, const std::vector<T>& d)
        : width(w), height(h), channels(c), data(d) {
        if (data.size() != static_cast<size_t>(w * h * c)) {
            throw std::runtime_error("Data size does not match image dimensions and channel count.");
        }
    }

    // Accessors for image dimensions.
    int getWidth() const { return width; }
    int getHeight() const { return height; }
    int getChannels() const { return channels; }

    // Access the underlying vector containing the pixel data.
    const std::vector<T>& getData() const { return data; }
    std::vector<T>& getData() { return data; }

    // Access pixel at (x, y) and channel c (row-major order).
    // Index is computed as: (y * width + x)*channels + c.
    T& at(int x, int y, int c) {
        if (x < 0 || x >= width || y < 0 || y >= height || c < 0 || c >= channels) {
            throw std::out_of_range("Image index out of range.");
        }
        return data[(y * width + x) * channels + c];
    }

    const T& at(int x, int y, int c) const {
        if (x < 0 || x >= width || y < 0 || y >= height || c < 0 || c >= channels) {
            throw std::out_of_range("Image index out of range.");
        }
        return data[(y * width + x) * channels + c];
    }

    // Convenience accessor for single-channel images.
    T& at(int x, int y) {
        if (channels != 1)
            throw std::runtime_error("at(x, y) is only available for single-channel images.");
        return at(x, y, 0);
    }
    const T& at(int x, int y) const {
        if (channels != 1)
            throw std::runtime_error("at(x, y) is only available for single-channel images.");
        return at(x, y, 0);
    }

    // Resize function using bilinear interpolation (parallelized using C++17).
    // It handles all channels by performing interpolation independently per channel.
    // Returns a new Image<T> of dimensions (newWidth, newHeight) with the same channel count.
    Image<T> resize(int newWidth, int newHeight) const {
        Image<T> resized(newWidth, newHeight, channels);

        // Compute scaling factors.
        double scaleX = static_cast<double>(width) / newWidth;
        double scaleY = static_cast<double>(height) / newHeight;

        // Create an index vector for rows.
        std::vector<int> rowIndices(newHeight);
        std::iota(rowIndices.begin(), rowIndices.end(), 0);

        // Parallelize the outer loop over rows.
        std::for_each(std::execution::par, rowIndices.begin(), rowIndices.end(),
            [&](int j) {
                // Map the output row center to a source coordinate.
                double srcY = (j + 0.5) * scaleY - 0.5;
                int y0 = std::max(0, static_cast<int>(std::floor(srcY)));
                int y1 = std::min(height - 1, y0 + 1);
                double dy = srcY - y0;

                for (int i = 0; i < newWidth; i++) {
                    double srcX = (i + 0.5) * scaleX - 0.5;
                    int x0 = std::max(0, static_cast<int>(std::floor(srcX)));
                    int x1 = std::min(width - 1, x0 + 1);
                    double dx = srcX - x0;

                    // Process each channel separately.
                    for (int c = 0; c < channels; c++) {
                        double v00 = static_cast<double>(at(x0, y0, c));
                        double v10 = static_cast<double>(at(x1, y0, c));
                        double v01 = static_cast<double>(at(x0, y1, c));
                        double v11 = static_cast<double>(at(x1, y1, c));

                        // Bilinear interpolation.
                        double value = (1 - dx) * (1 - dy) * v00 +
                                       dx       * (1 - dy) * v10 +
                                       (1 - dx) * dy       * v01 +
                                       dx       * dy       * v11;
                        resized.at(i, j, c) = static_cast<T>(value);
                    }
                }
            }
        );
        return resized;
    }

private:
    int width, height, channels;
    std::vector<T> data;
};

#endif // IMAGE_H
