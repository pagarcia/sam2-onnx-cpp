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
    // Default constructor creates an empty image.
    Image() : width(0), height(0) {}

    // Constructor that creates an image of given width and height.
    Image(int w, int h)
        : width(w), height(h), data(w * h) {}

    // Constructor that creates an image from existing data.
    // Throws if the data size does not match the given dimensions.
    Image(int w, int h, const std::vector<T>& d)
        : width(w), height(h), data(d) {
        if (data.size() != static_cast<size_t>(w * h)) {
            throw std::runtime_error("Data size does not match image dimensions.");
        }
    }

    // Accessors for image dimensions.
    int getWidth() const { return width; }
    int getHeight() const { return height; }

    // Access the underlying vector containing the pixel data.
    const std::vector<T>& getData() const { return data; }
    std::vector<T>& getData() { return data; }

    // Access pixel at (x, y) in row-major order.
    T& at(int x, int y) {
        if (x < 0 || x >= width || y < 0 || y >= height) {
            throw std::out_of_range("Image index out of range.");
        }
        return data[y * width + x];
    }

    const T& at(int x, int y) const {
        if (x < 0 || x >= width || y < 0 || y >= height) {
            throw std::out_of_range("Image index out of range.");
        }
        return data[y * width + x];
    }

    // Resize function using bilinear interpolation (parallelized using C++17).
    // Returns a new Image<T> of dimensions (newWidth, newHeight).
    Image<T> resize(int newWidth, int newHeight) const {
        Image<T> resized(newWidth, newHeight);

        // Compute scaling factors.
        double scaleX = static_cast<double>(width) / newWidth;
        double scaleY = static_cast<double>(height) / newHeight;

        // Create an index vector for rows.
        std::vector<int> rowIndices(newHeight);
        std::iota(rowIndices.begin(), rowIndices.end(), 0);

        // Parallelize the outer loop over rows.
        std::for_each(std::execution::par, rowIndices.begin(), rowIndices.end(),
            [&](int j) {
                // Map the output row center to the source image.
                double srcY = (j + 0.5) * scaleY - 0.5;
                int y0 = std::max(0, static_cast<int>(std::floor(srcY)));
                int y1 = std::min(height - 1, y0 + 1);
                double dy = srcY - y0;

                for (int i = 0; i < newWidth; i++) {
                    double srcX = (i + 0.5) * scaleX - 0.5;
                    int x0 = std::max(0, static_cast<int>(std::floor(srcX)));
                    int x1 = std::min(width - 1, x0 + 1);
                    double dx = srcX - x0;

                    // Fetch the four neighboring pixels.
                    double v00 = static_cast<double>(at(x0, y0));
                    double v10 = static_cast<double>(at(x1, y0));
                    double v01 = static_cast<double>(at(x0, y1));
                    double v11 = static_cast<double>(at(x1, y1));

                    // Bilinear interpolation formula.
                    double value = (1 - dx) * (1 - dy) * v00 +
                                   dx       * (1 - dy) * v10 +
                                   (1 - dx) * dy       * v01 +
                                   dx       * dy       * v11;

                    resized.at(i, j) = static_cast<T>(value);
                }
            }
        );
        return resized;
    }

private:
    int width, height;
    std::vector<T> data;
};

#endif // IMAGE_H
