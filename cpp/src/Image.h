#ifndef IMAGE_H
#define IMAGE_H

#include <vector>
#include <stdexcept>

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

private:
    int width, height;
    std::vector<T> data;
};

#endif // IMAGE_H
