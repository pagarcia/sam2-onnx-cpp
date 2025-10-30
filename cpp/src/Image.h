// sam2-onnx-cpp/cpp/src/Image.h
#ifndef IMAGE_H
#define IMAGE_H

#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <numeric>    // std::iota
#include <thread>
#include <atomic>
#include <exception>

namespace sam2_detail {

// Minimal portable parallel_for (no external deps, works on macOS & Windows).
template <class F>
inline void parallel_for(std::size_t begin, std::size_t end, F&& fn, unsigned nthreads = 0) {
    if (end <= begin) return;

    if (nthreads == 0) nthreads = std::max(1u, std::thread::hardware_concurrency());
    const std::size_t total = end - begin;

    // Small ranges: do it serially (avoids thread overhead)
    if (nthreads <= 1 || total < 64) {
        for (std::size_t i = begin; i < end; ++i) fn(i);
        return;
    }

    const std::size_t chunk = (total + nthreads - 1) / nthreads;
    std::vector<std::thread> threads;
    threads.reserve(nthreads);

    std::atomic<bool> failed{false};
    std::exception_ptr eptr = nullptr;

    for (unsigned t = 0; t < nthreads; ++t) {
        const std::size_t b = begin + t * chunk;
        if (b >= end) break;
        const std::size_t e = std::min(end, b + chunk);

        threads.emplace_back([&, b, e]() {
            try {
                for (std::size_t i = b; i < e; ++i) {
                    fn(i);
                }
            } catch (...) {
                if (!failed.exchange(true)) {
                    eptr = std::current_exception();
                }
            }
        });
    }

    for (auto& th : threads) th.join();
    if (eptr) std::rethrow_exception(eptr);
}

} // namespace sam2_detail

template <typename T>
class Image {
public:
    Image() : width(0), height(0), channels(1) {}

    Image(int w, int h, int c = 1)
        : width(w), height(h), channels(c), data(static_cast<std::size_t>(w) * h * c) {
        if (w < 0 || h < 0 || c <= 0) throw std::runtime_error("Invalid image dimensions/channels");
    }

    Image(int w, int h, int c, const std::vector<T>& d)
        : width(w), height(h), channels(c), data(d) {
        if (w < 0 || h < 0 || c <= 0) throw std::runtime_error("Invalid image dimensions/channels");
        if (data.size() != static_cast<std::size_t>(w) * h * c) {
            throw std::runtime_error("Data size does not match image dimensions and channel count.");
        }
    }

    int getWidth()    const { return width; }
    int getHeight()   const { return height; }
    int getChannels() const { return channels; }

    const std::vector<T>& getData() const { return data; }
    std::vector<T>&       getData()       { return data; }

    // Interleaved (RGBRGB...) -> Planar ([all R][all G][all B])
    std::vector<T> getDataPlanarFormat() const {
        const std::size_t W = static_cast<std::size_t>(width);
        const std::size_t H = static_cast<std::size_t>(height);
        const std::size_t C = static_cast<std::size_t>(channels);

        if (C == 1) {
            // Single channel: return a copy as-is
            return data;
        }

        const std::size_t totalPixels = W * H;
        const std::size_t totalValues = totalPixels * C;
        std::vector<T> planar(totalValues);

        // Parallelize across channels
        sam2_detail::parallel_for(0, C, [&](std::size_t c) {
            const std::size_t dstOffset = c * totalPixels;
            for (std::size_t y = 0; y < H; ++y) {
                const std::size_t rowBase = y * W;
                for (std::size_t x = 0; x < W; ++x) {
                    const std::size_t p   = rowBase + x;
                    const std::size_t src = p * C + c;
                    planar[dstOffset + p] = data[src];
                }
            }
        });

        return planar;
    }

    // Accessors
    T& at(int x, int y, int c) {
        boundsCheck(x, y, c);
        return data[indexOf(x, y, c)];
    }

    const T& at(int x, int y, int c) const {
        boundsCheck(x, y, c);
        return data[indexOf(x, y, c)];
    }

    T& at(int x, int y) {
        if (channels != 1) throw std::runtime_error("at(x,y) only valid for single-channel images.");
        return at(x, y, 0);
    }
    const T& at(int x, int y) const {
        if (channels != 1) throw std::runtime_error("at(x,y) only valid for single-channel images.");
        return at(x, y, 0);
    }

    // Bilinear resize (parallelized by rows)
    Image<T> resize(int newWidth, int newHeight) const {
        if (newWidth <= 0 || newHeight <= 0) {
            throw std::runtime_error("resize: invalid target size");
        }

        Image<T> out(newWidth, newHeight, channels);

        const double scaleX = static_cast<double>(width)  / static_cast<double>(newWidth);
        const double scaleY = static_cast<double>(height) / static_cast<double>(newHeight);

        sam2_detail::parallel_for(0, static_cast<std::size_t>(newHeight), [&](std::size_t jz) {
            const int j = static_cast<int>(jz);
            const double srcY = (static_cast<double>(j) + 0.5) * scaleY - 0.5;
            int y0 = static_cast<int>(std::floor(srcY));
            int y1 = y0 + 1;
            if (y0 < 0) y0 = 0;
            if (y1 >= height) y1 = height - 1;
            const double dy = srcY - static_cast<double>(y0);

            for (int i = 0; i < newWidth; ++i) {
                const double srcX = (static_cast<double>(i) + 0.5) * scaleX - 0.5;
                int x0 = static_cast<int>(std::floor(srcX));
                int x1 = x0 + 1;
                if (x0 < 0) x0 = 0;
                if (x1 >= width) x1 = width - 1;
                const double dx = srcX - static_cast<double>(x0);

                for (int c = 0; c < channels; ++c) {
                    const double v00 = static_cast<double>(at(x0, y0, c));
                    const double v10 = static_cast<double>(at(x1, y0, c));
                    const double v01 = static_cast<double>(at(x0, y1, c));
                    const double v11 = static_cast<double>(at(x1, y1, c));

                    // Bilinear
                    const double v0 = v00 + (v10 - v00) * dx;
                    const double v1 = v01 + (v11 - v01) * dx;
                    const double v  = v0  + (v1  - v0)  * dy;

                    out.at(i, j, c) = static_cast<T>(v);
                }
            }
        });

        return out;
    }

private:
    inline std::size_t indexOf(int x, int y, int c) const {
        return static_cast<std::size_t>((y * width + x) * channels + c);
    }

    inline void boundsCheck(int x, int y, int c) const {
        if (x < 0 || x >= width || y < 0 || y >= height || c < 0 || c >= channels) {
            throw std::out_of_range("Image index out of range.");
        }
    }

private:
    int width, height, channels;
    std::vector<T> data;
};

#endif // IMAGE_H
