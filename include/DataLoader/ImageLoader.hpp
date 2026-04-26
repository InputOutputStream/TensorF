#ifndef __IMAGE_LOADER_HPP
#define __IMAGE_LOADER_HPP

/*
 * ImageLoader.hpp
 *
 * Loads images from disk into Matrix<T> and Tensor_t<T> using stb_image.
 *
 * Single image shape  : {H, W, C}      (HWC layout)
 * Batch  image shape  : {N, H, W, C}   (NHWC layout)
 *
 * Pixel normalization modes (NormMode):
 *   ZERO_ONE    — divide by 255                       → [0.0, 1.0]
 *   MEAN_STD    — (x/255 - mean) / std  per channel   → ImageNet style
 *   NONE        — cast uint8 to T as-is               → [0, 255]
 *
 * Resize:
 *   Pass target_h / target_w > 0 to bilinear-resize all images to a fixed
 *   spatial size. Required when building batches of mixed-resolution images.
 *
 * Dependencies:
 *   stb_image.h  (single-header, place next to this file)
 *   See stb_image.h stub for download and setup instructions.
 */

#include "stb_image.h"

#include "../Types/types.hpp"   // shape_t
#include "../DataStructures/Matrix.hpp"
#include "../DataStructures/Tensor.hpp"

#include <string>
#include <vector>
#include <array>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <filesystem>
#include <memory>

// ─────────────────────────────────────────────────────────────────────────────
// Normalization mode
// ─────────────────────────────────────────────────────────────────────────────
enum class NormMode {
    ZERO_ONE,   // pixel / 255.0
    MEAN_STD,   // (pixel/255 - mean) / std   (per channel, ImageNet default)
    NONE        // raw uint8 cast to T
};

// ─────────────────────────────────────────────────────────────────────────────
// Channel mode — what stb_image is asked to produce
// ─────────────────────────────────────────────────────────────────────────────
enum class ChannelMode {
    KEEP_ORIGINAL = 0,  // whatever the file contains
    GRAY          = 1,
    GRAY_ALPHA    = 2,
    RGB           = 3,
    RGBA          = 4
};

// ─────────────────────────────────────────────────────────────────────────────
// ImageNet per-channel mean and std  (RGB order)
// ─────────────────────────────────────────────────────────────────────────────
template<typename T>
struct ImageNetStats {
    static constexpr T mean[3] = { T(0.485), T(0.456), T(0.406) };
    static constexpr T std [3] = { T(0.229), T(0.224), T(0.225) };
};

// ─────────────────────────────────────────────────────────────────────────────
// ImageLoader<T>
// ─────────────────────────────────────────────────────────────────────────────
template<typename T>
class ImageLoader {

private:

    NormMode    norm_mode;
    ChannelMode channel_mode;
    int         target_h;   // 0 = no resize
    int         target_w;

    // Per-channel mean/std used when norm_mode == MEAN_STD
    std::vector<T> mean_vals;
    std::vector<T> std_vals;

    // ── Bilinear resize ──────────────────────────────────────────────────────
    // Resizes a flat HWC uint8 buffer from (src_h, src_w, c) to (dst_h, dst_w, c).
    // Returns a new heap-allocated buffer that must be freed with delete[].
    stbi_uc* bilinear_resize(const stbi_uc* src,
                             int src_h, int src_w,
                             int dst_h, int dst_w,
                             int c) const
    {
        stbi_uc* dst = new stbi_uc[dst_h * dst_w * c];

        float scale_h = static_cast<float>(src_h) / dst_h;
        float scale_w = static_cast<float>(src_w) / dst_w;

        for (int dy = 0; dy < dst_h; dy++) {
            for (int dx = 0; dx < dst_w; dx++) {

                // Map destination pixel to source coordinates
                float fy = (dy + 0.5f) * scale_h - 0.5f;
                float fx = (dx + 0.5f) * scale_w - 0.5f;

                int y0 = std::max(0, static_cast<int>(std::floor(fy)));
                int x0 = std::max(0, static_cast<int>(std::floor(fx)));
                int y1 = std::min(src_h - 1, y0 + 1);
                int x1 = std::min(src_w - 1, x0 + 1);

                float wy = fy - y0;   // vertical weight
                float wx = fx - x0;   // horizontal weight

                for (int ch = 0; ch < c; ch++) {
                    float v00 = src[(y0 * src_w + x0) * c + ch];
                    float v01 = src[(y0 * src_w + x1) * c + ch];
                    float v10 = src[(y1 * src_w + x0) * c + ch];
                    float v11 = src[(y1 * src_w + x1) * c + ch];

                    float val = (1 - wy) * ((1 - wx) * v00 + wx * v01)
                                +    wy  * ((1 - wx) * v10 + wx * v11);

                    dst[(dy * dst_w + dx) * c + ch] =
                        static_cast<stbi_uc>(std::round(std::clamp(val, 0.f, 255.f)));
                }
            }
        }
        return dst;
    }

    // ── Normalize a single pixel value for channel ch ───────────────────────
    T normalize(stbi_uc raw, int ch) const {
        switch (norm_mode) {
            case NormMode::ZERO_ONE:
                return static_cast<T>(raw) / T(255);

            case NormMode::MEAN_STD: {
                T v = static_cast<T>(raw) / T(255);
                int ci = std::min(ch, static_cast<int>(mean_vals.size()) - 1);
                return (v - mean_vals[ci]) / std_vals[ci];
            }
            case NormMode::NONE:
            default:
                return static_cast<T>(raw);
        }
    }

    // ── Core load: path → flat std::vector<T> + shape ───────────────────────
    struct RawImage {
        std::vector<T> data;
        int h, w, c;
    };

    RawImage load_raw(const std::string& path) const {
        int w, h, file_channels;
        int desired = static_cast<int>(channel_mode);

        stbi_uc* pixels = stbi_load(path.c_str(), &w, &h, &file_channels, desired);

        if (!pixels)
            throw std::runtime_error(
                "ImageLoader: failed to load '" + path + "': " +
                std::string(stbi_failure_reason()));

        int c = (desired == 0) ? file_channels : desired;

        // Optional resize
        stbi_uc* final_pixels = pixels;
        int final_h = h, final_w = w;
        bool resized = false;

        if (target_h > 0 && target_w > 0 && (h != target_h || w != target_w)) {
            final_pixels = bilinear_resize(pixels, h, w, target_h, target_w, c);
            final_h = target_h;
            final_w = target_w;
            resized = true;
        }

        // Normalize + copy into vector<T>
        size_t n = static_cast<size_t>(final_h * final_w * c);
        std::vector<T> out;
        out.reserve(n);

        for (int y = 0; y < final_h; y++)
            for (int x = 0; x < final_w; x++)
                for (int ch = 0; ch < c; ch++)
                    out.push_back(normalize(
                        final_pixels[(y * final_w + x) * c + ch], ch));

        // Free stb memory
        stbi_image_free(pixels);
        if (resized) delete[] final_pixels;

        return { std::move(out), final_h, final_w, c };
    }

public:

    // ── Constructor ──────────────────────────────────────────────────────────
    explicit ImageLoader(
        NormMode    norm    = NormMode::ZERO_ONE,
        ChannelMode channel = ChannelMode::RGB,
        int         resize_h = 0,
        int         resize_w = 0)
        : norm_mode(norm), channel_mode(channel),
          target_h(resize_h), target_w(resize_w)
    {
        // Default ImageNet stats for MEAN_STD mode
        int c = static_cast<int>(channel);
        if (c == 0) c = 3; // assume RGB if unknown

        mean_vals.resize(c, T(0));
        std_vals .resize(c, T(1));

        if (norm_mode == NormMode::MEAN_STD && c >= 3) {
            for (int i = 0; i < 3; i++) {
                mean_vals[i] = ImageNetStats<T>::mean[i];
                std_vals [i] = ImageNetStats<T>::std [i];
            }
        }
    }

    // ── Custom mean/std setter ───────────────────────────────────────────────
    void set_mean_std(const std::vector<T>& mean, const std::vector<T>& std) {
        mean_vals = mean;
        std_vals  = std;
    }

    // ────────────────────────────────────────────────────────────────────────
    // load()  — single image → Matrix<T> with shape {H, W, C}
    // ────────────────────────────────────────────────────────────────────────
    Matrix<T> load(const std::string& path) {
        RawImage img = load_raw(path);
        return Matrix<T>(img.data, { (long)img.h, (long)img.w, (long)img.c });
    }

    // ────────────────────────────────────────────────────────────────────────
    // load_tensor() — single image → Tensor_t<T> with shape {H, W, C}
    // ────────────────────────────────────────────────────────────────────────
    Tensor_t<T> load_tensor(const std::string& path) {
        return std::make_shared<Tensor<T>>(load(path));
    }

    // ────────────────────────────────────────────────────────────────────────
    // load_batch()  — list of paths → Matrix<T> with shape {N, H, W, C}
    //
    // All images must have the same H, W, C after optional resize.
    // If no resize is set and images have different sizes, throws.
    // ────────────────────────────────────────────────────────────────────────
    Matrix<T> load_batch(const std::vector<std::string>& paths) {
        if (paths.empty())
            throw std::runtime_error("ImageLoader::load_batch: empty path list.");

        std::vector<T> batch_data;
        int ref_h = -1, ref_w = -1, ref_c = -1;

        for (const auto& p : paths) {
            RawImage img = load_raw(p);

            if (ref_h == -1) {
                ref_h = img.h; ref_w = img.w; ref_c = img.c;
                batch_data.reserve(paths.size() * ref_h * ref_w * ref_c);
            } else if (img.h != ref_h || img.w != ref_w || img.c != ref_c) {
                throw std::runtime_error(
                    "ImageLoader::load_batch: image '" + p +
                    "' has shape (" + std::to_string(img.h) + "," +
                    std::to_string(img.w) + "," + std::to_string(img.c) +
                    ") but expected (" + std::to_string(ref_h) + "," +
                    std::to_string(ref_w) + "," + std::to_string(ref_c) +
                    "). Set target_h/target_w in constructor to auto-resize.");
            }

            batch_data.insert(batch_data.end(), img.data.begin(), img.data.end());
        }

        long N = static_cast<long>(paths.size());
        return Matrix<T>(batch_data, { N, (long)ref_h, (long)ref_w, (long)ref_c });
    }

    // ────────────────────────────────────────────────────────────────────────
    // load_batch_tensor() — list of paths → Tensor_t<T> with shape {N,H,W,C}
    // ────────────────────────────────────────────────────────────────────────
    Tensor_t<T> load_batch_tensor(const std::vector<std::string>& paths) {
        return std::make_shared<Tensor<T>>(load_batch(paths));
    }

    // ────────────────────────────────────────────────────────────────────────
    // load_directory()
    // Loads all images with a given extension from a directory.
    // Returns Matrix<T> {N, H, W, C}. Requires target_h/w set for safety.
    // ────────────────────────────────────────────────────────────────────────
    Matrix<T> load_directory(const std::string& dir_path,
                             const std::string& extension = ".jpg") {
        if (target_h == 0 || target_w == 0)
            throw std::runtime_error(
                "ImageLoader::load_directory: set target_h and target_w in "
                "constructor before loading a directory, images may differ in size.");

        std::vector<std::string> paths;
        for (const auto& entry : std::filesystem::directory_iterator(dir_path)) {
            if (!entry.is_regular_file()) continue;
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == extension)
                paths.push_back(entry.path().string());
        }

        std::sort(paths.begin(), paths.end()); // deterministic order

        if (paths.empty())
            throw std::runtime_error(
                "ImageLoader::load_directory: no '" + extension +
                "' files found in '" + dir_path + "'.");

        std::cout << "[ImageLoader] Found " << paths.size()
                  << " images in " << dir_path << "\n";

        return load_batch(paths);
    }

    // ────────────────────────────────────────────────────────────────────────
    // shape_of()  — query shape without loading pixel data into a Matrix
    //               Useful for sanity checks before batching.
    // ────────────────────────────────────────────────────────────────────────
    shape_t shape_of(const std::string& path) const {
        int w, h, c;
        // stbi_info just reads the header, no pixel decoding
        // Fallback: load and discard  (stbi_info not in our stub)
        int desired = static_cast<int>(channel_mode);
        stbi_uc* px = stbi_load(path.c_str(), &w, &h, &c, desired);
        if (!px) throw std::runtime_error("ImageLoader::shape_of: cannot open " + path);
        stbi_image_free(px);
        int actual_c = (desired == 0) ? c : desired;
        int fh = (target_h > 0) ? target_h : h;
        int fw = (target_w > 0) ? target_w : w;
        return { (long)fh, (long)fw, (long)actual_c };
    }
};

#endif