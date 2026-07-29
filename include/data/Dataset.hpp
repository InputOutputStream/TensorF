#ifndef __DATASET_HPP
#define __DATASET_HPP

#include <vector>
#include <numeric>
#include <random>
#include <stdexcept>
#include <algorithm>
#include <iostream>

#include "../Types/types.hpp"

/**
 * Dataset
 *
 * Wraps a feature matrix X and label matrix Y.
 * Provides indexed access to individual samples.
 */
template<typename T>
class Dataset {

public:
    Matrix<T> X; // features  [n_samples, n_features]
    Matrix<T> Y; // labels    [n_samples, n_outputs]

    Dataset(Matrix<T> features, Matrix<T> labels)
        : X(features), Y(labels)
    {
        if (X.shape[0] != Y.shape[0])
            throw std::runtime_error(
                "Dataset: X and Y must have the same number of rows.");
    }

    size_t size() const {
        return static_cast<size_t>(X.shape[0]);
    }

    // Returns a single sample as a pair {x_row, y_row}
    std::pair<Matrix<T>, Matrix<T>> get(size_t idx) const {
        if (idx >= size())
            throw std::out_of_range("Dataset index out of range.");

        size_t n_feat = X.shape[1];
        size_t n_out  = Y.shape[1];

        std::vector<T> x_row(X.data.begin() + idx * n_feat,
                              X.data.begin() + idx * n_feat + n_feat);
        std::vector<T> y_row(Y.data.begin() + idx * n_out,
                              Y.data.begin() + idx * n_out  + n_out);

        return {
            Matrix<T>(x_row, {1, n_feat}),
            Matrix<T>(y_row, {1, n_out})
        };
    }
};


/**
 * DataLoader
 *
 * Iterates over a Dataset in mini-batches.
 * Supports shuffling at the start of each epoch.
 *
 * Usage:
 *   DataLoader<float> loader(dataset, 32, true);
 *   for (auto [xb, yb] : loader) { ... }
 */
template<typename T>
class DataLoader {

private:
    const Dataset<T>&    dataset;
    size_t               batch_size;
    bool                 shuffle;
    std::vector<size_t>  indices;
    std::mt19937         rng;

public:

    DataLoader(const Dataset<T>& ds, size_t batch_size, bool shuffle = true, uint32_t seed = 42)
        : dataset(ds), batch_size(batch_size), shuffle(shuffle), rng(seed)
    {
        indices.resize(ds.size());
        std::iota(indices.begin(), indices.end(), 0);
    }

    // -----------------------------------------------------------------------
    // Iterator
    // -----------------------------------------------------------------------
    struct Iterator {
        const DataLoader<T>& loader;
        size_t               cursor;

        Iterator(const DataLoader<T>& l, size_t c) : loader(l), cursor(c) {}

        bool operator!=(const Iterator& other) const {
            return cursor != other.cursor;
        }

        Iterator& operator++() {
            cursor += loader.batch_size;
            if (cursor > loader.dataset.size())
                cursor = loader.dataset.size();
            return *this;
        }

        // Returns {X_batch [batch, n_feat], Y_batch [batch, n_out]}
        std::pair<Matrix<T>, Matrix<T>> operator*() const {
            size_t end  = std::min(cursor + loader.batch_size, loader.dataset.size());
            size_t n  = end - cursor;

            size_t n_feat = loader.dataset.X.shape[1];
            size_t n_out  = loader.dataset.Y.shape[1];

            std::vector<T> xb, yb;
            xb.reserve(n * n_feat);
            yb.reserve(n * n_out);

            for (size_t i = cursor; i < end; i++) {
                size_t idx = loader.indices[i];
                auto [x, y] = loader.dataset.get(idx);
                xb.insert(xb.end(), x.data.begin(), x.data.end());
                yb.insert(yb.end(), y.data.begin(), y.data.end());
            }

            return {
                Matrix<T>(xb, {n, n_feat}),
                Matrix<T>(yb, {n, n_out})
            };
        }
    };

    // Reshuffle at the start of each epoch
    void on_epoch_start() {
        if (shuffle)
            std::shuffle(indices.begin(), indices.end(), rng);
    }

    Iterator begin() {
        on_epoch_start();
        return Iterator(*this, 0);
    }

    Iterator end() const {
        return Iterator(*this, dataset.size());
    }

    size_t num_batches() const {
        return (dataset.size() + batch_size - 1) / batch_size;
    }
};

#endif