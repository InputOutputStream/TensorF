#pragma once
#include <vector>
#include <fstream>
#include <string>
#include <stdexcept>

#include "DataLoading.hpp"
#include "core/DataStructures/Matrix.hpp"
#include "core/Types/types.hpp"

template <typename T>
class LazyDataLoader {
    TextDataset<T> ds;
    std::pair<Tensor_t<T>, Tensor_t<T>> get_batch(const std::string& split) {
        return ds.get_batch(split);
    }

public:
    LazyDataLoader(const std::string& path, size_t block_size, size_t batch_size)
        : ds(path, block_size, batch_size)
    {
        ds.load();
    }

    std::pair<Tensor_t<T>, Tensor_t<T>> getNextBatch(const std::string& label = "train") {
        return get_batch(label);
    }
};