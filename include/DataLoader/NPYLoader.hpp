#ifndef __NPY_LOADER_HPP
#define __NPY_LOADER_HPP

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>

#include "../Types/types.hpp"

/**
 * NpyLoader
 *
 * Loads NumPy .npy files (single array) directly into Matrix<T>.
 * .npy format spec: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
 *
 * .npz is a zip archive of multiple .npy files.
 * Full .npz support requires a zip library (miniz or zlib).
 * A stub is provided with instructions.
 */
template<typename T>
class NpyLoader {

private:

    static constexpr char NPY_MAGIC[] = "\x93NUMPY";

    struct NpyHeader {
        int         major_version;
        int         minor_version;
        std::string dtype;       // e.g. "<f4", "<f8", "<i4"
        bool        fortran_order;
        shape_t     shape;
    };

    // -----------------------------------------------------------------------
    // Parse the Python-dict header string embedded in the .npy file
    // -----------------------------------------------------------------------
    NpyHeader parse_header(std::ifstream& f) {
        // Magic: 6 bytes  "\x93NUMPY"
        char magic[6];
        f.read(magic, 6);
        if (std::memcmp(magic, NPY_MAGIC, 5) != 0)
            throw std::runtime_error("Not a valid .npy file: bad magic bytes.");

        uint8_t major, minor;
        f.read(reinterpret_cast<char*>(&major), 1);
        f.read(reinterpret_cast<char*>(&minor), 1);

        // Header length: 2 bytes (v1) or 4 bytes (v2)
        uint32_t header_len = 0;
        if (major == 1) {
            uint16_t hl;
            f.read(reinterpret_cast<char*>(&hl), 2);
            header_len = hl;
        } else if (major == 2) {
            f.read(reinterpret_cast<char*>(&header_len), 4);
        } else {
            throw std::runtime_error("Unsupported .npy version: " + std::to_string(major));
        }

        std::string header(header_len, ' ');
        f.read(&header[0], header_len);

        NpyHeader h;
        h.major_version = major;
        h.minor_version = minor;

        // Extract dtype
        auto dtype_pos = header.find("'descr'");
        if (dtype_pos == std::string::npos)
            throw std::runtime_error("Cannot find 'descr' in .npy header.");
        auto q1 = header.find('\'', dtype_pos + 8);
        auto q2 = header.find('\'', q1 + 1);
        h.dtype = header.substr(q1 + 1, q2 - q1 - 1);

        // Extract fortran_order
        h.fortran_order = header.find("'fortran_order': True") != std::string::npos;
        if (h.fortran_order)
            std::cerr << "[NpyLoader] Warning: Fortran-order array detected. "
                         "Data will be loaded as-is (column-major).\n";

        // Extract shape tuple
        auto shape_pos = header.find("'shape'");
        if (shape_pos == std::string::npos)
            throw std::runtime_error("Cannot find 'shape' in .npy header.");
        auto paren_open  = header.find('(', shape_pos);
        auto paren_close = header.find(')', paren_open);
        std::string shape_str = header.substr(paren_open + 1, paren_close - paren_open - 1);

        std::istringstream ss(shape_str);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            tok.erase(0, tok.find_first_not_of(" \t"));
            tok.erase(tok.find_last_not_of(" \t,") + 1);
            if (!tok.empty())
                h.shape.push_back(static_cast<long>(std::stol(tok)));
        }

        return h;
    }

    // -----------------------------------------------------------------------
    // Compute total number of elements from shape
    // -----------------------------------------------------------------------
    size_t total_elements(const shape_t& shape) {
        size_t n = 1;
        for (auto d : shape) n *= static_cast<size_t>(d);
        return n;
    }

public:

    // -----------------------------------------------------------------------
    // Load a single .npy file into Matrix<T>
    // The dtype of the file must match T (float32 for T=float, etc.)
    // -----------------------------------------------------------------------
    Matrix<T> load_npy(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f.is_open())
            throw std::runtime_error("Cannot open .npy file: " + path);

        NpyHeader h = parse_header(f);

        // Warn if dtype mismatch
        std::cout << "[NpyLoader] dtype=" << h.dtype
                  << "  sizeof(T)=" << sizeof(T) << "\n";

        size_t n = total_elements(h.shape);
        std::vector<T> data(n);
        f.read(reinterpret_cast<char*>(data.data()), n * sizeof(T));

        if (!f)
            throw std::runtime_error("Failed to read full array data from: " + path);

        return Matrix<T>(data, h.shape);
    }

    // -----------------------------------------------------------------------
    // .npz support stub
    //
    // .npz is just a ZIP archive. Each member is a .npy file named after the
    // array key (e.g. "weight.npy", "bias.npy").
    //
    // To implement fully:
    //   1. Link against miniz (single-header) or zlib.
    //   2. Open the zip, iterate entries.
    //   3. Decompress each entry into a memory buffer.
    //   4. Parse the buffer as .npy (reuse parse_header with istringstream).
    //   5. Return a map<string, Matrix<T>>.
    //
    // Example with miniz:
    //   mz_zip_archive zip = {};
    //   mz_zip_reader_init_file(&zip, path.c_str(), 0);
    //   int n = mz_zip_reader_get_num_files(&zip);
    //   for each file: mz_zip_reader_extract_to_heap(...)
    // -----------------------------------------------------------------------
    std::map<std::string, Matrix<T>> load_npz(const std::string& path) {
        (void)path;
        throw std::runtime_error(
            ".npz loading requires miniz or zlib. "
            "See the comment in NpyLoader::load_npz() for integration steps.");
    }
};

#endif