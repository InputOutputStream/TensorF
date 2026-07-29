#ifndef _TABULAR__HPP
#define _TABULAR__HPP

#include <vector>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <stdexcept>
#include <random>
#include <numeric>

#include "../Types/types.hpp"

enum TYPE {
    CSV,
    TSV,
    XLS,
    XLSX
};

/**
 * Tabular Data Loader
 * Supports CSV and TSV natively via std::ifstream.
 * XLS/XLSX require linking against libxls and OpenXLSX respectively.
 */
template<typename T>
class Tabular {

private:

    size_t row_count = 0;
    size_t col_count = 0;
    bool has_header   = false;
    std::vector<std::string> headers;

    // -----------------------------------------------------------------------
    // File type detection from extension
    // -----------------------------------------------------------------------
    TYPE getFileType(const std::string& path) {
        size_t dot = path.rfind('.');
        if (dot == std::string::npos)
            throw std::runtime_error("Cannot determine file type: no extension found in path: " + path);

        std::string ext = path.substr(dot + 1);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext == "csv")  return CSV;
        if (ext == "tsv")  return TSV;
        if (ext == "xls")  return XLS;
        if (ext == "xlsx") return XLSX;

        throw std::runtime_error("Unsupported file extension: " + ext);
    }

    // -----------------------------------------------------------------------
    // Parse a single line into fields using a given delimiter
    // -----------------------------------------------------------------------
    std::vector<std::string> split_line(const std::string& line, char delim) {
        std::vector<std::string> fields;
        std::istringstream ss(line);
        std::string field;

        while (std::getline(ss, field, delim)) {
            // Strip surrounding whitespace / carriage returns
            field.erase(0, field.find_first_not_of(" \t\r\n"));
            field.erase(field.find_last_not_of(" \t\r\n") + 1);
            fields.push_back(field);
        }
        return fields;
    }

    // -----------------------------------------------------------------------
    // Core delimited-text loader (used for both CSV and TSV)
    // -----------------------------------------------------------------------
    Matrix<T> load_delimited(const std::string& path, char delim) {
        std::ifstream file(path);
        if (!file.is_open())
            throw std::runtime_error("Could not open file: " + path);

        std::vector<std::vector<T>> rows;
        std::string line;
        bool first_line = true;

        while (std::getline(file, line)) {
            if (line.empty()) continue;

            std::vector<std::string> fields = split_line(line, delim);

            // Handle header row
            if (first_line && has_header) {
                headers = fields;
                first_line = false;
                continue;
            }
            first_line = false;

            std::vector<T> row;
            row.reserve(fields.size());

            for (const auto& f : fields) {
                try {
                    row.push_back(static_cast<T>(std::stod(f)));
                } catch (...) {
                    // Non-numeric field — push NaN equivalent
                    row.push_back(static_cast<T>(std::numeric_limits<double>::quiet_NaN()));
                }
            }

            // Validate column consistency
            if (!rows.empty() && row.size() != rows[0].size())
                throw std::runtime_error("Inconsistent column count at row " + std::to_string(rows.size() + 1));

            rows.push_back(row);
        }

        if (rows.empty())
            throw std::runtime_error("File is empty or contains only headers: " + path);

        row_count = rows.size();
        col_count = rows[0].size();

        // Flatten row-major into Matrix
        std::vector<T> flat;
        flat.reserve(row_count * col_count);
        for (auto& r : rows)
            flat.insert(flat.end(), r.begin(), r.end());

        return Matrix<T>(flat, {row_count, col_count});
    }

    Matrix<T> load_csv(const std::string& path)  { return load_delimited(path, ','); }
    Matrix<T> load_tsv(const std::string& path)  { return load_delimited(path, '\t'); }

    // -----------------------------------------------------------------------
    // XLS — requires libxls  (link with -lxlsreader)
    // -----------------------------------------------------------------------
    Matrix<T> load_xls(const std::string& path) {
        // libxls usage:
        //   #include <xls.h>
        //   xls_error_t err = LIBXLS_OK;
        //   xlsWorkBook* wb = xls_open_file(path.c_str(), "UTF-8", &err);
        //   xlsWorkSheet* ws = xls_getWorkSheet(wb, 0);
        //   xls_parseWorkSheet(ws);
        //   then iterate ws->rows.lastrow / ws->rows.lastcol
        //   access cells via ws->rows.row[r].cells.cell[c].d  (double value)
        //   xls_close_WS(ws); xls_close_WB(wb);
        (void)path;
        throw std::runtime_error(
            "XLS loading requires libxls. "
            "Install it and uncomment the implementation in load_xls().");
    }

    // -----------------------------------------------------------------------
    // XLSX — requires OpenXLSX  (link with -lOpenXLSX)
    // -----------------------------------------------------------------------
    Matrix<T> load_xlsx(const std::string& path) {
        // OpenXLSX usage:
        //   #include <OpenXLSX.hpp>
        //   OpenXLSX::XLDocument doc; doc.open(path);
        //   auto ws = doc.workbook().worksheet("Sheet1");
        //   iterate ws.rowCount() / ws.columnCount()
        //   access cells via ws.cell(row, col).value().get<double>()
        //   doc.close();
        (void)path;
        throw std::runtime_error(
            "XLSX loading requires OpenXLSX. "
            "Install it and uncomment the implementation in load_xlsx().");
    }

public:

    // -----------------------------------------------------------------------
    // Constructor
    // -----------------------------------------------------------------------
    explicit Tabular(bool header = false) : has_header(header) {}

    // -----------------------------------------------------------------------
    // Main entry point
    // -----------------------------------------------------------------------
    Matrix<T> load(const std::string& path) {
        TYPE t = getFileType(path);
        switch (t) {
            case CSV:  return load_csv(path);
            case TSV:  return load_tsv(path);
            case XLS:  return load_xls(path);
            case XLSX: return load_xlsx(path);
            default:   throw std::runtime_error("Unsupported format");
        }
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------
    size_t rows() const { return row_count; }
    size_t cols() const { return col_count; }
    const std::vector<std::string>& get_headers() const { return headers; }

    // -----------------------------------------------------------------------
    // Train / test split  — returns {train, test}
    // -----------------------------------------------------------------------
    std::pair<Matrix<T>, Matrix<T>> train_test_split(
            const Matrix<T>& data, float test_ratio = 0.2f, bool shuffle = true) {

        size_t n      = data.shape[0];
        size_t n_test = static_cast<size_t>(std::ceil(n * test_ratio));
        size_t n_train = n - n_test;

        std::vector<size_t> idx(n);
        std::iota(idx.begin(), idx.end(), 0);

        if (shuffle) {
            std::mt19937 rng(42);
            std::shuffle(idx.begin(), idx.end(), rng);
        }

        std::vector<T> train_flat, test_flat;
        size_t cols = data.shape[1];
        train_flat.reserve(n_train * cols);
        test_flat.reserve(n_test  * cols);

        for (size_t i = 0; i < n; i++) {
            auto& target = (i < n_train) ? train_flat : test_flat;
            for (size_t c = 0; c < cols; c++)
                target.push_back(data.data[idx[i] * cols + c]);
        }

        return {
            Matrix<T>(train_flat, {n_train, cols}),
            Matrix<T>(test_flat,  {n_test,  cols})
        };
    }
};

#endif