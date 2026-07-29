#ifndef __MATRIX_CLASS_INCLUDED__
#define __MATRIX_CLASS_INCLUDED__

#include "Types/types.hpp"
#include "Overloads/Overload.hpp"

#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <memory>
#include <algorithm>
#include <random>
#include <cblas.h>
#include <optional>

template<typename U>
class Matrix;

template<typename T>
class Broadcast{

    protected:

        bool assertBroadcast(Matrix<T> t1, Matrix<T> t2)
        {

            shape_t s1 = t1.shape;
            shape_t s2 = t2.shape;

            if(s1.size() == s2.size())
            {
                for(int i = s1.size()-1; i >= 0; i--)
                {
                    if(s1[i] != s2[i] && (s2[i] != 1 && s1[i] != 1))
                        return 0;
                }

                return 1;
            }

            if(s1.size() < s2.size())
            { 
                int i, j;
                for(i = s1.size()-1, j = s2.size()-1; i >= 0 && j >= 0; i--, j--)
                {
                    if(s1[i] != s2[j] && (s2[j] != 1 && s1[i] != 1))
                        return false;
                }

                return true;
            }

            return assertBroadcast(t2, t1);
        }

        shape_t computeBroadcastResultShape(Matrix<T> t1, Matrix<T> t2)
        {
            if(assertBroadcast(t1, t2)== false)
                throw std::runtime_error("Invalid broadcast operation");


            int i, j;
            shape_t s1 = t1.shape;
            shape_t s2 = t2.shape;
            shape_t resShape;

            for(i = s1.size()-1, j = s2.size()-1; i >= 0 && j >= 0; i--, j--)
            {
                resShape.push_back(std::max(s1[i], s2[j]));
            }     

            if(s1.size() < s2.size())
            {
                size_t n  =  s2.size() - s1.size();
                for(int k = (n-1); k >= 0; k--){
                    resShape.push_back(std::max((size_t)1, s2[k]));
                }
            }
            else if(s1.size() > s2.size())
            {
                size_t n  =  s1.size() - s2.size();
                for(int k = (n-1); k >= 0; k--){
                    resShape.push_back(std::max((size_t)1, s1[k]));
                }        

            }

            std::reverse(resShape.begin(), resShape.end());
            return resShape;
        }

        shape_t computeShapes(const shape_t shape)
        {
            shape_t numElementsSeen(shape.size());
            size_t p{1};
            for(int i = shape.size()-1; i>=0 ;i--)
            {
                numElementsSeen[i] = p;
                p *= shape.at(i);
            }

            return numElementsSeen;
        }

    public: 
        
        std::pair<Matrix<T>, Matrix<T>> broadcast(Matrix<T> t1, Matrix<T> t2){
            shape_t resShape = this->computeBroadcastResultShape(t1, t2);
            return std::make_pair(this->broadcastTo(t1, resShape), this->broadcastTo(t2, resShape));
        }
       
        Matrix<T> broadcastTo(Matrix<T> source, shape_t new_shape)
        {
            size_t ne=1;
            shape_t nr = computeShapes(new_shape);
            shape_t ns = computeShapes(source.shape);
            size_t offset = new_shape.size() - source.ndims;

            std::vector<T> res;

            // Fail loudly here rather than segfaulting silently inside the loop.
            size_t expectedSourceSize = 1;
            for (size_t s : source.shape) expectedSourceSize *= s;
            if (source.data.size() < expectedSourceSize)
                throw std::runtime_error("Broadcast::broadcastTo: source matrix has shape " +
                    std::to_string(expectedSourceSize) + " elements but data vector has only " +
                    std::to_string(source.data.size()) + " — was the matrix constructed without initializing its data?");

            for(auto s: new_shape)
                ne *= s;

            for(size_t i = 0; i<ne; i++)
            {
                shape_t new_index;
                size_t id = i;
                for(auto j: nr)
                {
                    new_index.push_back((size_t)(id / j));
                    id = id%j;
                }

                for(int k = new_shape.size()-1; k >= 0; k--)
                {

                    if((size_t)k < offset)
                        new_index[k] = 0;
                    if((size_t)k >= offset)    
                    {
                        if(source.shape[(size_t)k - offset] == 1)
                            new_index[k] = 0;
                    }
                }

                size_t npos = 0;
                for(size_t t = 0; t <source.shape.size(); t++)
                {
                    npos += ns[t] * new_index[t + offset];
                }

                res.push_back(source.data[npos]);
            }

            return Matrix<T>(res, new_shape);
        }

         /**
         * 
         * 
                The algorithm:

                Compute total number of elements in the result shape
                For each flat index k in 0..total:

                    Convert k to a multi-index in the result shape (this is just repeated division/modulo — you already do this in your transpose code)
                    For each dimension, if the source size on that dimension is 1, clamp that index component to 0, otherwise keep it
                    Convert the clamped multi-index back to a flat index in the source
                    Copy source.data[flat_source_index] into result.data[k]


                Return a Matrix with the new data and new_shape

                You already have computeShapes which gives you the stride array (elements per step in each dimension) — 
                that's exactly what you need for the flat↔multi-index conversion. Look at how your transpose method does it, the index decomposition logic is identical.
         */
};


template <typename T>
class Matrix 
{

    protected:
    shape_t numElementsSeen{}; 
    Broadcast<T> b;

    bool verifyShape(const std::vector<T> &data, const shape_t &shape)
    {
        size_t p = 1;
        for(size_t i = 0; i < shape.size(); i++) {
            p *= shape[i];
        }
        // Allow the data vector to be exactly the logical size OR tail-padded for AVX2
        return (data.size() == p || data.size() == avx2_pad(p));
    }         

     //There is an error in the computes shapes method as we go from 1D to 2D 
        //Solution
        //I was using a class attr this->numElementsSeen instead of a local variable numElementsSeen which xas wronf since i am returning it
    shape_t computeShapes(const shape_t shape)
    {
        shape_t numElementsSeen(shape.size());
        size_t p{1};
        for(int i = shape.size()-1; i>=0 ;i--)
        {
            numElementsSeen[i] = p;
            p *= shape.at(i);
        }

        return numElementsSeen;
    }

    template <typename U> // cloudy
    void extractShape(const U& data, shape_t& shape)
    {
        if constexpr(std::is_same_v<U, T>){ 
            return; // scalar
        }
        else{
            this->shape.push_back(data.size());
            extractShape(*data.begin(), shape);
        }
    }

    bool dotShapesAssert(const shape_t &shape)
    {
        if(this->shape.size()  == 1 || shape.size() == 1)
            return false;

        size_t second_to_last_dim = shape.size()-2;
        if(shape[second_to_last_dim] != this->shape.back())
        {
            return false;
        }

        return true;
    }

    static std::mt19937& get_gen(std::optional<unsigned int> seed = std::nullopt) {
        static std::mt19937 gen(std::random_device{}());
        if(seed.has_value())
            gen.seed(seed.value());
        return gen;
    }
            
    bool isRegular2D(const std::vector<std::vector<T>> data)
    {
        if(data.size() == 0)
            return true;

        std::vector<T> j = data[0];
        for(size_t i=1; i<data.size(); i++)
        { 
            if(j.size() != data[i].size())
                return false;
        }

        return true;
    }

    bool isRegular2D(const std::initializer_list<std::initializer_list<T>>& data)
    {
        if (data.size() == 0) return true;

        size_t cols = data.begin()->size();

        for (const auto& row : data)
        {
            if (row.size() != cols)
                return false;
        }
        return true;
    }

    bool isRegular3D(const std::initializer_list<std::initializer_list<std::initializer_list<T>>>& data)
    {
        if (data.size() == 0) return true;

        size_t dim1 = data.begin()->size();
        size_t dim2 = data.begin()->begin()->size();

        for (const auto& row : data)
        {
            if (row.size() != dim1)
                return false;
        
            for (const auto& subrow : row)
            {
                if (subrow.size() != dim2)
                return false;
            }
        }

        return true;
    }

        
    // Check if shapes are equal element wise in the std::vector 

    static inline size_t avx2_pad(size_t n) {
        return ((n + 7) / 8) * 8;
    } // std::vector<T> data(avx2_pad(n), T(0));


    bool isShape1D()
    {
        if(this->shape.size() == 1)
            return true;

        return false;
    }

    bool isShape2D()
    {
        if(this->shape.size() == 2)
            return true;
        
        return false;
    }


    bool areShapes1D(const shape_t &lshape, const shape_t &rshape)
    {
        if(rshape.size() == 1 && lshape.size() == 1)
            return true;

        return false;
    }

    bool areShapes2D(const shape_t &lshape, const shape_t &rshape)
    {
        if(lshape.size() == 2 && rshape.size() == 2)
            return true;
        
        return false;
    }

//°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°

    T sum_1D()
    {
        T s = 0;
        for(auto i : this->data)
            s+=i;
        return s;
    }


    T sum_1D(std::vector<T> data_1D)
    {
        T s = 0;
        for(auto i: data_1D)
            s+=i;
        return s;
    }


    void sum_2D(int axis, size_t lhsStart, std::vector<T> &res)
    {
        if (axis > 1)
            throw std::runtime_error("Invalid input axis recieved for sum 2D\n");

        if(axis == 1)
        {
            T s = 0;
            size_t index = 0;
            for(size_t i = 0; i<this->shape[this->ndims - 2]; i++)
            {
                for(size_t j = 0; j<this->shape[this->ndims - 1]; j++)
                {
                    s += this->data[lhsStart + index + j];
                }
                index = this->shape[this->ndims - 1] * (i+1);   
                res.push_back(s); 
                s = 0;

            }

            // res.shape.push_back(this->shape[0]);
        }

        else if(axis == 0)
        {
            T s = 0;
            for(size_t i = 0; i<this->shape[this->ndims - 1]; i++)
            {
                for(size_t j = 0; j<this->shape[this->ndims - 2]; j++)
                {
                    s += this->data[lhsStart + j * this->shape[this->ndims - 1] + i];
                }
                res.push_back(s); 
                s = 0;
            }
            // res.shape.push_back(this->shape[1]);
        }
    }

    void _sum_(int axis, size_t lhsStart, std::vector<T> &res)
    {
        if (axis > 1)
            throw std::runtime_error("_sum_: axis must be 0 or 1\n");
        if(axis == 1)
        {
            T s = 0;
            size_t index = 0;
            for(size_t i = 0; i<this->shape[this->ndims - 2]; i++)
            {
                for(size_t j = 0; j<this->shape[this->ndims - 1]; j++)
                {
                    s += this->data[lhsStart + index + j];
                }
                index = this->shape[this->ndims - 1] * (i+1);   
                res.push_back(s); 
                s = 0;

            }

        }

        else if(axis == 0)
        {
            int nslice = (this->data.size() - lhsStart) / this->shape[0];
            T s = 0;
        
            for(int i = 0; i<nslice; i++){

                for(size_t j = 0; j<this->shape[0]; j++)
                {
                    s+=this->data[lhsStart + j * nslice + i];
                }
            res.push_back(s);
            s=0;

            }
        }
    }


    void sum(std::vector<T> &res, 
                shape_t &indexStack, size_t lhsStart,
                size_t axis,
                size_t dim)  {

        if (dim >= this->shape.size())
            throw std::runtime_error("Sum: invalid sum dimension");
        
        if(this->isShape1D())
            {
                res.push_back(this->sum_1D());
                return;
            }

        if(this->isShape2D())
            {
                int local_axis = (axis == this->ndims - 1) ? 1 : 0;
                this->sum_2D(local_axis, 0, res);
                return;
            }

        if(axis== 0)
            {
                int local_axis = (axis == this->ndims - 1) ? 1 : 0;
                this->_sum_(local_axis, lhsStart, res);
                return;
            }
            
        if(indexStack.size()  == (this->shape.size()-2))
        {            
            for(size_t i{0}; i<indexStack.size(); i++)
            {
                lhsStart += indexStack.at(i) * this->numElementsSeen.at(i);
            }

            int local_axis = (axis == this->ndims - 1) ? 1 : 0;
            this->sum_2D(local_axis, lhsStart, res);
            return;
        }

        // Push the extra dimensions to the index stack and recursively traverse the indices, then pop one once the operation for that index has been done
        if (dim == axis)
        {
            this->sum(res, indexStack, lhsStart, axis, dim+1);
            return;
        }
        
        for(size_t i=0; i<this->shape[dim]; i++)
        {
            indexStack.push_back(i);
            this->sum(res, indexStack, lhsStart, axis, dim+1);
            indexStack.pop_back(); 
        }
    }

//°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°

    Matrix<T> transpose_1D()
    {            
        if(this->shape.size() == 2 && (this->shape[0] == 1))
           return Matrix<T>(this->data, {this->shape[1], 1});
        else if(this->shape.size() == 2 && (this->shape[1] == 1))
           return Matrix<T>(this->data, {1, this->shape[0]});
        else if(this->shape.size() == 1)
           return Matrix<T>(this->data, {this->shape[0], 1});
        else{
                throw std::runtime_error("transpose_1D: invalid shape for 1D transpose\n");
            }
    }

    std::vector<T> transpose_2D()
        {            
            size_t row = this->shape[this->ndims - 2];
            size_t col = this->shape[this->ndims - 1];

            std::vector<T> res = this->data;

            for(size_t i=0; i<row; i++)
            {
                for(size_t j=0; j<col; j++)
                {
                    res[j*row + i] = this->data[i*col + j];
                }
            }

        return res;
    }

    void transpose(const shape_t perm, const shape_t resShape, std::vector<T>& res)
    {
        auto ns = this->numElementsSeen;          // src strides
        auto nr = this->computeShapes(resShape);  // dst strides

        size_t dsize = 1;
        for (size_t s : resShape) dsize *= s;
        res.reserve(dsize);

        for (size_t i = 0; i < dsize; i++)
        {
            // decompose i into dst multi-index
            shape_t dst_idx(resShape.size());
            size_t k = i;
            for (size_t d = 0; d < resShape.size(); d++) {
                dst_idx[d] = k / nr[d];
                k          = k % nr[d];
            }

            // map through permutation to src index
            size_t npos = 0;
            for (size_t d = 0; d < resShape.size(); d++)
                npos += ns[perm[d]] * dst_idx[d];

            res.push_back(this->data[npos]);
        }
    }

    // void transpose(const shape_t resShape, std::vector<T> &res)
    // {
    //     Matrix<T> temp(this);
    //     auto ns = temp.numElementsSeen;
    //     auto nr = temp.computeShapes(resShape);

    //     size_t dsize = 1; // num elements
    //     for(size_t i: resShape)
    //     {
    //         dsize *= i;
    //     }

    //     for(size_t i = 0; i<dsize; i++)
    //     {
    //         shape_t new_index;
    //         auto k = i;
    //         for(auto j: nr)
    //         {
    //             new_index.push_back((size_t)(k / j));
    //             k = k%j;
    //         }

    //         shape_t rev;
    //         rev.insert(rev.end(), new_index.rbegin(),  new_index.rend());
    //         size_t npos = 0;
    //         for(size_t id = 0; id <temp.shape.size(); id++)
    //         {
    //             npos += ns[id] * rev[id];
    //         }

    //         res.push_back(temp.data[npos]);
    //     }
    // }

//°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°

    T dotProduct1D(const std::vector<T> &lhs, const std::vector <T> &rhs){
            T sum = 0;
            for(size_t i = 0, j = 0; i<lhs.size() && j<rhs.size(); i++, j++){
                sum += lhs.at(i) * rhs.at(j);
            }
            return sum;
        }
    
    Matrix<T> dotProduct2D(const Matrix<T> &rhs)
    {
        
        T result = T(0);

        if constexpr (std::is_same_v<T, float>)
            result = cblas_sdot(
                this->data.size(),   // number of elements
                this->data.data(), 1, // vector A, stride 1
                rhs.data.data(), 1  // vector B, stride 1
            );
        else if constexpr (std::is_same_v<T, double>)
            result = cblas_ddot(
                this->data.size(),
                this->data.data(), 1,
                rhs.data.data(), 1
            );
        else {
            for (size_t i = 0; i < this->data.size(); i++)
                result += this->data[i] * rhs.data[i];
        }

        return Matrix<T>({result});
    }


    Matrix<T> matProduct2D(const Matrix<T>& rhs,
                        size_t lhsStart,
                        size_t rhsStart,      
                        size_t resStart,
                        std::vector<T>& result)
    {
        // Slice dimensions: last 2 axes of each operand
        size_t M = this->shape[this->shape.size() - 2];
        size_t K = this->shape[this->shape.size() - 1];
        size_t N = rhs.shape[rhs.shape.size() - 1];

        if constexpr (std::is_same_v<T, float>)
            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K,
                1.0f,
                this->data.data() + lhsStart, K,   // ← offset into lhs slice
                rhs.data.data()  + rhsStart,  N,   // ← offset into rhs slice
                0.0f,
                result.data()    + resStart,  N    // ← offset into output slice
            );
        else if constexpr (std::is_same_v<T, double>)
            cblas_dgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K,
                1.0,
                this->data.data() + lhsStart, K,
                rhs.data.data()  + rhsStart,  N,
                0.0,
                result.data()    + resStart,  N
            );
        else
        {
            for (size_t i = 0; i < M; i++)
                for (size_t k = 0; k < N; k++) {
                    T sum = 0;
                    for (size_t j = 0; j < K; j++)
                        sum += this->data[lhsStart + i*K + j]
                            * rhs.data  [rhsStart + j*N + k]; // ← rhsStart
                    result[resStart + i*N + k] = sum;
                }
        }
        return Matrix<T>();
    }

    Matrix<T> matmul(const Matrix<T>& rhs,
                    shape_t& indexStack,
                    shape_t& resElements,
                    size_t dim,
                    std::vector<T>& out)   
    {
        if (indexStack.size() == (this->shape.size() - 2))
        {
            size_t lhsStart{0}, rhsStart{0}, resStart{0};

            size_t rhs_batch_dims = (rhs.shape.size() >= 2) ? rhs.shape.size() - 2 : 0;
            for (size_t i = 0; i < indexStack.size(); i++) {
                lhsStart += indexStack[i] * this->numElementsSeen[i];
                resStart  += indexStack[i] * resElements[i];
                if (i < rhs_batch_dims)
                    rhsStart += indexStack[i] * rhs.numElementsSeen[i];
                // rhs is 2D weight: rhsStart stays 0 for all batches
            }

            matProduct2D(rhs, lhsStart, rhsStart, resStart, out);
            return Matrix<T>();
        }

        for (size_t i = 0; i < this->shape[dim]; i++) {
            indexStack.push_back(i);
            this->matmul(rhs, indexStack, resElements, dim + 1, out);
            indexStack.pop_back();
        }
        return Matrix<T>();
    }

//°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°

    protected: 
        mutable size_t size;
        mutable size_t ndims;
        bool gpu_nv = false;
        bool gpu_it = false;

    public:
    std::vector<T> data;
    shape_t shape;
    bool gpu = false;

    // Constructors 


    static shape_t getShape(const std::initializer_list<size_t> shape)
    {
        if (shape.size() == 0) return shape_t{0};

        shape_t s;

        for (const auto& item : shape)
        {   
            s.push_back(item);
        }
        return s;
    }
    
    Matrix(){
        this->data.clear();
        this->shape.clear();
        this->numElementsSeen.clear();
        this->ndims = 0; 
        this->size  = 0;
    };

    // NOTE: template<U> scalar constructor removed — Matrix(const T&) handles int/float/double scalars.
    // Keeping it caused ambiguity when T == float or T == int.

    // requires (std::is_arithmetic_v<T>)
    Matrix(const T& indata)
    {
        this->shape.push_back(1);
        this->data.push_back(indata);
        this->numElementsSeen = computeShapes(this->shape);
        this->ndims = this->shape.size();
        this->size  = this->data.size();
    }

    Matrix(const Matrix<T>* two)
    {
        if (two == nullptr)
            throw std::runtime_error("Matrix(ptr): null pointer input\n");

        this->data = two->data;
        this->shape = two->shape; 
        this->numElementsSeen = two->numElementsSeen;
        this->ndims = two->shape.size();
        this->size  = this->data.size();
    }

    Matrix(const Matrix<T>& two)
    {
        this->data = two.data;
        this->shape = two.shape; 
        this->numElementsSeen = two.numElementsSeen;
        this->ndims = two.shape.size();
        this->size  = two.data.size();
    }
 
    Matrix(std::vector<T> indata)
    {
        size_t logical = indata.size();
        this->shape.push_back(logical);

        // indata.resize(avx2_pad(logical), T(0));  // pad BEFORE storing
        this->data = indata;

        this->numElementsSeen = this->computeShapes(this->shape);
        this->ndims = this->shape.size();
        this->size  = logical;   // logical count
    }

    Matrix(std::vector<T> indata, shape_t inshape)
    {
        if (!verifyShape(indata, inshape))
            throw std::runtime_error("Matrix: shape and number of elements do not match");

        // Compute the true logical size from the shape dimensions
        size_t logical = 1;
        for (size_t s : inshape) logical *= s;

        this->data = indata;
        this->shape = inshape;
        this->numElementsSeen = this->computeShapes(this->shape);
        this->ndims = this->shape.size();
        this->size  = logical; // Keep logical size clean of tail padding
    }

    Matrix(std::vector<std::vector<T>> indata)
    {
        // 1. shape
        this->shape.push_back(indata.size());
        this->shape.push_back(indata.begin()->size());
        // 2. validate
        if (this->isRegular2D(indata) == false)
            throw std::runtime_error("Matrix: shape must be uniform\n");
        // 3. data
        this->flattenReccursive(indata, this->data);
        // 4. derived fields — exactly once
        this->numElementsSeen = this->computeShapes(this->shape);
        this->ndims = this->shape.size();
        this->size  = this->data.size();
    }

    Matrix(std::vector<std::vector<T>> indata, std::initializer_list<size_t> inshape)
    {
        // 1. shape
        this->shape = Matrix<T>::getShape(inshape);
        // 2. validate regularity
        if (this->isRegular2D(indata) == false)
            throw std::runtime_error("Matrix: shape must be uniform\n");
        // 3. data
        this->flattenReccursive(indata, this->data);
        // 4. validate element count vs shape
        if (this->verifyShape(this->data, this->shape) == false)
            throw std::runtime_error("Matrix: shape and number of elements do not match\n");
        // 5. derived fields — exactly once
        this->numElementsSeen = this->computeShapes(this->shape);
        this->ndims = this->shape.size();
        this->size  = this->data.size();
    }

    Matrix(std::vector<T> indata, std::initializer_list<size_t> inshape)
    {
        this->shape = Matrix<T>::getShape(inshape);
        if (!this->verifyShape(indata, this->shape))   // verify on original size
            throw std::runtime_error("Matrix: shape and number of elements do not match\n");

        // indata.resize(avx2_pad(indata.size()), T(0));  // pad AFTER verify
        this->data = indata;
        this->numElementsSeen = this->computeShapes(this->shape);
        this->ndims = this->shape.size();
        this->size  = this->data.size();
    }

    Matrix(std::initializer_list<std::initializer_list<T>> indata)
    {
        // 1. shape
        this->shape.push_back(indata.size());
        this->shape.push_back(indata.begin()->size());
        // 2. validate
        if (this->isRegular2D(indata) == false)
            throw std::runtime_error("Matrix: shape must be uniform\n");
        // 3. data
        this->flattenReccursive(indata, this->data);
        // 4. derived fields — exactly once
        this->numElementsSeen = this->computeShapes(this->shape);
        this->ndims = this->shape.size();
        this->size  = this->data.size();
    }

    Matrix(std::initializer_list<T> indata, std::initializer_list<size_t> inshape)
    {
        // 1. shape
        this->shape = Matrix<T>::getShape(inshape);
        // 2. data
        this->flattenReccursive(indata, this->data);
        // 3. validate
        if (this->verifyShape(this->data, this->shape) == false)
            throw std::runtime_error("Shape and number of elements of matrix do not match!!!\n");
        // 4. derived fields — exactly once
        this->numElementsSeen = this->computeShapes(this->shape);
        this->ndims = this->shape.size();
        this->size  = this->data.size();
    }

    Matrix(std::initializer_list<std::initializer_list<std::initializer_list<T>>> indata)
    {
        // 1. shape
        this->shape.push_back(indata.size());
        this->shape.push_back(indata.begin()->size());
        this->shape.push_back(indata.begin()->begin()->size());
        // 2. validate
        if (this->isRegular3D(indata) == false)
            throw std::runtime_error("Matrix shape must be uniform!!!\n");
        // 3. data
        this->flattenReccursive(indata, this->data);
        // 4. derived fields — exactly once
        this->numElementsSeen = this->computeShapes(this->shape);
        this->ndims = this->shape.size();
        this->size  = this->data.size();
    }

    template <typename U> 
    void flattenReccursive(const U& data, std::vector<T> &out )
    {
        if constexpr (std::is_same_v<U, T>)
        {
            out.push_back(data);
        }     
        else
        {
            for (const auto& elem : data)
                flattenReccursive(elem, out);
        }
    }

//°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°

    // Matrix Arithmetic Operations 
    

    Matrix<T> operator + (const Matrix<T> &rhs)
    {
        if(this->shape == rhs.shape)
            return Matrix<T>(data + rhs.data, shape);
        else
        {
            auto res = b.broadcast(*this, rhs);
            return Matrix<T>(res.first.data + res.second.data, res.first.shape);

        }
    }

    Matrix<T> operator -()
    {
        return Matrix<T>(-data, shape);
    }
    
    Matrix<T> operator -(const Matrix<T> &rhs)
    {
        if(this->shape == rhs.shape)
            return Matrix<T>(data - rhs.data, shape);
        else
        {
            auto res = b.broadcast(*this, rhs);
            return Matrix<T>(res.first.data - res.second.data, res.first.shape);

        }
    }

    Matrix<T> operator * (const Matrix<T> &rhs)
    {
        if(this->shape == rhs.shape)
            return Matrix<T>(data * rhs.data, shape);
        else
        {
            auto res = b.broadcast(*this, rhs);
            return Matrix<T>(res.first.data * res.second.data, res.first.shape);

        }
    }  
    
    Matrix<T> operator / (const Matrix<T> &rhs)
    {
        if(this->shape == rhs.shape)
            return Matrix<T>(data / rhs.data, shape);
        else
        {
            auto res = b.broadcast(*this, rhs);
            return Matrix<T>(res.first.data / res.second.data, res.first.shape);

        }
    }

    Matrix<T>& operator=(const Matrix<T>& rhs)
    {
        this->data = rhs.data;
        this->shape = rhs.shape;
        this->numElementsSeen = rhs.numElementsSeen;
        this->ndims = rhs.ndims;
        this->size = rhs.size;
        return *this;
    }

    bool operator ==(const Matrix<T> &rhs)
    {
        return (this->shape == rhs.shape) && (this->data == rhs.data);
    }

    template <typename U>
    requires std::is_arithmetic_v<U>
    Matrix<bool> operator==(const U val) {
        std::vector<bool> res;
        res.reserve(this->data.size());
        for (auto& x : this->data)
            res.push_back(x == static_cast<T>(val));
        return Matrix<bool>(res, this->shape);
    }

    template <typename U>
    requires std::is_arithmetic_v<U>
    Matrix<bool> operator!=(const U val) {
        std::vector<bool> res;
        res.reserve(this->data.size());
        for (auto& x : this->data)
            res.push_back(x != static_cast<T>(val));
        return Matrix<bool>(res, this->shape);
    }

    Matrix<T> pow(const T rhs)
    {
        return Matrix<T>(pow(data , rhs), shape);
    }

    Matrix<T> pow(const  Matrix<T> rhs)
    {
        return Matrix<T>(pow(data, rhs.data), shape);
    }

    Matrix<T> exponent() 
    {
        std::vector<T> arr;
        auto n = this->data.size();
        arr.reserve(n);
        for(size_t i=0; i<n; i++)
        { 
            T prod = (T)std::exp(this->data.at(i));
            arr.push_back(prod);
        }
        return Matrix<T>(arr, this->shape);
    } 

    Matrix<T> mean(){
       return this->sum() /(T)this->data.size;
    }

    Matrix<T> std(){
        return this->variance().sqrt();
    }

    Matrix<T> variance(){

        Matrix<T> s(0);
        auto mn = mean();

        for(auto d: this->data){
            s = s + (((T)(d - mn.data[0]))*((T)(d - mn.data[0])));
        }

       return s / this->size;
    }


     // ── axis reductions (n-D) ────────────────────────────────────────────────
 
    Matrix<T> mean(size_t axis)
    {
        if (axis >= this->shape.size())
            throw std::runtime_error("mean: axis out of range\n");
 
        Matrix<T> s = this->sum(axis);
 
        T n = (T)this->shape[axis];   // number of elements collapsed
        std::vector<T> res;
        res.reserve(s.data.size());
        for (auto v : s.data)
            res.push_back(v / n);
 
        return Matrix<T>(res, s.shape);
    }
 
    Matrix<T> var(size_t axis, bool ddof0 = true)
    {
        if (axis >= this->shape.size())
            throw std::runtime_error("var: axis out of range\n");
 
        Matrix<T> mn = this->mean(axis);          // shape: this->shape minus axis dim
 
        // Broadcast mn back to original shape so we can subtract element-wise.
        // Insert the reduced axis back as size-1, then broadcastTo original shape.
        shape_t exp_shape = mn.shape;
        exp_shape.insert(exp_shape.begin() + axis, 1);
        Matrix<T> mn_bc = this->b.broadcastTo(Matrix<T>(mn.data, exp_shape), this->shape);
 
        // Squared deviations, then sum along axis, then divide.
        Matrix<T> diff = *this - mn_bc;           // element-wise subtract
        Matrix<T> sq   = diff * diff;             // element-wise square
 
        Matrix<T> sq_sum = sq.sum(axis);
 
        T n = (T)(ddof0 ? this->shape[axis] : this->shape[axis] - 1);
        std::vector<T> res;
        res.reserve(sq_sum.data.size());
        for (auto v : sq_sum.data)
            res.push_back(v / n);
 
        return Matrix<T>(res, sq_sum.shape);
    }
 
    Matrix<T> std(size_t axis, bool ddof0 = true)
    {
        if (axis >= this->shape.size())
            throw std::runtime_error("std: axis out of range\n");
 
        return this->var(axis, ddof0).sqrt();
    }
 

    Matrix<T> sqrt() 
    {
        std::vector<T> arr;
        auto n = this->data.size();
        arr.reserve(n);
        for(size_t i=0; i< n; i++)
        { 
            T prod = (T)std::sqrt(this->data.at(i));
            arr.push_back(prod);
        }
        return Matrix<T>(arr, this->shape);
    } 

    Matrix<T> cbrt() 
    {
        std::vector<T> arr;
        auto n = this->data.size();
        arr.reserve(n);
        for(size_t i=0; i<n; i++)
        { 
            T prod = (T)std::cbrt(this->data.at(i));
            arr.push_back(prod);
        }
        return Matrix<T>(arr, this->shape);
    } 

    Matrix<T> ln() 
    {
        std::vector<T> arr;
        auto n = this->data.size();
        arr.reserve(n);
        for(size_t i=0; i< n; i++)
        { 
            T prod = (T)std::log(std::max(this->data.at(i), 1e-9f));
            arr.push_back(prod);
        }
        return Matrix<T>(arr, this->shape);
    } 
//°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°

    size_t get_size() const
    {
        this->size = this->data.size();
        return this->data.size();
    }

    size_t get_ndims() const
    {
        this->ndims = this->shape.size();
        return this->shape.size();
    }

    Matrix<T> col(size_t idx) {
        if(this->shape.size() == 1)
            return Matrix<T>(this);

        if(idx >= this->shape[1])
            throw std::runtime_error("Invalid Row Index");

        size_t rows = this->shape[0];
        size_t cols = this->shape[1];
        std::vector<T> res;
        for(size_t i = 0; i < rows; i++){
            res.push_back(this->data[cols * i + idx]);
        }
        return Matrix<T>(res, {rows, 1});
    }


    Matrix<T> at(std::initializer_list<size_t> inshape)
    {
        shape_t index = Matrix<T>::getShape(inshape);

        if(index.size() > this->shape.size())
            throw std::runtime_error("Invalid Index");

        if(index.size() <= this->shape.size())
        {   
            auto i = index[0];
            Matrix<T> temp(this->row(i));
            index.erase(index.begin());
            if(index.size() == 0)
                return Matrix<T>(temp);
            else
                return Matrix<T>(temp.flatten().at(index));
        }
        else 
            throw std::runtime_error("Invalid shape");
    }

    Matrix<T> at(shape_t index)
    {
        if(index.size() > this->shape.size())
            throw std::runtime_error("Invalid Index");

        if(index.size() <= this->shape.size())
        {   
            auto i = index[0];
            Matrix<T> temp(this->row(i));
            index.erase(index.begin());
            if(index.size() == 0)
                return Matrix<T>(temp);
            else
                return Matrix<T>(temp.flatten().at(index));
        }
        else 
            throw std::runtime_error("Invalid shape");
    }

    Matrix<T> at(Matrix<bool> index)
    {
        if(index.get_size() > this->shape.size())
            throw std::runtime_error("Invalid Index matrix");

        if(index.get_size() <= this->shape.size())
        {           
            std::vector<T> res;
            for(size_t i=0; i< this->data.size(); i++)
            {
                if(index.data.at(i))
                    res.push_back(this->data.at(i));
                else
                    res.push_back(0);
            }
        }
        else 
            throw std::runtime_error("Index Matrix not of the same size");
    }

    Matrix<T> row(size_t idx) {
        if(this->shape.size() == 1)
            return Matrix<T>(this);
        
        if(idx >= this->shape[0])
            throw std::runtime_error("Invalid Row Index");

        size_t cols = this->shape[1];
        std::vector<T> res;
        res.reserve(cols);
        for(size_t i = 0; i < cols; i++){
            res.push_back(this->data[idx * cols+i]);
        }
        return Matrix<T>(res, {1, cols});
    }

    Matrix<T> slice_row(size_t start, size_t end) {
        if(this->shape.size() == 1)
            throw std::runtime_error("not impl for 1D");

        if((end - start) > this->shape[0])
            throw std::runtime_error("Invalid Row Slice");

        std::vector<std::vector<T>> resf;
        size_t col = this->shape[this->ndims - 1];
       
        for(size_t j = start; j < end; j++){
            std::vector<T> res;
            for(size_t i = 0; i<col; i++)
            {
                res.push_back(this->data[col * j + i]);
            }

            resf.push_back(res);
        }

        size_t s = end-start;
        return Matrix<T>(resf, {s, this->shape[this->ndims - 1]});
    }

    Matrix<T> slice_cols(size_t start, size_t end) {
        if(this->shape.size() == 1)
            throw std::runtime_error("Not a 2D matrix");

        if((end - start) > this->shape[1])
            throw std::runtime_error("Invalid Col Slice");

        std::vector<T> resf;
        size_t row = this->shape[this->ndims - 2];
        size_t col = this->shape[this->ndims - 1];
       
        for(size_t j = 0; j < row; j++){
            for(size_t i = start; i<end; i++)
            {
                resf.push_back(this->data[col * j + i]);
            }

        }

        size_t s = end - start; 
        return Matrix<T>(resf, {this->shape[this->ndims - 2], s});
    }

    Matrix<T> pow(Matrix<T> input, T power)
    {
        std::vector<T> res;
        size_t numElems = 1;

        for(auto i: input.shape)
            numElems *= i;

        res.reserve(numElems);
        for(size_t k=0; k<numElems; k++)
        {
            res.push_back((T)std::pow(input.data[k]), power);
        }
        
        return Matrix<T>(res, input.shape);
    }


    Matrix<T> flatten()
    {
        return Matrix<T>(this->data);
    }

    Matrix<T> reshape(std::initializer_list<size_t> new_shape) {
        return this->reshape(Matrix<T>::getShape(new_shape));
    }

    Matrix<T> reshape(shape_t new_shape) {
        size_t n = 1;
        for (auto d : new_shape) n *= d;
        if (n != this->data.size())
            throw std::runtime_error("reshape: size mismatch");
        return Matrix<T>(this->data, new_shape);
    }

    std::vector<T> get_data(){
        return this->data;
    }

    // Matrix static functions
    //********************************************************************************* */

    static T inf() { return std::numeric_limits<T>::infinity(); }
    static T nan() { return std::numeric_limits<T>::quiet_NaN(); }

    static Matrix<T> ravel(Matrix<T> mat)
    {
        return Matrix<T>(mat.data);
    }

    Matrix<T> slice_axis(size_t start, size_t end, size_t axis) {
        if (axis >= this->ndims)
            throw std::runtime_error("slice_axis: axis out of range");

        // Build output shape
        shape_t out_shape = this->shape;
        out_shape[axis] = end - start;

        size_t total = 1;
        for (auto d : out_shape) total *= d;

        // Strides for source
        auto src_strides = this->computeShapes(this->shape);
        auto dst_strides = this->computeShapes(out_shape);

        std::vector<T> out(total);

        for (size_t flat = 0; flat < total; flat++) {
            // Decompose flat index into nd-index in output
            shape_t idx(out_shape.size());
            size_t tmp = flat;
            for (int d = (int)out_shape.size() - 1; d >= 0; d--) {
                idx[d] = tmp % out_shape[d];
                tmp   /= out_shape[d];
            }

            // Map sliced axis back to source coordinate
            shape_t src_idx = idx;
            src_idx[axis] += start;

            // Flat index in source
            size_t src_flat = 0;
            for (size_t d = 0; d < this->shape.size(); d++)
                src_flat += src_idx[d] * src_strides[d];

            out[flat] = this->data[src_flat];
        }

        return Matrix<T>(out, out_shape);
    }

    static Matrix<T> expand_dims(const Matrix<T>& m, size_t axis) {
        shape_t new_shape = m.shape;
        new_shape.insert(new_shape.begin() + axis, 1);
        Matrix<T> out;
        out.data  = m.data;  
        out.shape = new_shape;
        out.ndims = new_shape.size();
        out.size  = m.data.size();
        return out;
    }

    static bool any(const Matrix<T>& m) {
        for (auto& v : m.data)
            if (v != T(0)) return true;
        return false;
    }

    static bool hasNaN(const Matrix<T>& m) {
        for (auto& v : m.data)
            if (std::isnan(v)) return true;
        return false;
    }

    template <typename Pred>
    static bool any(const Matrix<T>& m, Pred pred) {
        for (auto& v : m.data)
            if (pred(v)) return true;
        return false;
    }

    static Matrix<T> concat(std::initializer_list<Matrix<T>> list, size_t axis)
    {
        
        if (list.size() == 0) return Matrix<T>();

        std::vector<Matrix<T>> s;

        for (const auto& item : list)
        {   
            s.push_back(item);
        }
        
        return Matrix<T>::concat(s, axis);
        
    }

    static Matrix<T> concat(const std::vector<Matrix<T>>& mats, size_t axis) {
        for (size_t i = 1; i < mats.size(); i++)
            for (size_t d = 0; d < mats[0].shape.size(); d++)
                if (d != axis && mats[i].shape[d] != mats[0].shape[d])
                    throw std::runtime_error("concat: shape mismatch on non-concat axis");

        shape_t out_shape = mats[0].shape;
        for (size_t i = 1; i < mats.size(); i++)
            out_shape[axis] += mats[i].shape[axis];

        size_t total = 1;
        for (auto d : out_shape) total *= d;
        std::vector<T> out_data(total);

        // Iterate every output index, map back to source matrix
        for (size_t flat = 0; flat < total; flat++) {
            // Convert flat index to nd-index in output
            shape_t idx(out_shape.size());
            size_t tmp = flat;
            for (int d = out_shape.size()-1; d >= 0; d--) {
                idx[d] = tmp % out_shape[d];
                tmp   /= out_shape[d];
            }

            // Find which input matrix owns this axis coordinate
            size_t axis_coord = idx[axis];
            size_t mat_i = 0;
            for (; mat_i < mats.size()-1; mat_i++) {
                if (axis_coord < mats[mat_i].shape[axis]) break;
                axis_coord -= mats[mat_i].shape[axis];
            }

            // Convert nd-index to flat index in source matrix
            idx[axis] = axis_coord;
            size_t src_flat = 0, stride = 1;
            for (int d = mats[mat_i].shape.size()-1; d >= 0; d--) {
                src_flat += idx[d] * stride;
                stride   *= mats[mat_i].shape[d];
            }

            out_data[flat] = mats[mat_i].data[src_flat];
        }

        return Matrix<T>(out_data, out_shape);
    }

    static Matrix<T> where(const Matrix<bool>& cond, T if_true, T if_false) {
        std::vector<T> res;
        res.reserve(cond.data.size());
        for (size_t i = 0; i < cond.data.size(); i++)
        {   
            if(cond.data[i]) 
                res.push_back(if_true);
            else
                res.push_back(if_false);
        }
        return Matrix<T>(res, cond.shape);
    }

    static Matrix<T> randomn(std::initializer_list<size_t> s){ return randn(getShape(s)); }
    static Matrix<T> randomn(shape_t s)                      { return randn(s); }
    static Matrix<T> eye(std::initializer_list<size_t> s){ return eye(getShape(s)[0]); }

    static Matrix<T> stack(std::vector<Matrix<T>> list, size_t axis)
    {        
        std::vector<Matrix<T>> s = list;
        std::vector<T> res;
        
        auto row_shape = s[0].shape[0];
        auto col_shape = s[0].shape[1];
        
        for (size_t i = 1; i<s.size(); i++)
        {   
            if(s[i].shape[0] != row_shape )
                throw std::runtime_error("Invalid Matrix shape for axis 0 stacking");
            else if(col_shape != s[i].shape[1])
                throw std::runtime_error("Invalid Matrix shape for axis 0 stacking");

        }

        if(axis == 0)
        {
            for(auto item: s){

                for(size_t k=0; k<item.get_size(); k++)
                {
                    res.push_back(item.data[k]);
                }

            }

            return Matrix<T>(res, {s[0].shape[0]*s.size(), s[0].shape[1]});
        }else if(axis == 1){
             
            for(size_t row = 0; row < s[0].shape[0]; row++){
                for(size_t i = 0; i < s.size(); i++){
                    for(size_t col = 0; col < s[i].shape[1]; col++){
                        res.push_back(s[i].data[row * s[i].shape[1] + col]);
                        
                    }
                }
            }    

            return Matrix<T>(res, {s[0].shape[0], s[0].shape[1]*s.size()});
        }
        else if(axis == 2)
        {
            size_t rows = s[0].shape[0];
            size_t cols = s[0].shape[1];
            size_t depth = s.size();

            for(size_t row = 0; row < rows; row++){
                for(size_t col = 0; col < cols; col++)
                    for(size_t d = 0; d < depth; d++)
                        res.push_back(s[d].data[row * cols + col]);
            }

            return Matrix<T>(res, {rows, cols, depth});
        }
        else
        {
            throw std::runtime_error("axis must be 0, 1, or 2");
        }
    }

    static Matrix<T> stack(std::initializer_list<Matrix<T>> list, size_t axis)
    {
        
        if (list.size() == 0) return Matrix<T>();

        std::vector<Matrix<T>> s;

        for (const auto& item : list)
        {   
            s.push_back(item);
        }
        
        return Matrix<T>::stack(s, axis);
        
    }

    Matrix<T> elemsAt(Matrix<T> indices) {
        size_t dim        = this->shape.back();   
        size_t vocab_size = this->shape[0];
        size_t n_tokens   = indices.data.size();

        std::vector<T> out;
        out.reserve(n_tokens * dim);

        for (size_t i = 0; i < n_tokens; i++) {
            size_t idx = static_cast<size_t>(std::round(indices.data[i]));
            if (idx >= vocab_size)
                throw std::runtime_error("Index out of bounds in embedding lookup: " + std::to_string(idx));
            out.insert(out.end(),
                this->data.begin() + idx * dim,
                this->data.begin() + idx * dim + dim);
        }

        // output shape = index shape + [embed_dim]
        shape_t out_shape = indices.shape;
        out_shape.push_back(dim);
        return Matrix<T>(out, out_shape);
    }

    static Matrix<T> arrange(T stop)
    {
        return Matrix<T>::arrange(0, stop, 1);  
    }

    static Matrix<T> arrange(T start, T stop, T step = 1) {
        std::vector<T> res;
        size_t n = (size_t)std::ceil((stop - start) / step);
        res.reserve(n);
        for (size_t i = 0; i < n; i++)
            res.push_back(start + (T)i * step);
        return Matrix<T>(res, {res.size()});
    }
    
    static Matrix<T> zeros(shape_t shape)
    {
        size_t n = 1;
        for (auto d : shape) n *= d;
        return Matrix<T>(std::vector<T>(n, (T)0), shape);
    }

    static Matrix<T> ones(shape_t shape)
    {
        size_t n = 1;
        for (auto d : shape) n *= d;
        return Matrix<T>(std::vector<T>(n, (T)1), shape);
    }

    static Matrix<T> zeros(std::initializer_list<size_t> inshape)
    {
       return Matrix<T>::zeros(Matrix<T>::getShape(inshape));
    }

    static Matrix<T> ones(std::initializer_list<size_t> inshape)
    { 
        return Matrix<T>::ones(Matrix<T>::getShape(inshape));
    }

    static Matrix<T> random(std::initializer_list<size_t> inshape)
    {
        return  Matrix<T>::random(Matrix<T>::getShape(inshape));
    }

    // Samples a single integer index from prob distribution in `probs`
    // probs should be 1D and sum to 1
    static Matrix<T> choice(size_t n, const Matrix<T>& probs) {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double r = dist(get_gen());
        double cumsum = 0.0;
        for (size_t i = 0; i < n; i++) {
            cumsum += (double)probs.data[i];
            if (r <= cumsum)
                return Matrix<T>({(T)i}, {1});
        }
        // Fallback: return last index (handles floating point rounding)
        return Matrix<T>({(T)(n-1)}, {1});
    }

    static Matrix<T> random(shape_t shape)
    {
        std::vector<T> res;
        size_t numElems = 1;

        for(auto i: shape)
            numElems *= i;

        for(size_t k=0; k<numElems; k++)
        {
            res.push_back((T)std::rand());
        }
        
        return Matrix<T>(res, shape);
    }

    static Matrix<T> sin(Matrix<T> input)
    {
        std::vector<T> res;
        size_t numElems = 1;

        for(auto i: input.shape)
            numElems *= i;

        for(size_t k=0; k<numElems; k++)
        {
            res.push_back((T)std::sin(input.data[k]));
        }
        
        return Matrix<T>(res, input.shape);
    }

    static Matrix<T> cos(Matrix<T> input)
    {
        std::vector<T> res;
        size_t numElems = 1;

        for(auto i: input.shape)
            numElems *= i;

        for(size_t k=0; k<numElems; k++)
        {
            res.push_back((T)std::cos(input.data[k]));
        }
        
        return Matrix<T>(res, input.shape);
    }

    static Matrix<T> tan(Matrix<T> input)
    {
        std::vector<T> res;
        size_t numElems = 1;

        for(auto i: input.shape)
            numElems *= i;

        for(size_t k=0; k<numElems; k++)
        {
            res.push_back((T)std::tan(input.data[k]));
        }
        
        return Matrix<T>(res, input.shape);
    }

    static Matrix<T> randu(std::initializer_list<size_t> inshape)
    {
        return  Matrix<T>::randu(Matrix<T>::getShape(inshape));
    }

    static Matrix<T> randu(T start, T stop, std::initializer_list<size_t> inshape)
    {
        return  Matrix<T>::randu(start, stop, Matrix<T>::getShape(inshape));
    }

    static Matrix<T> he(std::initializer_list<size_t> inshape)
    {
        return Matrix<T>::he(Matrix<T>::getShape(inshape));
    }
  
    static Matrix<T> randu(shape_t shape)
    {
        size_t numElems = 1;
        for(auto i : shape) numElems *= i;

        std::uniform_real_distribution<T> dist(0.0, 1.0);
        std::vector<T> res;
        res.reserve(numElems);
        for(size_t k = 0; k < numElems; k++)
            res.push_back(dist(get_gen()));

        return Matrix<T>(res, shape);
    }

    static Matrix<T> randu(T start, T stop, shape_t shape)
    {
        size_t numElems = 1;
        for(auto i : shape) numElems *= i;

        std::vector<T> res;
        res.reserve(numElems);

        if constexpr (std::is_integral_v<T>) {
            std::uniform_int_distribution<T> dist(start, stop - 1);
            for(size_t k = 0; k < numElems; k++)
                res.push_back(dist(get_gen()));
        } else {
            std::uniform_real_distribution<T> dist(start, stop);
            for(size_t k = 0; k < numElems; k++)
                res.push_back(dist(get_gen()));
        }

        return Matrix<T>(res, shape);
    }

    static void manual_seed(unsigned int seed) {
        get_gen(seed);
    }

    static Matrix<T> randn(shape_t shape)
    {
        size_t numElems = 1;
        for(auto i : shape) numElems *= i;

        std::normal_distribution<T> dist(0.0, 1.0);
        std::vector<T> res;
        res.reserve(numElems);
        for(size_t k = 0; k < numElems; k++)
            res.push_back(dist(get_gen()));

        return Matrix<T>(res, shape);
    }

    // He/Kaiming 
    static Matrix<T> he(shape_t shape)
    {
        size_t fan_in = shape[0];
        T std_dev = std::sqrt((T)2.0 / (T)fan_in);

        size_t numElems = 1;
        for(auto i : shape) numElems *= i;

        std::normal_distribution<T> dist(0.0, std_dev);
        std::vector<T> res;
        res.reserve(numElems);
        for(size_t k = 0; k < numElems; k++)
            res.push_back(dist(get_gen()));

        return Matrix<T>(res, shape);
    }

    static Matrix<T> log(Matrix<T> mat)
    {
        std::vector<T> arr;
        for(size_t i=0; i< mat.data.size(); i++)
        { 
            T prod = (T)std::log(mat.data.at(i));
            arr.push_back(prod);
        }
        return Matrix<T>(arr, mat.shape);
    }

    static Matrix<T> eye(size_t inshape){
        size_t numElems = 1;

        std::vector<T> res;
        for(int i=0; i<(inshape*inshape); i++)
            res.push_back(0);

        for(auto i=0; i<inshape; i++)
        {
            for(auto j=0; j<inshape; j++)
                res.push_back(res[i * inshape + j] = (i == j) ? 1 : 0);
        }

        return Matrix<T>(res, {inshape, inshape});
    }


    static Matrix<T> tril(size_t n) {
        std::vector<T> res(n * n, 0);
        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j <= i; j++)  // <= to include diagonal
                res[i * n + j] = 1;
        return Matrix<T>(res, {n, n});
    }

    static Matrix<T> tril(Matrix<T> input) {
        std::vector<T> res = input.data;
        size_t n = input.shape[0];
        for (size_t i = 0; i < n; i++)
            for (size_t j = i + 1; j < n; j++)
                res[i * n + j] = 0;
        return Matrix<T>(res, input.shape);
    }
    
    static Matrix<T> triup(size_t n) {
        std::vector<T> res(n * n, 0);
        for (size_t i = 0; i < n; i++)
            for (size_t j = i; j < n; j++)  // include diagonal
                res[i * n + j] = 1;
        return Matrix<T>(res, {n, n});
    }

    static Matrix<T> triup(Matrix<T> input) {
        std::vector<T> res = input.data;
        size_t n = input.shape[0];  
        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j < i; j++)  
                res[i * n + j] = 0;
        return Matrix<T>(res, input.shape);
    }

    static Matrix<T> one_hot(Matrix<T> labels, size_t num_classes)
    {
        size_t n = labels.get_size();
        std::vector<T> res(n * num_classes, (T)0);
        for (size_t i = 0; i < n; i++) {
            size_t cls = (size_t)labels.data[i];
            res[i * num_classes + cls] = (T)1;
        }
        return Matrix<T>(res, {n, num_classes});
    }

    void ones()
    {
        auto shape = this->shape;
        size_t numElems = 1;

        for(auto i: shape)
            numElems *= i;

        this->data.assign(numElems, 1);
    }

    void zeros()
    {
        auto shape = this->shape;
        size_t numElems = 1;

        for(auto i: shape)
            numElems *= i;

        this->data.assign(numElems, 0);

    }

    void copy_from(Matrix<T>* two)
    {
        if (two == nullptr)
            throw std::runtime_error("copy_from: null pointer input\n");
        
        this->data = two->data;
        this->shape = two->shape; 
        this->numElementsSeen = two->numElementsSeen;
        this->ndims = two->shape.size();
        this->size = two->data.size();
    }

    void copy_from(Matrix<T>& two)
    {      
        this->data = two.data;
        this->shape = two.shape; 
        this->numElementsSeen = two.numElementsSeen;
        this->ndims = two.shape.size();
        this->size = two.data.size();
    }

    void copy_from(const Matrix<T>& two)
    {
        this->data = two.data;
        this->shape = two.shape; 
        this->numElementsSeen = two.numElementsSeen;
        this->ndims = two.shape.size();
        this->size = two.data.size();
    }
 
    Matrix maximum(const T a){
        std::vector<T> res;
        for(auto i : this->data)
        {
            if(i < a)
                res.push_back(0);
            else
                res.push_back(i);
        }
        return Matrix<T>(res, this->shape);
    }

    void clear(){        
        this->data.clear();
        this->shape.clear();
        this->size  = 0;
        this->ndims = 0;
        this->numElementsSeen.clear();
    }
//°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°

    Matrix<T> transpose()
    {
        if (shape.size() == 1) return this->transpose_1D();
        if (shape.size() == 2) {
            shape_t resShape = {this->shape[1], this->shape[0]};
            return Matrix<T>(this->transpose_2D(), resShape);
        }
        size_t ndims = this->shape.size();
        shape_t perm(ndims), resShape(ndims);
        for (size_t i = 0; i < ndims; i++) {
            perm[i]     = ndims - 1 - i;
            resShape[i] = this->shape[perm[i]];
        }
        std::vector<T> res;
        transpose(perm, resShape, res);
        return Matrix<T>(res, resShape);
    }

    Matrix<T> transpose(shape_t perm)
    {
        if (shape.size() == 1) return this->transpose_1D();
        
        if (perm.size() != this->shape.size())
            throw std::runtime_error("transpose: perm size must match number of dimensions\n");

            
        size_t ndims = this->shape.size();
        if (ndims == 2) {
            shape_t resShape = {this->shape[1], this->shape[0]};
            return Matrix<T>(this->transpose_2D(), resShape);
        }
        shape_t resShape(ndims);
        for (size_t i = 0; i < ndims; i++)
            resShape[i] = this->shape[perm[i]];
        std::vector<T> res;
        transpose(perm, resShape, res);
        return Matrix<T>(res, resShape);
    }

    Matrix<T> transpose(std::initializer_list<size_t> inperm)
    {
        return this->transpose(Matrix<T>::getShape(inperm));
    }

    Matrix<T> sum(size_t axis)
    {
        shape_t resShape; 

        for(size_t i=0; i<this->shape.size(); i++)
        { 
            if (i == axis)
                continue;
            resShape.push_back(this->shape.at(i));
        }

        std::vector<T> res;
        shape_t indexStack{}; 
        size_t lhsStart = 0;
        size_t dim = 0;

        this->sum(res, indexStack, lhsStart, axis, dim);
        return Matrix<T>(res, resShape); 
    }

    T sum()
    {
        return this->sum_1D(); 
    }

    Matrix<T> matmul(const Matrix<T> &rhs)
    {
        if (areShapes1D(this->shape, rhs.shape) == true)
            throw std::runtime_error("matmul: cannot multiply two 1D tensors, use dot() instead\n");

        shape_t resShape{}; 

        for(size_t i=0; i<this->shape.size()-1; i++)
        { 
            resShape.push_back(this->shape.at(i));
        }

        resShape.push_back(rhs.shape.back());

        size_t total = 1;
        for (auto s : resShape) total *= s;

        shape_t indexStack{}; 
        std::vector<T> out(avx2_pad(total), T(0));  // padded for AVX2 alignment during compute

        shape_t resElements = this->computeShapes(resShape); 
        size_t dim=0;

        this->matmul(rhs, indexStack, resElements, dim, out);

        return Matrix<T>(out, resShape);
    }
    

    Matrix<T> dot(const Matrix<T> &rhs)
    {
        if(areShapes1D(this->shape, rhs.shape))
        {
            Matrix<T> res({this->dotProduct1D(this->data, rhs.data)});
            return res;
        }
        
        if(areShapes2D(this->shape, rhs.shape))
        {
            return this->dotProduct2D(rhs);
        }

        if (dotShapesAssert(rhs.shape)== false)
            throw std::runtime_error("dot: invalid shapes for dot product\n");
        
        shape_t resShape;
        size_t size = 1;

        for(size_t i=0; i<this->shape.size()-1; i++)
        { 
            size *= this->shape.at(i);
            resShape.push_back(this->shape.at(i));
        }

        size *= rhs.shape.back();
        resShape.push_back(rhs.shape.back()); // column dimension of the right hand side matrix
        
        shape_t indexStack{}; 
        auto resElements = this->computeShapes(resShape);
        size_t total = 1;
        for (auto s : resShape) total *= s;
        std::vector<T> out(avx2_pad(total), T(0));  // padded for AVX2 alignment during compute
        
        this->matmul(rhs, indexStack, resElements, 0, out);

        out.resize(total);  // trim padding before wrapping in Matrix
        return Matrix<T>(out, resShape);
        
    } 


    std::ostream& print(std::ostream &out, shape_t &indexStack, size_t dim)
        {
            if(indexStack.size()  == this->shape.size()-1)
            {
                // We are in the state where rhs and lhs matrices are both on 2d matrix format
                //find the position in the lhs array where we are at
                size_t lhsStart{0};

                for(size_t i{0}; i<indexStack.size(); i++)
                {
                    lhsStart += indexStack.at(i);
                }
                out<<" [";
                for(size_t i{0}; i<this->shape.at(dim); i++)
                {
                    out<<this->data.at(lhsStart+i)<<",";
                }
                out<<"]\n";
                return out;
            }

            out <<"[\n";
            // Push the extra dimensions to the index stack and recursively traverse the indices, then pop once one the operation for that index has been done
            for(size_t i=0; i<this->shape[dim]; i++)
            {
                indexStack.push_back(this->numElementsSeen[dim] * i);//calculate how many elements have been processed to get the pointer to the right location in data and push to the stack
                print(out, indexStack, dim+1);
                indexStack.pop_back(); //pops out of stack
            }
            out <<"]";
            return out;
        }


    // template <typename E>
    // friend std::ostream & operator <<(std::ostream &out, Matrix<E> &m);

    template <typename E>
    friend std::ostream & operator <<(std::ostream &out, Matrix<E> m);

    friend class Broadcast<T>;
};

    template <typename E>
    std::ostream& operator << (std::ostream &out, Matrix<E> m)
    {
        //out<<m.data<<"\t";
        //out<<"Shape:"<<m.shape;
        size_t dim = 0;
        shape_t stack;
        m.print(out, stack, dim);
        return out;
    }

    // template <typename E>
    // std::ostream& operator << (std::ostream &out, Matrix<E> &m)
    // {
    //     //out<<m.data<<"\t";
    //     //out<<"Shape:"<<m.shape;
    //     size_t dim = 0;
    //     shape_t stack;
    //     m.print(out, stack, dim);
    //     return out;
    // }

    // Matrix Arithmetic Operations 
   
    template <typename T>
    Matrix<T> operator * (const T a, Matrix<T> rhs)
    {
        return Matrix<T>(rhs.data * a, rhs.shape);
    }

    template <typename T>
    Matrix<T> operator * (Matrix<T> lhs, const T a)
    {
        return Matrix<T>( a * lhs.data, lhs.shape);
    }
    
    template <typename T>
    Matrix<T> operator / (Matrix<T> lhs, const T a)
    {
        return Matrix<T>(lhs.data/a, lhs.shape);
    }

    template <typename T>
    Matrix<T> operator / (const T a, Matrix<T> lhs)
    {
        return Matrix<T>( a / lhs.data, lhs.shape);
    }
    
    template <typename T>
    Matrix<T> pow(Matrix<T> lhs, const T a)
    {
        return Matrix<T>( pow(lhs.data, a), lhs.shape);
    }
    
    template <typename T>
    Matrix<T> pow(Matrix<T> a, Matrix<T> b)
    {
        return Matrix<T>(pow(a.data, b.data), a.shape);
    }
  
    template <typename T>
    Matrix<T> operator + (Matrix<T> lhs, const T a)
    {
        return Matrix<T>(lhs.data + a, lhs.shape);
    }

    template <typename T>
    Matrix<T> operator + ( const T a, Matrix<T> lhs)
    {
        return Matrix<T>( a + lhs.data, lhs.shape);
    }
  
     template <typename T>
    Matrix<T> operator - (Matrix<T> lhs, const T a)
    {
        return Matrix<T>(lhs.data - a, lhs.shape);
    }

    template <typename T>
    Matrix<T> operator - (const T a,  Matrix<T> lhs)
    {
        return Matrix<T>( a - lhs.data, lhs.shape);
    }
  
    //................................................................................

    template <typename T>
    Matrix<T> operator < (const T a, const Matrix<T> &rhs)
    {
        return Matrix<T>(a < rhs.data, rhs.shape);
    }

    template <typename T>
    Matrix<T> operator < (const Matrix<T> &rhs, const T a)
    {
        return Matrix<T>(rhs.data < a , rhs.shape);
    }

    template <typename T>
    Matrix<T> operator > (const Matrix<T> &lhs, const T a)
    {
        return Matrix<T>( lhs.data > a, lhs.shape);
    }

    template <typename T>
    Matrix<T> operator > (const T a, const Matrix<T> &lhs)
    {
        return Matrix<T>( a > lhs.data, lhs.shape);
    }

    template <typename T>
    Matrix<T> operator <= (const Matrix<T> &rhs, const T a)
    {
        return Matrix<T>(rhs.data <= a , rhs.shape);
    }

    template <typename T>
    Matrix<T> operator <= (const T a, const Matrix<T> &rhs)
    {
        return Matrix<T>(rhs.data <= a , rhs.shape);
    }

    template <typename T>
    Matrix<T> operator >= (const T a, const Matrix<T> &lhs)
    {
        return Matrix<T>( a >= lhs.data, lhs.shape);
    }

    template <typename T>
    Matrix<T> operator >= (const Matrix<T> &lhs, const T a)
    {
        return Matrix<T>( a >= lhs.data, lhs.shape);
    }

    //--------------------------------------------------------------------------------

    template <typename T>
    Matrix<T>& operator +=(Matrix<T>& lhs, Matrix<T> rhs) {
        if (lhs.shape == rhs.shape) {
            for (size_t i = 0; i < lhs.data.size(); i++)
                lhs.data[i] += rhs.data[i];
            return lhs;
        }

        Broadcast<T> bc;
        Matrix<T> rhs_bc = bc.broadcastTo(rhs, lhs.shape);
        for (size_t i = 0; i < lhs.data.size(); i++)
            lhs.data[i] += rhs_bc.data[i];
        return lhs;
    }


    template <typename T>
    Matrix<T>& operator -=(Matrix<T>& lhs, Matrix<T> rhs) {
        if (lhs.shape == rhs.shape) {
            for (size_t i = 0; i < lhs.data.size(); i++)
                lhs.data[i] -= rhs.data[i];
            return lhs;
        }

        Broadcast<T> bc;
        Matrix<T> rhs_bc = bc.broadcastTo(rhs, lhs.shape);
        for (size_t i = 0; i < lhs.data.size(); i++)
            lhs.data[i] -= rhs_bc.data[i];
        return lhs;
    }

    template <typename T>
    Matrix<T>& operator *=(Matrix<T>& lhs, Matrix<T> rhs) {
        if (lhs.shape == rhs.shape) {
            for (size_t i = 0; i < lhs.data.size(); i++)
                lhs.data[i] *= rhs.data[i];
            return lhs;
        }

        Broadcast<T> bc;
        Matrix<T> rhs_bc = bc.broadcastTo(rhs, lhs.shape);
        for (size_t i = 0; i < lhs.data.size(); i++)
            lhs.data[i] *= rhs_bc.data[i];
        return lhs;
    }

    template <typename T>
    Matrix<T>& operator /=(Matrix<T>& lhs, Matrix<T> rhs) {
        if (lhs.shape == rhs.shape) {
            for (size_t i = 0; i < lhs.data.size(); i++)
                lhs.data[i] /= rhs.data[i];
            return lhs;
        }

        Broadcast<T> bc;
        Matrix<T> rhs_bc = bc.broadcastTo(rhs, lhs.shape);
        for (size_t i = 0; i < lhs.data.size(); i++)
            lhs.data[i] /= rhs_bc.data[i];
        return lhs;
    }

    template <typename T>
    Matrix<T> operator +=(Matrix<T> &lhs,  const T cte)
    {
        return lhs.data += cte;
    }

    template <typename T>
    Matrix<T> operator -=(Matrix<T> &lhs,  const T cte)
    {
        return lhs.data -= cte;
    }

    template <typename T>
    Matrix<T> operator *=(Matrix<T> &lhs,  const T cte)
    {
        return lhs.data *= cte;
    }

    template <typename T>
    Matrix<T> operator /=(Matrix<T> &lhs,  const T cte)
    {
        return lhs.data /= cte;
    }

//------------------------------------------------------------------------------------

    // void clip_and_noise(Matrix<float>& delta, float clip_norm = 1.0f, float noise_std = 0.01f) {
    //     float norm = delta.frobenius_norm();
    //     if (norm > clip_norm) delta = delta * (clip_norm / norm);
    //     delta = delta + Matrix<float>::gaussian_noise(delta.shape, 0.0f, noise_std);
    // }

    template <typename T>
    Matrix<T> sumGradForBroadcast(Matrix<T> grad, std::vector<size_t> originalShape) {
        Matrix<T> res = grad;
        
        // Keep summing leading dimensions until rank matches
        while (res.shape.size() > originalShape.size()) {
            res = res.sum(0);
        }
        
        // Sum any dimension where original was size 1
        for (int i = (int)res.shape.size() - 1; i >= 0; i--) {
            if (originalShape[i] == 1 && res.shape[i] > 1) {
                res = res.sum(i);
                // sum(i) on a kept dim should leave shape[i]=1; if it collapses, reshape
                if (res.shape.size() < originalShape.size()) {
                    shape_t s = res.shape;
                    s.insert(s.begin() + i, 1);
                    res = Matrix<T>(res.data, s);
                }
            }
        }
        
        // Final shape correction
        if (res.shape != originalShape)
            res = Matrix<T>(res.data, originalShape);
        
        return res;
    }
    
#endif