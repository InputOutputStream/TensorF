#ifndef __MATRIX_CLASS_INCLUDED__
#define __MATRIX_CLASS_INCLUDED__

#include "../Types/types.hpp"
#include "../Overloads/Overload.hpp"

#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <memory>
#include <algorithm>
#include <random>

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
            assert(assertBroadcast(t1, t2) && "Invalid broadcast operation\n");

            long i, j;
            shape_t s1 = t1.shape;
            shape_t s2 = t2.shape;
            shape_t resShape;

            for(i = s1.size()-1, j = s2.size()-1; i >= 0 && j >= 0; i--, j--)
            {
                resShape.push_back(std::max(s1[i], s2[j]));
            }     

            if(s1.size() < s2.size())
            {
                long n  =  s2.size() - s1.size();
                for(long k = (n-1); k >= 0; k--){
                    resShape.push_back(std::max((long)1, s2[k]));
                }
            }
            else if(s1.size() > s2.size())
            {
                long n  =  s1.size() - s2.size();
                for(long k = (n-1); k >= 0; k--){
                    resShape.push_back(std::max((long)1, s1[k]));
                }        

            }

            std::reverse(resShape.begin(), resShape.end());
            return resShape;
        }

        shape_t computeShapes(const shape_t shape)
        {
            shape_t numElementsSeen(shape.size());
            size_t p{1};
            for(long i = shape.size()-1; i>=0 ;i--)
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

                for(long k = new_shape.size()-1; k >= 0; k--)
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
        bool s = 1;
        long p = 1;
        for(size_t i = 0; i<shape.size(); i++)
            {
                p *=shape[i];
            }
        if((long)data.size() != p)
            s = false;   

        return s;
    }               

     //There is an error in the computes shapes method as we go from 1D to 2D 
        //Solution
        //I was using a class attr this->numElementsSeen instead of a local variable numElementsSeen which xas wronf since i am returning it
    shape_t computeShapes(const shape_t shape)
    {
        shape_t numElementsSeen(shape.size());
        size_t p{1};
        for(long i = shape.size()-1; i>=0 ;i--)
        {
            numElementsSeen[i] = p;
            p *= shape.at(i);
        }

        return numElementsSeen;
    }

    bool areShapesEqual(const shape_t &shape)
        {
            if(shape.size() != this->shape.size())
            {
                return false;
            }

            for(size_t i = 0; i<this->shape.size(); i++)
                {
                    if(shape[i] != this->shape[i])
                    {
                        return false;
                    }
                }

            return true;
        }
    
    shape_t getShape(const std::initializer_list<long> shape)
    {
        if (shape.size() == 0) return shape_t{0};

        shape_t s;

        for (const auto& item : shape)
        {   
            s.push_back(item);
        }
        return s;
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

    template <typename U> // cloudy
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

            
    bool isRegular2D(const std::vector<std::vector<T>> data)
    {
        if(data.size() == 0)
            return true;

        std::vector<T> j = data[0];
        for(int i=1; i<data.size(); i++)
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
        assert(!(axis > 1) && "Invalid input axis recieved for sum 2D ");
        if(axis == 1)
        {
            T s = 0;
            size_t index = 0;
            for(long i = 0; i<this->shape[this->ndims - 2]; i++)
            {
                for(long j = 0; j<this->shape[this->ndims - 1]; j++)
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
            for(long i = 0; i<this->shape[this->ndims - 1]; i++)
            {
                for(long j = 0; j<this->shape[this->ndims - 2]; j++)
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
        assert(!(axis > 1) && "Invalid input axis recieved for sum 2D ");
        if(axis == 1)
        {
            T s = 0;
            size_t index = 0;
            for(long i = 0; i<this->shape[this->ndims - 2]; i++)
            {
                for(long j = 0; j<this->shape[this->ndims - 1]; j++)
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
        
            for(auto i = 0; i<nslice; i++){

                for(auto j = 0; j<this->shape[0]; j++)
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

        // assert(this->isShape1D() && "Invalid input shape");
        assert(dim <= this->shape.size() && "Invalid dim\n");

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
            for(unsigned long i{0}; i<indexStack.size(); i++)
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
        
        for(long i=0; i<this->shape[dim]; i++)
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
            assert(1 && "Invalid input shape, returnong unchanged object \n");
            return *this;
        }
    }

    std::vector<T> transpose_2D()
        {            
            long row = this->shape[this->ndims - 2];
            long col = this->shape[this->ndims - 1];

            std::vector<T> res = this->data;

            for(long i=0; i<row; i++)
            {
                for(long j=0; j<col; j++)
                {
                    res[j*row + i] = this->data[i*col + j];
                }
            }

        return res;
    }

    void transpose(const shape_t resShape, std::vector<T> &res)
    {
        Matrix<T> temp(this);
        auto ns = temp.numElementsSeen;
        auto nr = temp.computeShapes(resShape);

        long dsize = 1; // num elements
        for(long i: resShape)
        {
            dsize *= i;
        }

        for(long i = 0; i<dsize; i++)
        {
            shape_t new_index;
            auto k = i;
            for(auto j: nr)
            {
                new_index.push_back((size_t)(k / j));
                k = k%j;
            }

            shape_t rev;
            rev.insert(rev.end(), new_index.rbegin(),  new_index.rend());
            size_t npos = 0;
            for(size_t id = 0; id <temp.shape.size(); id++)
            {
                npos += ns[id] * rev[id];
            }

            res.push_back(temp.data[npos]);
        }
    }

//°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°

    T dotProduct1D(const std::vector<T> &lhs, const std::vector <T> &rhs){
            T sum = 0;
            for(size_t i = 0, j = 0; i<lhs.size() && j<rhs.size(); i++, j++){
                sum += lhs.at(i) * rhs.at(j);
            }
            return sum;
        }
    
    Matrix<T> dotProduct2D(const Matrix<T> &lhs, 
                        const Matrix<T> &rhs, 
                        long lhsStart, 
                        long resStart)
        {
            Matrix<T> res = zeros({lhs.shape[lhs.ndims - 2], rhs.shape[rhs.ndims - 1]});
            
            long row1 = lhs.shape[lhs.shape.size()-2];
            long col1 = lhs.shape[lhs.shape.size()-1];
            //long row2 = rhs.shape[rhs.shape.size()-2];
            long col2 = rhs.shape[rhs.shape.size()-1];

            T sum = 0;

            for(long i=0; i<row1; i++)
            {
               for(long k =0; k<col2; k++)
                {
                    sum = 0;
                    for(long j=0; j<col1; j++)
                    {
                        sum += lhs.data[lhsStart + i * col1 + j] * rhs.data[j*col2 + k];
                    }

                    res.data[(resStart + i*col2 + k)] = sum;

                }
            }

            return res;
        }

    void matProduct2D(const Matrix<T> &rhs, 
                        std::vector<T> &res, 
                        long lhsStart, 
                        long resStart)
        {
            
            long row1 = this->shape[this->shape.size()-2];
            long col1 = this->shape[this->shape.size()-1];
            long row2 = rhs.shape[rhs.shape.size()-2];
            long col2 = rhs.shape[rhs.shape.size()-1];

            assert(row2 == col1 && "FATAL ERRROR, MATRIX PRODUCT ATTEMPTED ON INdataID MATRICES");
            T sum = 0;

            for(long i=0; i<row1; i++)
            {
                for(long k =0; k<col2; k++)
                {
                    sum = 0;
                    for(long j=0; j<col1; j++)
                    {
                        sum += this->data[lhsStart + i * col1 + j] * rhs.data[j*col2 + k];
                    }

                    res[(resStart + i*col2 + k)] = sum;

                }
            }
        }


    void matmul(const Matrix<T> &rhs,
                std::vector<T> &res, 
                shape_t &indexStack, 
                shape_t &resElements, 
                long dim)
        {

        if(indexStack.size()  == (this->shape.size()-2))
        {
            // We are in the state where rhs and lhs matrices are both on 2d matrix format
            // find the position in the lhs array where we are at
            
            long lhsStart{0};
            long resStart {0};
            //rhsStart and lhsStart denotes respectively the starting points of the multiplication within the matrix dimensions

            for(unsigned long i{0}; i<indexStack.size(); i++)
            {
                lhsStart += indexStack.at(i) * this->numElementsSeen.at(i);
                resStart += indexStack.at(i) * resElements.at(i);
            }

            matProduct2D(rhs, res, lhsStart, resStart);
            return;
        }

        // Push the extra dimensions to the index stack and recursively traverse the indices, the pop one once the operation for that index has been done
        for(long i=0; i<this->shape[dim]; i++)
        {
            indexStack.push_back(i);
            this->matmul(rhs, res, indexStack, resElements, dim+1);
            indexStack.pop_back(); 
        }
    }

//°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°

    protected: 
        size_t size;
        size_t ndims;
        bool gpu_nv = false;
        bool gpu_it = false;

    public:
    std::vector<T> data;
    shape_t shape;
    bool gpu = false;

    // Constructors 

    Matrix(){
        this->data.clear();
        this->shape.clear();
        this->ndims = 0; 
        this->size  = 0;

    };

    Matrix(const T& indata)
    {
        this->extractShape(indata, this->shape);
        if(this->shape.size() == 0)
            this->shape.push_back(1);
        this->flattenReccursive(indata, this->data);
        this->numElementsSeen = computeShapes(this->shape);
        this->ndims = this->shape.size();
        this->size  = this->data.size();
    }

    // template <typename U>
    // Matrix(const U& indata)
    // {
    //     if constexpr(std::is_same_v<U, Matrix>){ 
    //         this->data = indata.data;
    //         this->shape = indata.shape; 
    //         this->numElementsSeen = indata.numElementsSeen;
    //         this->ndims = indata.shape.size();
    //         this->size  = indata.data.size();
    //         return;
    //     } 

    //     this->extractShape(indata, this->shape);
    //     if(this->shape.size() == 0)
    //         this->shape.push_back(1);
    //     this->flattenReccursive(indata, this->data);
    //     this->numElementsSeen = computeShapes(this->shape);
    //     this->ndims = this->shape.size();
    //     this->size  = this->data.size();
    // }

    // Matrix(std::initializer_list<T> indata)
    // {
    //     this->shape.push_back(indata.size());
    //     this->flattenReccursive(indata, this->data);
    //     this->numElementsSeen = this->computeShapes(this->shape);
    //     this->ndims = this->shape.size();
    //     this->size  = this->data.size();
    // }
    

    Matrix(const Matrix<T>* two)
    {
        assert((two != nullptr) && "Null matrix input\n");
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
        this->data = indata;
        this->shape.push_back(this->data.size()); 
        this->numElementsSeen = this->computeShapes(this->shape);
        this->ndims = this->shape.size();
        this->size  = this->data.size();
    }

    Matrix(std::vector<T> indata, shape_t inshape)
    {
        std::cout << " indata: "<< indata << "inshape: " << inshape << "\n";
        assert(this->verifyShape(indata, inshape) && "Shape and number of elements of matrix do not match");
        this->data = indata;
        this->shape = inshape; 
        this->numElementsSeen = this->computeShapes(this->shape);
        this->ndims = this->shape.size();
        this->size  = this->data.size();
    }

    Matrix(std::vector<std::vector<T>> indata)
    {
        this->shape.push_back(indata.size());
        this->shape.push_back(indata.begin()->size());
        assert("Matrix shape must be uniform" && this->isRegular2D(indata));
        this->flattenReccursive(indata, this->data);
        this->numElementsSeen = this->computeShapes(this->shape);
        this->ndims = this->shape.size();
        this->size  = this->data.size();
    }

    Matrix(std::vector<T> indata, std::initializer_list<long> inshape)
    {
        this->data = indata;
        this->shape = this->getShape(inshape); 
        assert("Shape and number of elements of matrix do not match" && this->verifyShape(this->data, this->shape));
        this->numElementsSeen = this->computeShapes(this->shape);
        this->ndims = this->shape.size();
        this->size  = this->data.size();
    }

    Matrix(std::initializer_list<std::initializer_list<T>> indata)
    {
        this->shape.push_back(indata.size());
        this->shape.push_back(indata.begin()->size());
        assert("Matrix shape must be uniform" && this->isRegular2D(indata));
        this->flattenReccursive(indata, this->data);
        this->numElementsSeen = this->computeShapes(this->shape);
        this->ndims = this->shape.size();
        this->size  = this->data.size();
    }

    Matrix(std::initializer_list<T> indata, std::initializer_list<long> inshape)
    {
        this->flattenReccursive(indata, this->data);
        this->shape = this->getShape(inshape); 
        assert("Shape and number of elements of matrix do not match" && this->verifyShape(this->data, this->shape));
        this->numElementsSeen = this->computeShapes(this->shape);
        this->ndims = this->shape.size();
        this->size  = this->data.size();
    }

    Matrix(std::initializer_list<std::initializer_list<std::initializer_list<T>>> indata)
    {
        this->shape.push_back(indata.size());
        this->shape.push_back(indata.begin()->size());
        this->shape.push_back(indata.begin()->begin()->size());
        assert("Matrix shape must be uniform" && this->isRegular3D(indata));
        this->flattenReccursive(indata, this->data);
        this->numElementsSeen = this->computeShapes(this->shape);
        this->ndims = this->shape.size();
        this->size  = this->data.size();
    }

//°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°

    // Matrix Arithmetic Operations 
    

    Matrix<T> operator + (const Matrix<T> &rhs)
    {
        if(this->areShapesEqual(rhs.shape))
            return Matrix<T>(data + rhs.data, shape);
        else
        {
            auto res = b.broadcast(*this, rhs);
            return Matrix<T>(res.first.data + res.second.data, res.first.shape);

        }
    }

    Matrix<T> operator -()
    {
        Matrix<T> rhs(-1);
        return Matrix<T>(data * rhs.data, shape);
    }
    
    Matrix<T> operator - (const Matrix<T> &rhs)
    {
        if(this->areShapesEqual(rhs.shape))
            return Matrix<T>(data - rhs.data, shape);
        else
        {
            auto res = b.broadcast(*this, rhs);
            return Matrix<T>(res.first.data - res.second.data, res.first.shape);

        }
    }

    Matrix<T> operator * (const Matrix<T> &rhs)
    {
        if(this->areShapesEqual(rhs.shape))
            return Matrix<T>(data * rhs.data, shape);
        else
        {
            auto res = b.broadcast(*this, rhs);
            return Matrix<T>(res.first.data * res.second.data, res.first.shape);

        }
    }  
    
    Matrix<T> operator / (const Matrix<T> &rhs)
    {
        if(this->areShapesEqual(rhs.shape))
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

    Matrix<T> operator ^(const T rhs)
    {
        return Matrix<T>(data ^ rhs, shape);
    }

    Matrix<T> operator ^(const  Matrix<T> rhs)
    {
        return Matrix<T>(data ^ rhs.data, shape);
    }

    Matrix<T> exponent() 
    {
        std::vector<T> arr;
        for(size_t i=0; i< this->data.size(); i++)
        { 
            T prod = (T)exp(this->data.at(i));
            arr.push_back(prod);
        }
        return Matrix<T>(arr, this->shape);
    } 
//°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°

    size_t get_size()
    {
        return this->size;
    }

    size_t get_ndims()
    {
        return this->ndims;
    }

    Matrix<T> flatten()
    {
        return Matrix<T>(this->data);
    }

    Matrix<T> reshape(shape_t shape)
    {
        return Matrix<T>(this->data, shape);
    }

    // Matrix mathematic functions

    Matrix<T> zeros(shape_t shape)
    {
        std::vector<T> res;
        long numElems = 1;

        for(auto i: shape)
            numElems *= i;

        for(long k=0; k<numElems; k++)
        {
            res.push_back(0);
        }
        
        return Matrix<T>(res, shape);
    }

    Matrix<T> ones(shape_t shape)
    {
        std::vector<T> res;
        long numElems = 1;

        for(auto i: shape)
            numElems *= i;

        for(long k=0; k<numElems; k++)
        {
            res.push_back(1);
        }
        
        return Matrix<T>(res, shape);
    }

    Matrix<T> random(std::initializer_list<long> inshape)
    {       
        return random(getShape(inshape));
    }

    Matrix<T> random(shape_t shape)
    {
        std::vector<T> res;
        long numElems = 1;

        for(auto i: shape)
            numElems *= i;

        for(long k=0; k<numElems; k++)
        {
            res.push_back((T)std::rand() / (T)RAND_MAX);
        }
        
        return Matrix<T>(res, shape);
    }

    void ones()
    {
        auto shape = this->shape;
        long numElems = 1;

        for(auto i: shape)
            numElems *= i;

        this->data.assign(numElems, 1);
    }

    void zeros()
    {
        auto shape = this->shape;
        long numElems = 1;

        for(auto i: shape)
            numElems *= i;

        this->data.assign(numElems, 0);

    }

    void copy_from(Matrix<T>* two)
    {
        assert((two != nullptr) && "Null matrix input\n");
        this->data = two->data;
        this->shape = two->shape; 
        this->numElementsSeen = two->numElementsSeen;
        this->ndims = two->shape.size();
    }

    void copy_from(Matrix<T>& two)
    {
        this->data = two.data;
        this->shape = two.shape; 
        this->numElementsSeen = two.numElementsSeen;
        this->ndims = two.shape.size();
    }

    void copy_from(const Matrix<T>& two)
    {
        this->data = two.data;
        this->shape = two.shape; 
        this->numElementsSeen = two.numElementsSeen;
        this->ndims = two.shape.size();
    }
 
    Matrix maximum(const T a){
        std::vector<T> res;
        for(auto i : this->data)
        {
            if(i < 0)
                res.push_back(0);
            else
                res.push_back(i);
        }
        return Matrix<T>(res, this->shape);
    }

    Matrix<T> clear(){
        this->data.clear();
        return Matrix<T>(this->data, {0});
    }
//°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°

    Matrix<T> transpose(shape_t resShape)
    {
        if(resShape.size() == 1)
            return this->transpose_1D();

        if(resShape.size() == 2)
            return Matrix<T>(this->transpose_2D(), resShape);

        std::vector<T> res;
        assert(this->shape.size() == resShape.size() && "Invalid result shape");
        transpose(resShape, res);
        return Matrix<T>(res, resShape);
    }

    Matrix<T> transpose(std::initializer_list<long> inshape)
    {
        if(inshape.size() == 1)
            return this->transpose_1D();

        shape_t resShape = this->getShape(inshape);
        if(inshape.size() == 2)
            return Matrix<T>(this->transpose_2D(), resShape);

        std::vector<T> res;
        assert(this->shape.size() == resShape.size() && "Invalid result shape");
        transpose(resShape, res);
        return Matrix<T>(res, resShape);
    }

    Matrix<T> transpose()
    {
        if(shape.size() == 1)
            return this->transpose_1D();

        shape_t resShape;
        resShape.insert(resShape.end(), this->shape.rbegin(),  this->shape.rend());
        
        if(shape.size() == 2)
            return Matrix<T>(this->transpose_2D(), resShape);

        std::vector<T> res;
        assert(this->shape.size() == resShape.size() && "Invalid result shape");
        transpose(resShape, res);
        return Matrix<T>(res, resShape);
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
        assert(!areShapes1D(this->shape, rhs.shape) && "Shapes 1D and 1D invalid for matrix product");

        shape_t resShape{}; 

        long numE = 1;

        for(size_t i=0; i<this->shape.size()-1; i++)
        { 
            numE *= this->shape.at(i);
            resShape.push_back(this->shape.at(i));
        }

        numE *= rhs.shape.back();
        resShape.push_back(rhs.shape.back());

        Matrix<T> res = zeros(resShape);
        shape_t indexStack{}; 
        shape_t resElements = this->computeShapes(resShape); 
        long dim=0;

        this->matmul(rhs, res.data, indexStack, resElements, dim);

        return res;
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
            return this->dotProduct2D(Matrix<T>(this->data, this->shape), rhs, 0, 0);
        }

        assert("Shapes invalid for dot product" && dotShapesAssert(rhs.shape));
        shape_t resShape;
        long size = 1;

        for(size_t i=0; i<this->shape.size()-1; i++)
        { 
            size *= this->shape.at(i);
            resShape.push_back(this->shape.at(i));
        }

        size *= rhs.shape.back();
        resShape.push_back(rhs.shape.back()); // column dimension of the right hand side matrix
        
        std::vector<T> resdata(size, 0);
        shape_t indexStack{}; 

        auto resElements = this->computeShapes(resShape);
        
        //perform dot product
        this->matmul(rhs, resdata, indexStack, resElements, 0);
       
        return Matrix<T>(resdata, resShape);
        
    } 


    std::ostream& print(std::ostream &out, shape_t &indexStack, long dim)
        {
            if(indexStack.size()  == this->shape.size()-1)
            {
                // We are in the state where rhs and lhs matrices are both on 2d matrix format
                //find the position in the lhs array where we are at
                long lhsStart{0};

                for(unsigned long i{0}; i<indexStack.size(); i++)
                {
                    lhsStart += indexStack.at(i);
                }
                out<<" [";
                for(auto i{0}; i<this->shape.at(dim); i++)
                {
                    out<<this->data.at(lhsStart+i)<<",";
                }
                out<<"]\n";
                return out;
            }

            out <<"[\n";
            // Push the extra dimensions to the index stack and recursively traverse the indices, then pop once one the operation for that index has been done
            for(long i=0; i<this->shape[dim]; i++)
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
    Matrix<T> operator * (const T a, const Matrix<T> &rhs)
    {
        return Matrix<T>(rhs.data * a, rhs.shape);
    }

    template <typename T>
    Matrix<T> operator * (const Matrix<T> &lhs, const T a)
    {
        return Matrix<T>( a * lhs.data, lhs.shape);
    }
    
    template <typename T>
    Matrix<T> operator / (const Matrix<T> &lhs, const T a)
    {
        return Matrix<T>(lhs.data * (1/a), lhs.shape);
    }

    template <typename T>
    Matrix<T> operator / (const T a, const Matrix<T> &lhs)
    {
        return Matrix<T>( a / lhs.data, lhs.shape);
    }
    
    template <typename T>
    Matrix<T> operator ^(const Matrix<T> &lhs, const T a)
    {
        return Matrix<T>( lhs.data^a, lhs.shape);
    }
    
    
  
     template <typename T>
    Matrix<T> operator + (const Matrix<T> &lhs, const T a)
    {
        return Matrix<T>(lhs.data + a, lhs.shape);
    }

    template <typename T>
    Matrix<T> operator + (const T a, const Matrix<T> &lhs)
    {
        return Matrix<T>( a + lhs.data, lhs.shape);
    }
  
     template <typename T>
    Matrix<T> operator - (const Matrix<T> &lhs, const T a)
    {
        return Matrix<T>(lhs.data - a, lhs.shape);
    }

    template <typename T>
    Matrix<T> operator - (const T a, const Matrix<T> &lhs)
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

//------------------------------------------------------------------------------------

    template <typename T>
    Matrix<T> sumGradForBroadcast(Matrix<T> grad, shape_t originalShape){
        shape_t gradShape = grad.shape;
        int i = 0, j = 0;
        Matrix<T> res(grad);
        j = originalShape.size()-1;

        for(i = gradShape.size()-1; i >= 0; i--)
        {
            if(j >=0 && originalShape[j] == 1 && gradShape[i] > 1)
            {  
                res = res.sum(i);
                gradShape = res.shape;  
                j--;
            }
            else if(j >= 0 && originalShape[j] == gradShape[i])
            {
                j--;
            }
            else if(j < 0){
                res = res.sum(i);
                gradShape = res.shape;  
            }
        }

        if(res.shape != originalShape)
        { 
            res = Matrix<T>(res.data, originalShape);
        }  
        
        return res;
    }
#endif
