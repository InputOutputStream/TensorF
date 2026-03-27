#ifndef __MATRIX_CLASS_INCLUDED__
#define __MATRIX_CLASS_INCLUDED__

#include "types.hpp"
#include "header.hpp"
#include <stack>


using shape_t = std::vector<long>;

template <typename T>
class Matrix // : public std::enable_shared_from_this<Matrix<T>>
{

    protected:
    shape_t numElementsSeen{}; 

    bool verifyShape(const std::vector<T> &data, const shape_t &shape)
    {
        bool s = 1;
        unsigned long p = 1;
        for(unsigned long i = 0; i<shape.size(); i++)
            {
                p *=shape[i];
            }

            if(data.size() != p)
                s = false;   

            return s;
    }               

     //There is an error in the computes shapes method as we go from 1D to 2D 
        //Solution
        //I was using a class attr this->numElementsSeen instead of a local variable numElementsSeen which xas wronf since i am returning it
    shape_t computeShapes(const shape_t shape)
    {
        shape_t numElementsSeen(shape.size());
        long p{1};
        for(long i = (long)shape.size()-1; i>=0 ;i--)
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

        
    // Check if shapes are equal element wise in the std::vector 
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


    template <typename U> // cloudy
    void extractShape(const U& data, shape_t& shape)
    {
        if constexpr(std::is_same_v<U, T>){ 
            return; // scalar
        }else{
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
            Matrix<T> res = zeros({lhs.shape[0], rhs.shape[1]});
            
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

    void matmul(const Matrix<T> &rhs,
                std::vector<T> &res, 
                shape_t &indexStack, 
                shape_t &resElements, 
                long dim)
        {

        if(this->shape.size() == 1 || rhs.shape.size() == 1)
            {
                res = this->dot(rhs).data;
                return;
            }

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


    public:
    std::vector<T> data;
    shape_t shape;

    Matrix()=delete;

    template <typename U>
    Matrix(const U& data)
    {
        this->extractShape(data, this->shape);
        if(this->shape.size() == 0)
            this->shape.push_back(1);
        this->flattenReccursive(data, this->data);
        this->numElementsSeen = computeShapes(this->shape);
    }

    template <typename E>
    Matrix(std::initializer_list<E> data)
    {
        this->shape.push_back(data.size());
        this->extractShape(data, this->shape);
        if(this->shape.size() == 0)
            this->shape.push_back(1);
        this->flattenReccursive(data, this->data);
        this->numElementsSeen = this->computeShapes(this->shape);
    }

     Matrix(std::vector<T> data)
    {
        this->data = data;
        this->shape.push_back(data.size()); 
        this->numElementsSeen = this->computeShapes(this->shape);
    }

    Matrix(std::vector<T> data, shape_t shape)
    {
        assert("Shape and number of elements of matrix do not match" && this->verifyShape(data, shape));
        this->data = data;
        this->shape = shape; 
        this->numElementsSeen = this->computeShapes(this->shape);
    }

     Matrix(std::vector<std::vector<T>> data)
    {
        this->shape.push_back(data.size());
        this->shape.push_back(data.begin()->size());

        assert("Matrix shape must be uniform" && this->isRegular2D(data));
        this->flattenReccursive(data, this->data);
        this->numElementsSeen = this->computeShapes(this->shape);
    }

    Matrix(std::initializer_list<T> data)
    {
        this->shape.push_back(data.size());
        this->flattenReccursive(data, this->data);
        this->numElementsSeen = this->computeShapes(this->shape);
    }

    Matrix(std::initializer_list<std::initializer_list<T>> data)
    {
        this->shape.push_back(data.size());
        this->shape.push_back(data.begin()->size());
        
        assert("Matrix shape must be uniform" && this->isRegular2D(data));
        this->flattenReccursive(data, this->data);
        this->numElementsSeen = this->computeShapes(this->shape);
    }
    
    // Matrix Arithmetic Operations 
    
    Matrix<T> operator + (const Matrix<T> &rhs)
    {
        assert(areShapesEqual(rhs.shape) && "Shape and number of elements do not match matrix dimensions");
        return Matrix<T>(data + rhs.data, shape);
    }
    
    Matrix<T> operator - (const Matrix<T> &rhs)
    {
        assert(areShapesEqual(rhs.shape) && "Shape and number of elements do not match matrix dimensions");
        return Matrix<T>(data - rhs.data, shape);
    }

    Matrix<T> operator * (const Matrix<T> &rhs)
    {
        assert(areShapesEqual(rhs.shape) && "Shape and number of elements do not match matrix dimensions");
        return Matrix<T>(data * rhs.data, shape);
    }  
    
    Matrix<T> operator / (const Matrix<T> &rhs)
    {
        assert(areShapesEqual(rhs.shape) && "Shape and number of elements do not match matrix dimensions");
        return Matrix<T>(data / rhs.data, shape);
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

    // functions

    Matrix<T> matmul(const Matrix<T> &rhs)
    {
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
            return Matrix<T>({this->dotProduct1D(this->data, rhs.data)});

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
                out<<"]";
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
            out <<"\n]";
            return out;
        }


    template <typename E>
    friend std::ostream & operator <<(std::ostream &out, Matrix<E> &m);
};



    template <typename E>
    std::ostream& operator << (std::ostream &out, Matrix<E> &m)
    {
        //out<<m.data<<"\t";
        //out<<"Shape:"<<m.shape;
        shape_t stack;
        m.print(out, stack, 0);
        return out;
    }

#endif
