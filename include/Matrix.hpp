#ifndef __MATRIX_CLASS_INCLUDED__
#define __MATRIX_CLASS_INCLUDED__

#include "types.hpp"
#include "header.hpp"
#include <stack>

template <typename T>
class Matrix : public std::enable_shared_from_this<Matrix<T>>
{
    protected:
        std::vector<long> numElementsSeen{}; 

        bool verifyShape(const std::vector<T> &val, const std::vector<long> &shape)
        {
            bool s = 1;
            unsigned long p = 1;
            for(unsigned long i = 0; i<shape.size(); i++)
                {
                    p *=shape[i];
                }

                if(val.size() != p)
                    s = false;   

                return s;
        }                   
        

        //There is an error in the computes shapes method as we go from 1D to 2D 
        //Solution
        //I was using a class attr this->numElementsSeen instead of a local variable numElementsSeen which xas wronf since i am returning it
        std::vector<long> computeShapes(const std::vector<long> shape)
        {
            std::vector<long> numElementsSeen(shape.size());
            long p{1};
            for(long i = (long)shape.size()-1; i>=0 ;i--)
            {
                numElementsSeen[i] = p;
                p *= shape.at(i);
            }

            return numElementsSeen;
        }


        //Check if shapes are equal element wise in the std::vector 
        bool areShapesEqual(const std::vector<long> &shape)
        {
            if(shape.size() != this->shape.size())
            {
                return false;
            }

            for(long i = 0; i<this->shape.size(); i++)
                {
                    if(shape[i] != this->shape[i])
                    {
                        return false;
                    }
                }

            return true;
        }

        bool matProductAssert(const std::vector<long> &shape)
        {
            if(this->shape.size() < 2 || shape.size() != 2)
                return false;

            long row1 = this->shape[shape.size()-2];
            // long col1 = this->shape[shape.size()-1];
            // long row2 = shape[shape.size()-2];
            long col2 = shape[shape.size()-1];

            if(col2 != row1)
                return false;
            
            return true;
        }

    public:
    std::vector<T> val;
    std::vector<long> shape;

    Matrix()=delete;

    Matrix(std::vector<T> val, std::vector<long> shape)
    {
        assert("Shape and number of elements of matrix do not match" && this->verifyShape(val, shape));
        this->val = val;
        this->shape = shape; 
        this->numElementsSeen = this->computeShapes(this->shape);
    }

    
// Matrix Arithmetic Operations 
    Matrix<T> operator + (const Matrix<T> &rhs)
    {
        assert(areShapesEqual(rhs.shape) && "Shape and number of elements do not match matrix dimensions");
        return Matrix<T>(val + rhs.val, shape);
    }
    
    Matrix<T> operator - (const Matrix<T> &rhs)
    {
        assert(areShapesEqual(rhs.shape) && "Shape and number of elements do not match matrix dimensions");
        return Matrix<T>(val - rhs.val, shape);
    }

    Matrix<T> operator * (const Matrix<T> &rhs)
    {
        assert(areShapesEqual(rhs.shape) && "Shape and number of elements do not match matrix dimensions");
        return Matrix<T>(val * rhs.val, shape);
    }  
    
    Matrix<T> operator / (const Matrix<T> &rhs)
    {
        assert(areShapesEqual(rhs.shape) && "Shape and number of elements do not match matrix dimensions");
        return Matrix<T>(val / rhs.val, shape);
    }

// Matrix Operations  ..........................................................................................   

    void matProduct2D(const Matrix<T> &rhs, 
                    std::vector<T> &res, 
                    long lhsStart, 
                    long resStart)
    {
        
        long row1 = this->shape[this->shape.size()-2];
        long col1 = this->shape[this->shape.size()-1];
        long row2 = rhs.shape[rhs.shape.size()-2];
        long col2 = rhs.shape[rhs.shape.size()-1];

        assert(col1 == row2 && "FATAL ERRROR, MATRIX PRODUCT ATTEMPTED ON INVALID MATRICES");
        T sum = 0;

        for(long i=0; i<row1; i++)
        {
            for(long k =0; k<col2; k++)
            {
                sum = 0;
                for(long j=0; j<col1; j++)
                {
                    sum += this->val[lhsStart + i * col1 + j] * rhs.val[j*col2 + k];
                }

                res[(resStart + i*col2 + k)] = sum;

            }
        }
    }

    void matmul(const Matrix<T> &rhs,
                std::vector<T> &res, 
                std::vector<long> &indexStack, 
                std::vector<long> &resElements, 
                long dim)
    {

        if(indexStack.size()  == (this->shape.size()-2))
        {
            // We are in the state where rhs and lhs matrices are both on 2d matrix format
            // find the position in the lhs array where we are at
            
            long lhsStart{0};
            long resStart {0};
            //rhsStart and lhsStart denotes respectively the starting polongs of the multiplication within the matrix dimensions

            for(unsigned long i{0}; i<indexStack.size(); i++)
            {
                lhsStart += indexStack.at(i) * this->numElementsSeen.at(i);
                resStart += indexStack.at(i) * resElements.at(i);
            }

            matProduct2D(rhs, res, lhsStart, resStart);
            return;
        }


        // Push the extra dimensions to the index stack and recursively traverse the indices, the pop one one the operation for that index has been done
        for(long i=0; i<this->shape[dim]; i++)
        {
            indexStack.push_back(i);
            matmul(rhs, res, indexStack,resElements, dim+1);
            indexStack.pop_back(); 
        }
    }



    std::ostream& print(std::ostream &out, std::vector<long> &indexStack, long dim)
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
                out<<"[";
                for(auto i{0}; i<this->shape.at(dim); i++)
                {
                    out<<this->val.at(lhsStart+i)<<" ";
                }
                out<<"]";
                return out;
            }

            out <<"[";
            // Push the extra dimensions to the index stack and recursively traverse the indices, the pop one one the operation for that index has been done
            for(long i=0; i<this->shape[dim]; i++)
            {
                indexStack.push_back(this->numElementsSeen[dim] * i);//calculate how many elements have been processed to get the pointer to the right location in val and push to the stack
                print(out, indexStack, dim+1);
                indexStack.pop_back(); //pops out of stack
            }
            out <<"]"<<std::endl;
            return out;
        }




    bool dotProductAssert(std::vector<long> shape){

    }




    Matrix<T> dot(const Matrix<T> &rhs)
    {
        assert("Shapes invalid for dot product" && dotProductAssert(rhs.shape));

        std::vector<long> resShape;
        auto size = 1;
        for(unsigned long i=0; i<this->shape.size()-1; i++)
        { 
            size *= this->shape.at(i);
            resShape.push_back(this->shape.at(i));
        }

        resShape.push_back(rhs.shape.at(1)); // column dimension of the right hand side matrix
        std::vector<T> resVal(size, 0);
        std::vector<long> indexStack(0,0); 

        auto resElements = this->computeShapes(resShape);
        
        //perform matrix multiplication
        this->matmul(rhs, resVal, indexStack, resElements, 0);
       
        return Matrix<T>(resVal, resShape);
        
    }


    template <typename E>
    friend std::ostream & operator <<(std::ostream &out, Matrix<E> &m);
};



    template <typename E>
    std::ostream& operator << (std::ostream &out, Matrix<E> &m)
    {
        //out<<m.val<<"\t";
        //out<<"Shape:"<<m.shape;
        std::vector<long> stack;
        m.print(out, stack, 0);
        return out;
    }

#endif
