#ifndef __MATRIX_CLASS_INCLUDED__
#define __MATRIX_CLASS_INCLUDED__

#include "types.hpp"
#include "header.hpp"
#include <stack>

template <typename T>
class Matrix // : public std::enable_shared_from_this<Matrix<T>>
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
        
        bool isRegular2D(const std::vector<std::vector<T>> val)
        {
            if(val.size() == 0)
                return true;

            std::vector<T> j = val[0];
            for(int i=1; i<val.size(); i++)
            { 
                if(j.size() != val[i].size())
                    return false;
            }

            return true;
        }

        bool isRegular2D(const std::initializer_list<std::initializer_list<T>>& val)
        {
            if (val.size() == 0) return true;

            size_t cols = val.begin()->size();

            for (const auto& row : val)
            {
                if (row.size() != cols)
                    return false;
            }
            return true;
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
        bool areShapes1D(const std::vector<long> &shape)
        {
            if(shape.size() == this->shape.size() || this->shape.size() < 2)
                return true;

            if(shape.size() == this->shape.size() || this->shape.size() == 2)
                if((shape[0]+shape[1] < 4) && (this->shape[0] + this->shape[1] < 4))
                    return true;

            return false;
        }

        bool areShapes2D(const std::vector<long> &shape)
        {
            if(shape.size() == this->shape.size() || this->shape.size() == 2)
                if((this->shape[0] + this->shape[1] >= 4) && (shape[0] + shape[1] >= 4))
                    return true;
            
            return false;
        }

        bool areShapesEqual(const std::vector<long> &shape)
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


        void matProduct2D(const Matrix<T> &rhs, 
                        std::vector<T> &res, 
                        long lhsStart, 
                        long resStart)
        {
            
            long row1 = this->shape[this->shape.size()-2];
            long col1 = this->shape[this->shape.size()-1];
            //long row2 = rhs.shape[rhs.shape.size()-2];
            long col2 = rhs.shape[rhs.shape.size()-1];

            assert(row1 == col2 && "FATAL ERRROR, MATRIX PRODUCT ATTEMPTED ON INVALID MATRICES");
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
            Matrix<T> res(lhs.val, lhs.shape);
            
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
                        sum += lhs.val[lhsStart + i * col1 + j] * rhs.val[j*col2 + k];
                    }

                    res.val[(resStart + i*col2 + k)] = sum;

                }
            }

            return res;
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
            matmul(rhs, res, indexStack, resElements, dim+1);
            indexStack.pop_back(); 
        }
    }

    std::vector<T> flatten(std::vector<std::vector<T>> val)
    {
        std::vector<T> res;
        for(auto i : val)
        {
            for(auto j : i)
            {
                res.push_back(j);
            }
        }

        return res;
    }

    std::vector<T> flatten(const std::initializer_list<std::initializer_list<T>>& val)
    {
        std::vector<T> result;

        for (const auto& row : val)
        {
            result.insert(result.end(), row.begin(), row.end());
        }

        return result;
    }



    public:
    std::vector<T> val;
    std::vector<long> shape;

    Matrix()=delete;

    Matrix(std::vector<T> val)
    {
        this->val = val;
        this->shape.push_back(val.size()); 
        this->numElementsSeen = this->computeShapes(this->shape);
    }

    Matrix(std::vector<T> val, std::vector<long> shape)
    {
        assert("Shape and number of elements of matrix do not match" && this->verifyShape(val, shape));
        this->val = val;
        this->shape = shape; 
        this->numElementsSeen = this->computeShapes(this->shape);
    }

     Matrix(std::vector<std::vector<T>> val)
    {
        this->shape.push_back(val.size());
        this->shape.push_back(val.begin()->size());

        assert("Matrix shape must be uniform" && this->isRegular2D(val));
        this->val = this->flatten(val);
        this->numElementsSeen = this->computeShapes(this->shape);
    }

    Matrix(std::initializer_list<std::initializer_list<T>> val)
    {
        this->shape.push_back(val.size());
        this->shape.push_back(val.begin()->size());
        
        assert("Matrix shape must be uniform" && this->isRegular2D(val));
        this->val = this->flatten(val);
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
    public:

    Matrix<T> matmul(const Matrix<T> &rhs)
    {
        std::vector<T> res; 
        std::vector<long> indexStack; 
        std::vector<long> resElements; 
        long dim=0;

        matmul(rhs, res, indexStack, resElements, dim);

        return Matrix<T>(res, {this->shape[1], rhs.shape[0]});
    }
    

    Matrix<T> dot(const Matrix<T> &rhs)
    {
        if(areShapes1D(rhs.shape) && areShapes1D(this->shape))
            return Matrix<T>({this->dotProduct1D(this->val, rhs.val)}, {1});

        if(areShapes2D(rhs.shape) && areShapes2D(this->shape))
        {
            return this->dotProduct2D(Matrix<T>(this->val, this->shape), rhs, 0, 0);
        }

        assert("Shapes invalid for dot product" && areShapesEqual(rhs.shape));
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
        
        //perform dot product
        this->matmul(rhs, resVal, indexStack, resElements, 0);
       
        return Matrix<T>(resVal, resShape);
        
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
