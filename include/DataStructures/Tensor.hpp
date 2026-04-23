#ifndef __TENSOR_CLASS_INCLUDED__
#define __TENSOR_CLASS_INCLUDED__

#include <memory>
#include <vector>
#include <ranges>
#include "Matrix.hpp"

#include "../Operations/AddOperation.hpp"
#include "../Operations/MultiplyOperation.hpp"
#include "../Operations/DivisionOperation.hpp"
#include "../Operations/ExponentOperation.hpp"
#include "../Operations/SubtractOperation.hpp"
#include "../Operations/ReluOperation.hpp"
#include "../Operations/DotOperation.hpp"
#include "../Operations/MatmulOperation.hpp"
#include "../Operations/SigmoidOperation.hpp"
#include "../Operations/SumOperation.hpp"
#include "../Operations/LogOperation.hpp"
#include "../Operations/TransposeOperation.hpp"

#include "../Types/types.hpp"
#include "../Overloads/tensor_overloads.hpp"
#include "../Overloads/Overload.hpp"


template <typename T>
class Tensor : public std::enable_shared_from_this<Tensor<T>>
{
    public:
        Matrix<T> data; //Tensor value
        Matrix<T> grad; //Tensor gradian
        Operation_t<T> frontOp = nullptr, backOp = nullptr;

    //....................................................................................................
    Tensor() //ov
    {
        this->data = 0;
    }

    Tensor(Matrix<T> *data) // ov
    {
        this->data.copy_from(data);
    }


    Tensor(const Matrix<T> &data) // ov
    {
        this->data.copy_from(data);
    }

    Tensor(Matrix<T> data, Operation_t<T> op)
    {
        this->data.copy_from(data);
        this->backOp = op;
    }

    Tensor(const Tensor_t<T> two) //Const Copy Constructor
    {
        this->data.copy_from(two->data);
        this->backOp = two->backOp;
        this->frontOp = two->frontOp;
        this->grad.copy_from(two->grad);
    }


//.....................................................................................
    void backward(Matrix<T> ingrad)
    { // x = x - f`(x)*x
        if(this->grad.get_size() > 0)
            this->grad = this->grad + ingrad;
        else
            this->grad.copy_from(ingrad);

        if(this->backOp != nullptr)
        { 
            // this->backOp->to_string();
            this->backOp->backward(ingrad); 
            this->backOp = nullptr;
            this->frontOp = nullptr;
        }
    }

    void backward(Tensor_t<T> ingrad)
    { // x = x - f`(x)*x
        if(this->grad.get_size() > 0)
            this->grad = this->grad + ingrad->data;
        else
            this->grad.copy_from(ingrad->data);

        if(this->backOp != nullptr)
        { 
            // this->backOp->to_string();
            this->backOp->backward(ingrad->data); 
            this->backOp = nullptr;
            this->frontOp = nullptr;
        }
    }

    void zero_grad()
    {
        this->grad.clear();

        if(this->backOp != nullptr)
        { 
            this->backOp->zero_grad(); 
        }
    }

    void reset_graph()
    {
        if(this->frontOp!= nullptr)
        { 
            this->frontOp->reset_graph(); 
            this->frontOp = nullptr;
        }
    }

    // Overloads..........................................................................

   Tensor<T>& operator=(const Tensor<T>& rhs)
    {
        this->data.copy_from(rhs.data);
        this->grad.copy_from(rhs.grad);
        this->backOp = rhs.backOp;
        this->frontOp = rhs.frontOp;
        return *this;
    }

        
    // Functions In graph...........................................................................
    Tensor_t<T> exp()
    {
        this->frontOp = std::make_shared<ExponentOperation<T>>((this->shared_from_this()));
        return this->frontOp->forward(); 
    }

    Tensor_t<T> ln()
    {
        this->frontOp = std::make_shared<LogOperation<T>>((this->shared_from_this()));
        return this->frontOp->forward(); 
    }

    Tensor_t<T> relu()
    {
        this->frontOp = std::make_shared<ReluOperation<T>>((this->shared_from_this()));
        return this->frontOp->forward(); 
    }

    Tensor_t<T> matmul(Tensor_t<T> x)
    {
        this->frontOp = std::make_shared<MatmulOperation<T>>((this->shared_from_this()), x);
        return this->frontOp->forward(); 
    }

    Tensor_t<T> dot(Tensor_t<T> x)
    {
        this->frontOp = std::make_shared<DotOperation<T>>((this->shared_from_this()), x);
        return this->frontOp->forward(); 
    }

    Tensor_t<T> sigmoid()
    {
        this->frontOp = std::make_shared<SigmoidOperation<T>>(this->shared_from_this());
        return this->frontOp->forward();
    }
    
    Tensor_t<T> sum()
    {
        this->frontOp = std::make_shared<SumOperation<T>>(this->shared_from_this());
        return this->frontOp->forward();
    }

     Tensor_t<T> transpose(std::initializer_list<long> inshape)
    {
        this->frontOp = std::make_shared<TransposeOperation<T>>(this->shared_from_this());
        return this->frontOp->forward(inshape);
    }

    Tensor_t<T> transpose(shape_t inshape)
    {
        this->frontOp = std::make_shared<TransposeOperation<T>>(this->shared_from_this());
        return this->frontOp->forward(inshape);
    }

     Tensor_t<T> transpose()
    {
        this->frontOp = std::make_shared<TransposeOperation<T>>(this->shared_from_this());
        return this->frontOp->forward();
    }
    // Functions Off graph...........................................................................


    Tensor_t<T> softmax()
    {
        return this->exp() / (this->exp())->sum();
    }
    
    Tensor_t<T> sum(size_t axis)
    {
        return make_tensor<T>(this->data.sum(axis));
    }

    // Static functions ********************************************************

    // loss functions 
    static T cross_entropy(Tensor_t<T> ytrue, Tensor_t<T> ypred)
    {
        T loss;
        for(auto [yt, yp] : std::views::zip(ypred, ytrue))
        {
            loss += yt * yp->ln();
        }

        return -loss;
    }

    static T binary_cross_entropy(Tensor_t<T> ytrue, Tensor_t<T> ypred)
    {
        T loss;
        auto n = ytrue->size();

        for(auto [yt, yp] : std::views::zip(ypred, ytrue))
        {
            loss += yt * yt->ln() + (1 - yt)*(1 - yp)->ln(); 
        }

        return -loss;
    }

    static Tensor_t<T> mse(Tensor_t<T> ytrue, Tensor_t<T> ypred)
    {
        return (((ytrue - ypred) ^ (T)2)->sum()) / (T)(ytrue->data.shape[0]);
    }


    static Tensor_t<T> zeros(std::initializer_list<long> shape){
        return make_tensor<T>(Matrix<T>::zeros(shape));
    }

    static Tensor_t<T> ones(std::initializer_list<long> shape){
        return make_tensor<T>(Matrix<T>::ones(shape));
    }

    static Tensor_t<T> randn(std::initializer_list<long> shape){
        return make_tensor<T>(Matrix<T>::randomn(shape));
    }

    static Tensor_t<T> random(std::initializer_list<long> shape){
        return make_tensor<T>(Matrix<T>::random(shape));
    }

    static Tensor_t<T> eye(std::initializer_list<long> shape){
        return make_tensor<T>(Matrix<T>::eye(shape));
    }

    template<typename k>
    static Tensor_t<T> from(k in){
        return make_tensor<T>(Matrix<T>::from(in));
    }


    /**
     *  create a friend of a static
     *    
     *  template <typename E>
     *  friend std::ostream & operator <<(std::ostream &out, Matrix<E> &m);
     */

};


#endif