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
#include "../Operations/SumAxisOperation.hpp"
#include "../Operations/SoftmaxOperation.hpp"
 
#include "../Types/types.hpp"
#include "../Overloads/tensor_overloads.hpp"
#include "../Overloads/Overload.hpp"


template <typename T>
class Tensor : public std::enable_shared_from_this<Tensor<T>>
{
    public:
        Matrix<T> val; //Tensor value
        Matrix<T> grad; //Tensor gradian
        Operation_t<T> frontOp = nullptr, backOp = nullptr;

    //....................................................................................................
    Tensor() //ov
    {
        this->val = 0;
    }

    Tensor(Matrix<T> *val) // ov
    {
        this->val.copy_from(val);
    }


    Tensor(const Matrix<T> &val) // ov
    {
        this->val.copy_from(val);
    }

    Tensor(Matrix<T> val, Operation_t<T> op)
    {
        this->val.copy_from(val);
        this->backOp = op;
    }

    Tensor(const Tensor_t<T> two) 
    {
        this->val.copy_from(two->val);
        this->backOp = nullptr;      
        this->frontOp = nullptr;
        this->grad.copy_from(two->grad);
    }


//.....................................................................................

    void backward(Matrix<T> ingrad)
    {// x = x - f`(x)*x
        if (this->grad.get_size() > 0) {
            if (this->grad.shape != ingrad.shape)
                throw std::runtime_error("Gradient shape mismatch in Tensor::backward");
            this->grad = this->grad + ingrad;
            //return;
        }
        else
            this->grad.copy_from(ingrad);

        if (this->backOp != nullptr) {
            // this->backOp->to_string();
            auto op = this->backOp;
            this->backOp = nullptr;   
            op->backward(ingrad);
        }
    }
    
    void backward(Tensor_t<T> ingrad)
    { // x = x - f`(x)*x
        this->backward(ingrad->val);
    }

    void zero_grad()
    {
        this->grad.clear();

        if(this->backOp != nullptr)
        { 
            this->backOp->zero_grad(); 
            this->backOp = nullptr;
            this->frontOp = nullptr;
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
        this->val.copy_from(rhs.val);
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

    size_t size(){
        return this->val.get_size();
    }

    Tensor_t<T> at(std::initializer_list<size_t> idx)
    {
        shape_t index = Matrix<T>::getShape(idx);
        return make_tensor<T>(this->val.at(index));
    }

    Tensor_t<T> transpose(std::initializer_list<size_t> inshape)
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

    Tensor_t<T> sqrt()
    {
        auto p = make_tensor<T>(T(1)/T(2));
        this->frontOp = std::make_shared<PowerOperation<T>>(this->shared_from_this(), p);
        return this->frontOp->forward();
    }

    Tensor_t<T> cbrt()
    {
        auto p = make_tensor<T>(T(1)/T(3));
        this->frontOp = std::make_shared<PowerOperation<T>>(this->shared_from_this(), p);
        return this->frontOp->forward();
    }

    Tensor_t<T> power(int n)
    {
        auto p = make_tensor<T>(n);
        this->frontOp = std::make_shared<PowerOperation<T>>(this->shared_from_this(), p);
        return this->frontOp->forward();
    }

    Tensor_t<T> softmax()
    {
        this->frontOp = std::make_shared<SoftmaxOperation<T>>(this->shared_from_this());
        return this->frontOp->forward();
    }

    Tensor_t<T> sum(size_t axis)
    {
        this->frontOp = std::make_shared<SumAxisOperation<T>>(this->shared_from_this(), axis);
        return this->frontOp->forward();
    }

    // Functions Off graph...........................................................................


   


    // Static functions ********************************************************

    //loss functions
    static Tensor_t<T> cross_entropy(Tensor_t<T> ytrue, Tensor_t<T> ypred)
    {
        size_t N = ypred->val.shape[0];  // batch size only
        return -(ytrue * ypred->ln())->sum() / make_tensor<T>((T)N);
    }

    // Binary Cross Entropy Loss: -sum(y * log(p) + (1-y) * log(1-p))
    static Tensor_t<T> binary_cross_entropy(Tensor_t<T> ytrue, Tensor_t<T> ypred)
    {
        auto lhs = ytrue * ypred->ln();
        auto rhs = (make_tensor<T>((T)1) - ytrue) * (make_tensor<T>((T)1) - ypred)->ln();
        return -(lhs + rhs)->sum();
    }
   

    static Tensor_t<T> mse(Tensor_t<T> ytrue, Tensor_t<T> ypred)
    {
        /*
        
        auto diff = ytrue - ypred;
        auto two = make_tensor<T>(2);
        auto sq = diff ^ two;          // tensor ^ tensor, no copy
        auto N   = make_tensor<T>((T)ytrue->val.shape[0]);
        auto loss = sq->sum() / N;     // tensor / tensor
        
        */
        return (((ytrue - ypred) ^ (T)2)->sum()) / (T)(ytrue->val.shape[0]);
    }


    static Tensor_t<T> transpose(Tensor_t<T> ten){
        return ten->transpose();
    }

    static Tensor_t<T> transpose(Tensor_t<T> ten, std::initializer_list<size_t> inshape){
        return ten->transpose(inshape);
    }

    static Tensor_t<T> transpose(Tensor_t<T> ten,  shape_t inshape){
        return ten->transpose(inshape);
    }


    static Tensor_t<T> zeros(std::initializer_list<size_t> shape){
        return make_tensor<T>(Matrix<T>::zeros(shape));
    }

    static Tensor_t<T> ones(std::initializer_list<size_t> shape){
        return make_tensor<T>(Matrix<T>::ones(shape));
    }

    static Tensor_t<T> randn(std::initializer_list<size_t> shape){
        return make_tensor<T>(Matrix<T>::randomn(shape));
    }

    static Tensor_t<T> random(std::initializer_list<size_t> shape){
        return make_tensor<T>(Matrix<T>::random(shape));
    }

    static Tensor_t<T> eye(std::initializer_list<size_t> shape){
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