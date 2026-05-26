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
#include "../Operations/ReshapeOperation.hpp"
#include "../Operations/ConcatOperation.hpp"
#include "../Operations/EmbeddingOperation.hpp"
#include "../Operations/IndexOperation.hpp"

#include "../Types/types.hpp"
#include "../Overloads/tensor_overloads.hpp"
#include "../Overloads/Overload.hpp"


template <typename T>
class Tensor : public std::enable_shared_from_this<Tensor<T>>
{
    public:
        Matrix<T> val; //Tensor value
        Matrix<T> grad; //Tensor gradian
        shape_t shape;
        size_t ndims;
        // Operation_t<T> frontOp = nullptr;
        Operation_t<T> backOp = nullptr;

    //....................................................................................................
    Tensor() //ov
    {
        this->val = 0;
    }

    Tensor(Matrix<T> *val) // ov
    {
        this->val.copy_from(val);
        this->shape = this->val.shape;
        this->ndims = this->val.get_ndims();
    }


    Tensor(const Matrix<T> &val) // ov
    {
        this->val.copy_from(val);
        this->shape = this->val.shape;
        this->ndims = this->val.get_ndims();
    }

    Tensor(Matrix<T> val, Operation_t<T> op)
    {
        this->val.copy_from(val);
        this->backOp = op;
        this->shape = this->val.shape;
        this->ndims = this->val.get_ndims();
    }

    Tensor(const Tensor_t<T> two) 
    {
        this->val.copy_from(two->val);
        this->backOp = nullptr;      
        // this->frontOp = nullptr;
        this->grad.copy_from(two->grad);
        this->shape = this->val.shape;
        this->ndims = this->val.get_ndims();
    }


//.....................................................................................

    void backward(Matrix<T> ingrad)
    {// x = x - f`(x)*x
        if (this->grad.get_size() > 0) {
            if (this->grad.shape != ingrad.shape)
                throw std::runtime_error("Gradient shape mismatch in Tensor::backward");
            this->grad += ingrad;
        }
        else
            this->grad.copy_from(ingrad);

        if (this->backOp != nullptr) {
            // this->backOp->to_string();
            auto op = this->backOp;
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
        }
    }

    void reset_graph() {
        if (this->backOp) {
            this->backOp->reset_graph(); // Tell the operation to clear its inputs
            this->backOp = nullptr;      // Break the cycle here!
        }
    }

    // Overloads..........................................................................

   Tensor<T>& operator=(const Tensor<T>& rhs)
    {
        this->val.copy_from(rhs.val);
        this->grad.copy_from(rhs.grad);
        this->backOp = rhs.backOp;
        // this->frontOp = rhs.frontOp;
        return *this;
    }

    size_t size(){
        return this->val.get_size();
    }
        
    // Functions In graph...........................................................................
    Tensor_t<T> matmul(Tensor_t<T> x) {
        auto op = std::make_shared<MatmulOperation<T>>(this->shared_from_this(), x);
        return op->forward();
    }

    Tensor_t<T> transpose() {
        auto op = std::make_shared<TransposeOperation<T>>(this->shared_from_this());
        return op->forward();
    }

    Tensor_t<T> transpose(shape_t inshape) {
        auto op = std::make_shared<TransposeOperation<T>>(this->shared_from_this());
        return op->forward(inshape);
    }

    Tensor_t<T> softmax() {
        auto op = std::make_shared<SoftmaxOperation<T>>(this->shared_from_this());
        return op->forward();
    }

    Tensor_t<T> relu() {
        auto op = std::make_shared<ReluOperation<T>>(this->shared_from_this());
        return op->forward();
    }

    Tensor_t<T> sum(size_t axis) {
        auto op = std::make_shared<SumAxisOperation<T>>(this->shared_from_this(), axis);
        return op->forward();
    }

    Tensor_t<T> sqrt() {
        auto p = make_tensor<T>((T)0.5);
        auto op = std::make_shared<PowerOperation<T>>(this->shared_from_this(), p);
        return op->forward();
    }

    Tensor_t<T> power(int n) {
        auto p = make_tensor<T>((T)n);
        auto op = std::make_shared<PowerOperation<T>>(this->shared_from_this(), p);
        return op->forward();
    }

    Tensor_t<T> ln() {
        auto op = std::make_shared<LogOperation<T>>(this->shared_from_this());
        return op->forward();
    }

    Tensor_t<T> exp() {
        auto op = std::make_shared<ExponentOperation<T>>(this->shared_from_this());
        return op->forward();
    }

    Tensor_t<T> sum() {
        auto op = std::make_shared<SumOperation<T>>(this->shared_from_this());
        return op->forward();
    }

    Tensor_t<T> sigmoid() {
        auto op = std::make_shared<SigmoidOperation<T>>(this->shared_from_this());
        return op->forward();
    }

    Tensor_t<T> reshape(shape_t new_shape) {
        auto op = std::make_shared<ReshapeOperation<T>>(this->shared_from_this(), new_shape);
        return op->forward();
    }

    Tensor_t<T> embed(Tensor_t<T> indices) {
        auto op = std::make_shared<EmbeddingOperation<T>>(this->shared_from_this(), indices);
        return op->forward();
    }

    Tensor_t<T> dot(Tensor_t<T> x)
    {
        auto op = std::make_shared<DotOperation<T>>((this->shared_from_this()), x);
        return op->forward();
    }

    Tensor_t<T> reshape(std::initializer_list<size_t> inshape)
    {
        shape_t sh = Matrix<T>::getShape(inshape);
        auto op = std::make_shared<ReshapeOperation<T>>(this->shared_from_this(), sh);
        return op->forward();
    }

    Tensor_t<T> bool_index(Tensor_t<bool> idx)
    {
        auto op = std::make_shared<IndexOperation<T>>(this->shared_from_this(), idx->val);
        return op ->forward();
    }

    Tensor_t<T> transpose(std::initializer_list<size_t> inshape)
    {
        auto op = std::make_shared<TransposeOperation<T>>(this->shared_from_this());
        return op->forward(inshape);
    }

    Tensor_t<T> cbrt()
    {
        auto p = make_tensor<T>((T)1/(T)3);
        auto op = std::make_shared<PowerOperation<T>>(this->shared_from_this(), p);
        return op->forward();
    }


    // Compose from graph 
    
    Tensor_t<T> mean(size_t axis) {
        // sum(axis) / N  
        T N = (T)this->val.shape[axis];
        auto s = this->sum(axis);          
        return s / make_tensor<T>(N);      
    }

    Tensor_t<T> var(size_t axis) {
        // var = mean((x - mean(x))^2)
        auto mu   = this->mean(axis);                          
        auto diff = this->shared_from_this() - mu;             
        auto sq   = diff * diff;                              
        return sq->mean(axis);                                 
    }

    Tensor_t<T> std(size_t axis) {
        return this->var(axis)->sqrt();                   
    }
 
    // Static on graph
    
    static Tensor_t<T> concat(std::vector<Tensor_t<T>> tens, size_t axis){
        auto concat_op = std::make_shared<ConcatOperation<T>>(tens, axis);
        return concat_op->forward();
    }

    // Static functions ********************************************************

    //loss functions

    // static Tensor_t<T> cross_entropy(Tensor_t<T> ytrue, Tensor_t<T> ypred)
    // {
    //     size_t N = ypred->val.shape[0];  // batch size only
    //     return -(ytrue * ypred->ln())->sum() / make_tensor<T>((T)N);
    // }

    static Tensor_t<T> cross_entropy(Tensor_t<T> ytrue, Tensor_t<T> ypred) {
        size_t N = ypred->val.shape[0];
        auto eps = make_tensor<T>((T)1e-9);
        auto safe_pred = ypred + eps;          // prevents log(0)
        return -(ytrue * safe_pred->ln())->sum() / make_tensor<T>((T)N);
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

        return pow((ytrue - ypred), (T)2)->sum() / (T)ytrue->val.shape[0];
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

    // Functions Off graph...........................................................................

    Tensor_t<T> maximum(int value)
    {
        return make_tensor<T>(this->val.maximum(value));
    }
   
    Tensor_t<T> at(std::initializer_list<size_t> idx)
    {
        shape_t index = Matrix<T>::getShape(idx);
        return make_tensor<T>(this->val.at(index));
    }

    Tensor_t<T> at(Tensor_t<bool> idx)
    {
        return make_tensor<T>(this->val.at(idx));
    }


    template<typename k>
    static Tensor_t<T> from(k in){
        return make_tensor<T>(Matrix<T>::from(in));
    }
};


#endif