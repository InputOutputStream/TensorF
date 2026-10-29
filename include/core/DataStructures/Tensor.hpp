#ifndef __TENSOR_CLASS_INCLUDED__
#define __TENSOR_CLASS_INCLUDED__

#include <memory>
#include <vector>
#include <ranges>
#include <unordered_set>

#include "Matrix.hpp"

#include "core/Operations/AddOperation.hpp"
#include "core/Operations/MultiplyOperation.hpp"
#include "core/Operations/DivisionOperation.hpp"
#include "core/Operations/ExponentOperation.hpp"
#include "core/Operations/SubtractOperation.hpp"
#include "core/Operations/ReluOperation.hpp"
#include "core/Operations/DotOperation.hpp"
#include "core/Operations/MatmulOperation.hpp"
#include "core/Operations/SigmoidOperation.hpp"
#include "core/Operations/SumOperation.hpp"
#include "core/Operations/LogOperation.hpp"
#include "core/Operations/TransposeOperation.hpp"
#include "core/Operations/SumAxisOperation.hpp"
#include "core/Operations/SoftmaxOperation.hpp"
#include "core/Operations/ReshapeOperation.hpp"
#include "core/Operations/ConcatOperation.hpp"
#include "core/Operations/EmbeddingOperation.hpp"
#include "core/Operations/IndexOperation.hpp"
#include "core/Operations/spMMOperation.hpp"
#include "core/Operations/MaxAxisOperation.hpp"
#include "core/Operations/LogSoftmaxOperation.hpp"
#include "core/Operations/GatherOperation.hpp"
#include "core/Operations/DroupoutOperation.hpp"

#include "core/Types/types.hpp"
#include "core/Overloads/tensor_overloads.hpp"
#include "core/Overloads/Overload.hpp"



// ────────────────────────────────────────────────────────────────────────────
// NoGrad scope — suppresses autograd graph construction in Tensor<T>.
//
//   {
//       mxd::NoGrad guard;
//       auto ev = l2.forward(adj, l1.forward(adj, X)->relu());   // no graph
//       double acc = accuracy(ev, ...);
//   }   // ~NoGrad restores the previous state (nestable)
//
// While active:
//   * Tensor(Matrix, Operation_t) sets backOp = nullptr, requires_grad = false
//   * Tensor(Matrix)        (leaves) sets requires_grad = false
//   * Tensor::backward()    short-circuits
// The val matrices are still computed — only the bookkeeping is skipped.
// ────────────────────────────────────────────────────────────────────────────
namespace mxd {
inline thread_local bool g_no_grad = false;

class NoGrad {
    bool prev_;
public:
    NoGrad() noexcept : prev_(g_no_grad) { g_no_grad = true; }
    ~NoGrad() { g_no_grad = prev_; }
    NoGrad(const NoGrad&) = delete;
    NoGrad& operator=(const NoGrad&) = delete;
};
} // namespace mxd

template <typename T>
class Tensor : public std::enable_shared_from_this<Tensor<T>>
{
    public:
        Matrix<T> val;
        Matrix<T> grad;
        shape_t shape;
        size_t ndims;
        Operation_t<T> backOp = nullptr;
        bool requires_grad = true;
    //....................................................................................................
    Tensor() {
        this->val  = Matrix<T>(T(0));
        this->ndims = 0;
        this->shape = {};
        this->grad = Matrix<T>(T(0));
    }

    Tensor(Matrix<T> *val)
    {
        this->val.copy_from(val);
        this->shape = this->val.shape;
        this->ndims = this->val.get_ndims();
        this->grad = Matrix<T>();
    }

    // J4: remove the duplicated copy_from(val).
    Tensor(const Matrix<T> &val)
    {
        this->val.copy_from(val);
        this->grad = Matrix<T>();
        this->shape = this->val.shape;
        this->ndims = this->val.get_ndims();
        if (mxd::g_no_grad) this->requires_grad = false;   // ← new
    }

    Tensor(Matrix<T> val, Operation_t<T> op)
    {
        this->val.copy_from(val);
        if (mxd::g_no_grad) {
            this->backOp = nullptr;
            this->requires_grad = false;
        } else {
            this->backOp = op;
            bool any_req = false;
            for (auto& in : op->inputs())
                if (in && in->requires_grad) { any_req = true; break; }
            this->requires_grad = any_req;          // ← new
        }
        this->shape = this->val.shape;
        this->ndims = this->val.get_ndims();
        this->grad = Matrix<T>();
    }

    Tensor(const Tensor_t<T> two)
    {
        this->val.copy_from(two->val);
        this->backOp = nullptr;
        this->grad.copy_from(two->grad);
        this->shape = this->val.shape;
        this->ndims = this->val.get_ndims();
    }

//.....................................................................................
 
    // Every tensor reachable from this one through backOp->inputs(), in topological order:
    // inputs come before the tensors computed from them, `this` is last. Iterative DFS (no
    // recursion depth limit), each node appears once, input order is fixed => deterministic.
    std::vector<Tensor<T>*> topo_order()
    {
        struct Frame { Tensor<T>* n; std::vector<Tensor_t<T>> in; size_t i = 0; };
        std::vector<Tensor<T>*> order;
        std::unordered_set<Tensor<T>*> seen{this};
        std::vector<Frame> st;
        st.push_back({this, backOp ? backOp->inputs() : std::vector<Tensor_t<T>>{}});
        while (!st.empty()) {
            Frame& f = st.back();
            if (f.i < f.in.size()) {
                Tensor<T>* c = f.in[f.i++].get();
                if (c && seen.insert(c).second)
                    st.push_back({c, c->backOp ? c->backOp->inputs() : std::vector<Tensor_t<T>>{}});
            } else { order.push_back(f.n); st.pop_back(); }
        }
        return order;
    }
 
    // Adds a gradient contribution; never recurses. Ops call this on their inputs.
    void accumulate(const Matrix<T>& g)
    {
        if (!requires_grad) return;
        if (grad.get_size() > 0) {
            if (grad.shape != g.shape)
                throw std::runtime_error("Gradient shape mismatch in Tensor::accumulate");
            grad += g;
        } else grad.copy_from(g);
    }
 
    // Reverse-mode driver: each node runs exactly once, after all of its consumers.
    void backward(Matrix<T> seed)
    {
        if (mxd::g_no_grad) return;              // ← new
        auto order = topo_order();
        // Gradients of computed (non-leaf) tensors are per-call scratch; only leaves accumulate across calls,
        // so a second backward() without zero_grad() adds exactly one more contribution to each leaf.
        for (Tensor<T>* n : order) if (n->backOp) n->grad.clear();
        accumulate(seed);
        for (auto it = order.rbegin(); it != order.rend(); ++it) {
            Tensor<T>* n = *it;
            if (n->grad.get_size() == 0) continue;
            if (Matrix<T>::hasNaN(n->grad)) {        
                if (n->backOp) n->backOp->to_string();
                std::cerr << "grad value: " << n->grad << "\n";
                throw std::runtime_error("Gradient NaN encountered during backprop");
            }
            if (n->backOp && n->requires_grad)
                n->backOp->backward(n->grad);       // pushes into the inputs via accumulate()
        }
    }
 

    void backward(Tensor_t<T> ingrad)
    {
        this->backward(ingrad->val);
    }

    // Clears the gradient of this tensor and everything it was computed from (each node once).
    void zero_grad()
    {
        for (Tensor<T>* n : topo_order()) n->grad.clear();
    }

    void reset_graph() {
        if (this->backOp) {
            this->backOp->reset_graph();
            this->backOp = nullptr;
        }
    }

    // J4: operator= must also copy shape / ndims / requires_grad.
    Tensor<T>& operator=(const Tensor<T>& rhs)
    {
        this->val.copy_from(rhs.val);
        this->grad.copy_from(rhs.grad);
        this->backOp = rhs.backOp;
        this->shape = rhs.shape;
        this->ndims = rhs.ndims;
        this->requires_grad = rhs.requires_grad;
        return *this;
    }

    size_t size(){
        return this->val.get_size();
    }

    Tensor_t<T> max() {
        return make_tensor<T>(this->val.max());
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

    Tensor_t<T> dropout(float p, std::mt19937& rng) {
        auto op = std::make_shared<DropoutOperation<T>>(this->shared_from_this(), p, rng);
        return op->forward();
    }

    Tensor_t<T> cbrt()
    {
        auto p = make_tensor<T>((T)1/(T)3);
        auto op = std::make_shared<PowerOperation<T>>(this->shared_from_this(), p);
        return op->forward();
    }

    Tensor_t<T> max(size_t axis) {
        auto op = std::make_shared<MaxAxisOperation<T>>(this->shared_from_this(), axis);
        return op->forward();
    }

    Tensor_t<T> log_softmax() {
        auto op = std::make_shared<LogSoftmaxOperation<T>>(this->shared_from_this());
        return op->forward();
    }

    Tensor_t<T> gather(Tensor_t<T> labels) {
        auto op = std::make_shared<GatherOperation<T>>(this->shared_from_this(), labels);
        return op->forward();
    }

    // Compose from graph

    Tensor_t<T> mean(size_t axis) {
        T N = (T)this->val.shape[axis];
        auto s = this->sum(axis);
        return s / make_tensor<T>(N);
    }

    // J3: scalar mean over all elements, used by rms_norm.
    Tensor_t<T> mean() {
        T N = (T)this->val.data.size();
        auto s = this->sum();
        return s / make_tensor<T>(N);
    }

    Tensor_t<T> var(size_t axis) {
        auto mu   = this->mean(axis);
        auto diff = this->shared_from_this() - mu;
        auto sq   = diff * diff;
        return sq->mean(axis);
    }

    Tensor_t<T> std(size_t axis) {
        return this->var(axis)->sqrt();
    }

    // Static functions ********************************************************

    static Tensor_t<float> dropout(Tensor_t<float> x, float p, std::mt19937& rng) {
        auto op = std::make_shared<DropoutOperation<float>>(x, p, rng);
        return op->forward();
    }

    static Tensor_t<T> concat(std::vector<Tensor_t<T>> tens, size_t axis){
        auto concat_op = std::make_shared<ConcatOperation<T>>(tens, axis);
        return concat_op->forward();
    }

    static Tensor_t<T> cross_entropy(Tensor_t<T> ytrue, Tensor_t<T> ypred, T eps = 1e-8) {
        size_t N = ypred->val.shape[0];
        auto Teps = make_tensor<T>(eps);
        auto safe_pred = ypred + Teps;
        return -(ytrue * safe_pred->ln())->sum() / make_tensor<T>((T)N);
    }

    static Tensor_t<T> masked_cross_entropy(Tensor_t<T> Y, Tensor_t<T> prob, size_t n_labelled) {
        return -(Y * prob->ln())->sum() / make_tensor<T>((T)n_labelled);
    }

    // Sum reduction by default would break every downstream caller;
    // F.binary_cross_entropy uses reduction='mean'.
    static Tensor_t<T> binary_cross_entropy(Tensor_t<T> ytrue, Tensor_t<T> ypred, bool mean = true)
    {
        auto one  = make_tensor<T>((T)1);
        auto lhs  = ytrue * ypred->ln();
        auto rhs  = (one - ytrue) * (one - ypred)->ln();
        auto total = -(lhs + rhs)->sum();
        if (mean) return total / make_tensor<T>((T)ytrue->val.data.size());
        return total;
    }

    static Tensor_t<T> cross_entropy_from_logits(Tensor_t<T> logits, Tensor_t<T> labels) {
        return -logits->log_softmax()->gather(labels);
    }

    // Mean of the per-element loss over the positions where `mask[i] > 0`.
    // n_labelled is passed explicitly so an empty mask is well-defined.
    static Tensor_t<T> masked_cross_entropy_from_logits(
        Tensor_t<T> logits, Tensor_t<T> labels, Tensor_t<T> mask, size_t n_labelled)
    {
        auto per_elem = cross_entropy_from_logits(logits, labels);   // [N]
        return (per_elem * mask)->sum() / make_tensor<T>((T)n_labelled);
    }

    // Numerically stable BCE from raw logits:
    //   loss(x, z) = max(x, 0) - x·z + log(1 + exp(-|x|))
    // Never overflows, never logs zero, well-defined for any z ∈ [0, 1] and any x.
    static Tensor_t<T> binary_cross_entropy_with_logits(Tensor_t<T> ytrue, Tensor_t<T> logits, bool mean = true)
    {
        auto one = make_tensor<T>((T)1);

        // |x| = relu(x) + relu(-x) — subgradient at 0 is 0 (safe, no NaN)
        auto abs_x    = logits->relu() + (-logits)->relu();

        // softplus(-|x|) = log(1 + exp(-|x|)), always in [0, log 2]
        auto softplus = (one + (-abs_x)->exp())->ln();

        // per-element loss
        auto per_elem = logits->relu() - logits * ytrue + softplus;

        auto total = per_elem->sum();
        if (mean) return total / make_tensor<T>((T)ytrue->val.data.size());
        return total;
    }

    static Tensor_t<T> mse(Tensor_t<T> ytrue, Tensor_t<T> ypred)
    {
        return pow((ytrue - ypred), (T)2)->sum() / (T)ytrue->val.data.size();
    }

    static Tensor_t<T> rms_norm(Tensor_t<T> x, Tensor_t<T> weight, T eps = 1e-5) {
        auto rms = (x * x)->mean() + eps;  // uses the new no-arg mean()
        return (x / rms->sqrt()) * weight;
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

    static Tensor_t<T> GeLU(Tensor_t<T> x)
    {
       return x * ((x * (T)1.702)->sigmoid());
    }

    static Tensor_t<T> SiLU(Tensor_t<T> x){
        return x * x->sigmoid();
    }

    static Tensor_t<T> randn(std::initializer_list<size_t> shape, std::optional<unsigned int> seed = std::nullopt){
        return make_tensor<T>(Matrix<T>::randomn(shape, seed));
    }

    static Tensor_t<T> random(std::initializer_list<size_t> shape, std::optional<unsigned int> seed = std::nullopt){
        return make_tensor<T>(Matrix<T>::random(shape, seed));
    }

    static Tensor_t<T> eye(std::initializer_list<size_t> shape){
        return make_tensor<T>(Matrix<T>::eye(shape));
    }

    Tensor_t<T> maximum(int value)
    {
        return make_tensor<T>(this->val.maximum(value));
    }

    Tensor_t<T> at(std::initializer_list<size_t> idx)
    {
        shape_t index = Matrix<T>::getShape(idx);
        return make_tensor<T>(this->val.at(index));
    }

    // J2: pass idx->val (Matrix<bool>), not idx (Tensor_t<bool>).
    Tensor_t<T> at(Tensor_t<bool> idx)
    {
        return make_tensor<T>(this->val.at(idx->val));
    }

    template<typename k>
    static Tensor_t<T> from(k in){
        return make_tensor<T>(Matrix<T>::from(in));
    }
};

#endif