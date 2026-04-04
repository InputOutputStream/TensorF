# TensorF

> A minimal deep learning engine built from scratch in C++ — tensors, automatic differentiation, and a full N-dimensional matrix library, with zero external dependencies.

---

## What is TensorF?

TensorF is a ground-up implementation of the core machinery behind frameworks like PyTorch and TensorFlow, written entirely in C++. The goal is not to replace those frameworks — it's to understand exactly what happens inside them.

At its heart, TensorF is:
- A **dynamic computation graph** that records operations as they happen
- A **reverse-mode automatic differentiation** engine (autograd) that propagates gradients backward through that graph
- A **generic N-dimensional Matrix class** supporting dot products, matrix multiplication, element-wise ops, and arbitrary shapes
- A **template-based Tensor system** with full operator overloading for natural mathematical syntax

---

## Architecture

```
TensorF/
├── types.hpp              # Forward declarations, shared_ptr aliases (Tensor_t, Operation_t)
├── header.hpp             # Central include hub
├── Tensor.hpp             # Core Tensor class — values, gradients, graph edges
├── Operation.hpp          # Abstract base class for all differentiable operations
├── AddOperation.hpp       # f(x,y) = x + y
├── SubtractOperation.hpp  # f(x,y) = x - y
├── MultiplyOperation.hpp  # f(x,y) = x * y
├── DivisionOperation.hpp  # f(x,y) = x / y
├── ExponentOperation.hpp  # f(x) = e^x
├── operation_impl.hpp     # Forward + backward implementations for all ops
├── Overload.hpp           # Vector arithmetic, scalar ops, element-wise math
├── Matrix.hpp             # N-dimensional Matrix with matmul, dot, reshape
└── main.cpp               # Usage examples and tests
```

---

## Core Concepts

### Tensors & Computation Graph

Every `Tensor` holds a value (`std::vector<T>`), a gradient (`std::vector<T>`), and two operation pointers: `backOp` (where it came from) and `frontOp` (what it feeds into). When you write:

```cpp
Tensor_t<float> a = make_tensor<float>({2.0f});
Tensor_t<float> b = make_tensor<float>({3.0f});
Tensor_t<float> c = a * b;  // c = 6.0
```

...a `MultiplyOperation` node is created and linked into the graph. No computation graph is explicitly constructed — it builds itself dynamically from operator overloads.

### Automatic Differentiation (Autograd)

Calling `backward()` on the output tensor initiates reverse-mode differentiation. Each operation knows the chain rule for its inputs:

| Operation | ∂f/∂x | ∂f/∂y |
|-----------|--------|--------|
| `x + y`   | grad   | grad   |
| `x - y`   | grad   | grad   |
| `x * y`   | grad × y | grad × x |
| `x / y`   | grad / y | −grad·x / y² |
| `exp(x)`  | grad × exp(x) | — |

```cpp
c->backward({1.0f});  // seed gradient = 1
// a->grad now contains dc/da = b = 3.0
// b->grad now contains dc/db = a = 2.0
```

### Full Example: Sigmoid Neuron

```cpp
Tensor_t<float> w0 = make_tensor<float>({2.0f});
Tensor_t<float> x0 = make_tensor<float>({-1.0f});
Tensor_t<float> w1 = make_tensor<float>({-3.0f});
Tensor_t<float> x1 = make_tensor<float>({-2.0f});
Tensor_t<float> bias = make_tensor<float>({-3.0f});

Tensor_t<float> one = make_tensor<float>({1.0f});

// σ(w0·x0 + w1·x1 + bias)
Tensor_t<float> z = (float)-1 * (w0*x0 + w1*x1 + bias);
Tensor_t<float> out = one / (one + z->exp());

out->backward({1.0f});

// Gradients available:
// w0->grad, x0->grad, w1->grad, x1->grad, bias->grad
```

This is precisely how backpropagation through a sigmoid neuron works — the same computation that sits at the heart of neural network training.

---

## Matrix Engine

The `Matrix<T>` class (in `Matrix.hpp`) provides a full N-dimensional array with:

- Construction from flat vectors, 2D vectors, or initializer lists
- **Recursive matmul** for batched N-dimensional matrix multiplication
- **Dot product** with automatic 1D/2D dispatch
- Element-wise `+`, `-`, `*`, `/` with shape assertions
- Pretty-printing with bracket notation

```cpp
Matrix<float> A({{1,2,3},{4,5,6}});   // shape: [2,3]
Matrix<float> B({{7,8},{9,10},{11,12}}); // shape: [3,2]

Matrix<float> C = A.dot(B);  // shape: [2,2]
std::cout << C;
// [[58,64],[139,154]]
```

The matmul is implemented recursively: it traverses batch dimensions via an index stack, then delegates to a 2D kernel at the innermost level — the same strategy used in production tensor libraries.

---

## Building

```bash
git clone https://github.com/InputOutputStream/TensorF
cd TensorF
make
./main
```

**Requirements:** C++17, g++ or clang++. No external dependencies.

---

## Current Capabilities

- [x] Scalar and vector tensors
- [x] Dynamic computation graph
- [x] Reverse-mode autograd (add, sub, mul, div, exp)
- [x] N-dimensional matrix multiplication (batched, recursive)
- [x] Operator overloading for natural math syntax
- [x] Sigmoid neuron example with full gradient computation

## Roadmap

- [ ] Activation functions (ReLU, Tanh, Softmax)
- [ ] Loss functions (MSE, Cross-Entropy)
- [ ] Optimizer (SGD, Adam)
- [ ] Shape broadcasting
- [ ] Mini neural network training example
- [ ] GPU backend exploration (CUDA)

---

## Why build this?

Modern deep learning frameworks abstract away the mathematics that make them work. Building an autograd engine from scratch forces a precise understanding of:
- How the chain rule generalizes to arbitrary computation graphs
- Why gradient accumulation works the way it does
- What happens under the hood when you call `.backward()` in PyTorch

TensorF is that exercise — done in C++, at the level of raw memory and templates, without hiding behind convenience APIs.

---

## Related Projects

- **[ndmath](https://github.com/InputOutputStream/ndmath)** — Earlier C attempt at the same idea (precursor to TensorF)
- **[lil-math-python](https://github.com/InputOutputStream/lil-math-python)** — Python version exploring the same concepts(precursor to TensorF)
- **[transformer-gpt-numpy](https://github.com/InputOutputStream/transformer-gpt-numpy)** — GPT transformer built with NumPy, using the same from-scratch philosophy (precursor to TensorF)

---

*Part of a broader exploration of the mathematical foundations of modern AI — from tensor calculus to transformer architectures.*