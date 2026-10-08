// Every Matrix call form that Tensor.hpp / tensor_overloads.hpp use (Operations/*.hpp are not available here).
#include "core/Types/types.hpp"
#include "core/DataStructures/Matrix.hpp"
#include "core/DataStructures/Graph.hpp"
#include <memory>
#include <cassert>
#include <sstream>
template <typename T> void tensor_like() {
    const T a = T(2);
    Matrix<T> v1({a});                                     // make_tensor(const T a)
    Matrix<T> v2(std::vector<T>{a, a});                    // make_tensor(std::vector<T>)
    Matrix<T> v3(std::vector<std::vector<T>>{{a}, {a}});   // make_tensor(vector<vector>)
    Matrix<T> v4(std::initializer_list<T>{a, a});          // make_tensor(init_list)
    Matrix<T> v5(std::initializer_list<T>{a, a}, {1, 2});  // make_tensor(init_list, shape)
    Matrix<T> v6(std::vector<T>{a, a}, {2, 1});            // make_tensor(vector, init_list shape)
    Matrix<T> v7(std::vector<T>{a, a}, shape_t{2});        // make_tensor(vector, shape_t)
    Matrix<T> v8({{a, a}, {a, a}});                        // make_tensor(init_list<init_list>)
    Matrix<T> v9({{{a, a}}, {{a, a}}});                    // 3-D
    Matrix<T> val; Matrix<T> grad;
    val.copy_from(v8); val.copy_from(&v8); Matrix<T>& r = v8; val.copy_from(r);                 // Tensor ctors
    size_t nd = val.get_ndims(); (void)nd; shape_t sh = val.shape; (void)sh;
    grad = Matrix<T>(); grad.copy_from(v8);
    if (grad.get_size() > 0) { assert(grad.shape == v8.shape); grad += v8; }                       // Tensor::backward
    bool nanp = Matrix<T>::hasNaN(grad); (void)nanp;
    std::ostringstream os; os << grad;
    grad.clear();
    shape_t gs = Matrix<T>::getShape({2, 3}); (void)gs;
    (void)Matrix<T>::zeros({2, 2}); (void)Matrix<T>::ones({2, 2}); (void)Matrix<T>::randomn({2, 2});
    (void)Matrix<T>::random({2, 2}); (void)Matrix<T>::eye({2});
    (void)val.maximum(0);                                                                          // Tensor::maximum(int)
    (void)val.at(shape_t{1, 1});                                                                   // Tensor::at(init_list)
    Matrix<bool> idx(std::vector<bool>(4, true), {2, 2}); (void)val.at(idx);                       // Tensor::at(Tensor_t<bool>) after J2
    (void)Matrix<T>::from(std::vector<T>{a}); (void)Matrix<T>::from(a); (void)Matrix<T>::from(v8);  // Tensor::from after J1
    Matrix<bool> b1 = val != a; Matrix<bool> b2 = val == a;                                        // tensor_overloads ==/!= with scalar
    Matrix<bool> b3 = a < val; Matrix<bool> b4 = val < a; Matrix<bool> b5 = a >= val; Matrix<bool> b6 = val <= a;   // make_tensor<bool>(a < val)
    (void)b1; (void)b2; (void)b3; (void)b4; (void)b5; (void)b6;
    Matrix<T> sc = val * T(2) + T(1) - T(3); (void)sc;
}
int main() {
    tensor_like<float>(); tensor_like<double>();
    // Graph.hpp (J6): float, double and FP8 features
    for (int round = 0; round < 3; round++) {
        Graph<float> g(2); for (int i = 0; i < 6; i++) g.add_node(std::vector<float>{(float)(i < 3 ? 1 : -1), (float)(i % 3 + 1)}, i < 3 ? 0 : 1);
        size_t e = g.connect_by_similarity(0.9f, 2); g.build(); assert(e == 4 && g.adjacency().n == 6 && g.adjacency().symmetric);
    }
    { Graph<double> g(2); g.add_node(std::vector<double>{1, 0}); g.add_node(std::vector<double>{1, 0.001}); assert(g.connect_by_similarity(0.99) == 1); g.build(); (void)g.dense_adjacency(); }
    { Graph<fp8_e4m3> g(2); g.add_node(std::vector<fp8_e4m3>{fp8_e4m3(1.f), fp8_e4m3(0.f)}); g.add_node(std::vector<fp8_e4m3>{fp8_e4m3(1.f), fp8_e4m3(0.f)});
      assert(g.connect_by_similarity(fp8_e4m3(0.9f)) == 1); }
    std::puts("contract OK");
}
