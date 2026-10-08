// matrix_tests.cpp — plain asserts, no framework. Expected values follow numpy semantics.
// Build: g++ -std=c++20 -Wall -Wextra -Wshadow -fsanitize=address,undefined -Iinc matrix_tests.cpp [stub/cblas.o -Istub | -DMATRIX_NO_BLAS]
#include "core/Types/types.hpp"
#include "core/DataStructures/Matrix.hpp"

#include <cmath>
#include <iostream>
#include <sstream>
#include <functional>

// explicit instantiation: every member of Matrix must compile for every supported element type (C9, F)
template class Matrix<float>;
template class Matrix<double>;
template class Matrix<int>;
template class Matrix<bool>;
template class Matrix<fp8_e4m3>;
template class Matrix<fp4_e2m1>;
template class Matrix<float16>;

static int g_fail = 0, g_n = 0;
#define CHECK(c) do { g_n++; if(!(c)) { g_fail++; std::cout << "FAIL " << __FILE__ << ":" << __LINE__ << "  " #c "\n"; } } while(0)
#define CHECK_THROWS(expr) do { g_n++; bool thrown_=false; try { (void)(expr); } catch(const std::exception&) { thrown_=true; } \
    if(!thrown_){ g_fail++; std::cout << "FAIL(no throw) " << __FILE__ << ":" << __LINE__ << "  " #expr "\n"; } } while(0)

using M  = Matrix<float>;
using MD = Matrix<double>;
using V  = std::vector<float>;
using S  = shape_t;

static bool near(float a, float b, float tol = 1e-5f) { return std::fabs(a - b) <= tol * (1.f + std::fabs(b)); }
static bool vnear(const V& a, const V& b, float tol = 1e-5f) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) if (!near(a[i], b[i], tol)) return false;
    return true;
}

// ── A ───────────────────────────────────────────────────────────────────────
static void test_A() {
    M a({{1, 2, 3}, {4, 5, 6}});
    // A1 mean()
    CHECK(near(a.mean().data[0], 3.5f)); CHECK(a.mean().shape == S{1});
    // variance / std of 1..6 : 35/12
    CHECK(near(a.variance().data[0], 35.f / 12.f)); CHECK(near(a.std().data[0], std::sqrt(35.f / 12.f)));
    // A2 pow
    CHECK(vnear(a.pow(2.f).data, {1, 4, 9, 16, 25, 36}));
    CHECK(vnear(pow(a, 2.f).data, {1, 4, 9, 16, 25, 36}));            // free function, scalar non-deduced
    CHECK(vnear(a.pow(M(2.f)).data, {1, 4, 9, 16, 25, 36}));          // Matrix exponent (broadcast scalar)
    CHECK(vnear(M({1, 2, 3}).pow(M({3, 2, 1})).data, {1, 4, 3}));
    CHECK(vnear(M::pow(a, 0.f).data, {1, 1, 1, 1, 1, 1}));
    // A3 ln / log
    MD d({1.0, std::exp(1.0)});
    CHECK(near((float)d.ln().data[1], 1.f));
    Matrix<int> mi({1, 10, 100});
    (void)mi.ln();                                       // double/int instantiation used to be a compile error
    CHECK(std::isfinite(M({0.f}).ln().data[0]));         // clamp
    CHECK(near(M({0.f}).ln().data[0], std::log(1e-9f)));
    CHECK(vnear(M::log(M({0.f, 1.f})).data, {std::log(1e-9f), 0.f}));
    // A4 at(Matrix<bool>)
    Matrix<bool> mask({{true, false, true}, {false, true, false}});
    CHECK(vnear(a.at(mask).data, {1, 0, 3, 0, 5, 0})); CHECK(a.at(mask).shape == (S{2, 3}));
    CHECK_THROWS(a.at(Matrix<bool>({true, false})));
    // A5 Matrix<float>(0) used to be ambiguous
    M z(0); CHECK(z.data.size() == 1 && z.data[0] == 0.f);
    // A6 eye
    CHECK(vnear(M::eye(3).data, {1, 0, 0, 0, 1, 0, 0, 0, 1})); CHECK(M::eye(3).shape == (S{3, 3}));
    CHECK(M::eye(1).data.size() == 1); CHECK(M::eye(0).data.empty());
    CHECK(vnear(M::eye({2}).data, {1, 0, 0, 1}));
    // A7 expand_dims then use strides
    M e = M::expand_dims(a, 0);
    CHECK(e.shape == (S{1, 2, 3})); CHECK(e.strides() == (S{6, 3, 1}));
    CHECK(vnear(e.sum(1).data, {5, 7, 9})); CHECK(e.sum(1).shape == (S{1, 3}));
    CHECK_THROWS(M::expand_dims(a, 5));
}

// ── B ───────────────────────────────────────────────────────────────────────
static void test_B() {
    M a({{1, 2, 3}, {4, 5, 6}});
    // B1 / B20 comparisons are Matrix<bool> with the right orientation
    static_assert(std::is_same_v<decltype(a < 3.f), Matrix<bool>>);
    static_assert(std::is_same_v<decltype(3.f <= a), Matrix<bool>>);
    auto bits = [](const Matrix<bool>& m) { std::vector<int> r; for (size_t i = 0; i < m.data.size(); i++) r.push_back(m.data[i]); return r; };
    using VI = std::vector<int>;
    CHECK(bits(2.f <= a) == (VI{0, 1, 1, 1, 1, 1}));    // numpy: 2 <= a
    CHECK(bits(a >= 3.f) == (VI{0, 0, 1, 1, 1, 1}));
    CHECK(bits(a <= 3.f) == (VI{1, 1, 1, 0, 0, 0}));
    CHECK(bits(3.f >= a) == (VI{1, 1, 1, 0, 0, 0}));
    CHECK(bits(a < 3.f) == (VI{1, 1, 0, 0, 0, 0}));
    CHECK(bits(3.f < a) == (VI{0, 0, 0, 1, 1, 1}));
    CHECK(bits(a > 3.f) == (VI{0, 0, 0, 1, 1, 1}));
    CHECK(bits(3.f > a) == (VI{1, 1, 0, 0, 0, 0}));
    CHECK(bits(a == 2) == (VI{0, 1, 0, 0, 0, 0})); CHECK(bits(a != 2) == (VI{1, 0, 1, 1, 1, 1}));
    CHECK(vnear(M::where(a > 3.f, 1.f, 0.f).data, {0, 0, 0, 1, 1, 1}));
    // B2 maximum
    CHECK(vnear(a.maximum(3.f).data, {3, 3, 3, 4, 5, 6})); CHECK(vnear(M({-1, 2}).maximum(0.f).data, {0, 2}));
    // B3 at
    CHECK(vnear(a.at({1, 2}).data, {6})); CHECK(a.at({1, 2}).shape == S{1});
    CHECK(vnear(a.at({1}).data, {4, 5, 6})); CHECK(a.at({1}).shape == (S{1, 3}));
    M t3 = M::arrange(24.f).reshape({2, 3, 4});
    CHECK(vnear(t3.at({1, 2, 3}).data, {23})); CHECK(vnear(t3.at({1, 2}).data, {20, 21, 22, 23})); CHECK(t3.at({1}).shape == (S{3, 4}));
    CHECK_THROWS(a.at({2, 0})); CHECK_THROWS(a.at({0, 3})); CHECK_THROWS(a.at(shape_t{})); CHECK_THROWS(a.at({0, 0, 0}));
    // B4 row/col
    CHECK(vnear(a.row(1).data, {4, 5, 6})); CHECK(a.row(1).shape == (S{1, 3}));
    CHECK(vnear(a.col(2).data, {3, 6})); CHECK(a.col(2).shape == (S{2, 1}));
    CHECK_THROWS(a.row(2)); CHECK_THROWS(a.col(3)); CHECK_THROWS(t3.row(0)); CHECK_THROWS(t3.col(0));
    try { a.col(3); } catch (const std::exception& e) { CHECK(std::string(e.what()).find("Col") != std::string::npos); }
    // B5 slices
    CHECK(vnear(a.slice_row(1, 2).data, {4, 5, 6})); CHECK(a.slice_row(0, 0).data.empty());
    CHECK_THROWS(a.slice_row(2, 1)); CHECK_THROWS(a.slice_row(1, 3)); CHECK_THROWS(a.slice_cols(2, 1)); CHECK_THROWS(a.slice_cols(0, 4));
    CHECK(vnear(a.slice_cols(1, 3).data, {2, 3, 5, 6})); CHECK(a.slice_cols(1, 3).shape == (S{2, 2}));
    CHECK(vnear(t3.slice_axis(1, 3, 2).data.size() == 12 ? V{1} : V{0}, {1}));       // shape {2,3,2} -> 12 elems
    M sa = t3.slice_axis(1, 3, 2);
    CHECK(sa.shape == (S{2, 3, 2})); CHECK(near(sa.data[0], 1) && near(sa.data[1], 2) && near(sa.data[2], 5) && near(sa.data[11], 22));
    CHECK_THROWS(t3.slice_axis(0, 5, 2)); CHECK_THROWS(t3.slice_axis(2, 1, 0)); CHECK_THROWS(t3.slice_axis(0, 1, 3));
    // B6 sum(axis) on every axis of a rank-4 tensor vs a brute-force reference
    {
        S sh{2, 3, 4, 5};
        M t4 = M::arrange(120.f).reshape(sh);
        for (size_t ax = 0; ax < 4; ax++) {
            M r = t4.sum(ax);
            S exp_shape; for (size_t i = 0; i < 4; i++) if (i != ax) exp_shape.push_back(sh[i]);
            CHECK(r.shape == exp_shape);
            std::vector<double> ref(r.data.size(), 0.0);
            S st = {60, 20, 5, 1};
            for (size_t i0 = 0; i0 < 2; i0++) for (size_t i1 = 0; i1 < 3; i1++) for (size_t i2 = 0; i2 < 4; i2++) for (size_t i3 = 0; i3 < 5; i3++) {
                size_t idx[4] = {i0, i1, i2, i3}, flat = 0, mul = 1;
                for (int d = 3; d >= 0; d--) { if ((size_t)d == ax) continue; flat += idx[d] * mul; mul *= sh[d]; }
                ref[flat] += (double)(i0 * st[0] + i1 * st[1] + i2 * st[2] + i3 * st[3]);
            }
            bool ok = true; for (size_t i = 0; i < ref.size(); i++) ok = ok && near(r.data[i], (float)ref[i]);
            CHECK(ok);
        }
        CHECK_THROWS(t4.sum(4)); CHECK_THROWS(t4.mean(7));
    }
    CHECK(vnear(a.sum(0).data, {5, 7, 9})); CHECK(vnear(a.sum(1).data, {6, 15})); CHECK(near(a.sum(), 21));
    CHECK(vnear(a.mean(0).data, {2.5f, 3.5f, 4.5f})); CHECK(vnear(a.mean(1).data, {2, 5}));
    CHECK(vnear(a.var(1).data, {2.f / 3.f, 2.f / 3.f})); CHECK(vnear(a.var(1, false).data, {1, 1}));
    CHECK(vnear(a.std(0).data, {1.5f, 1.5f, 1.5f}));
    // B7 transpose
    CHECK(vnear(a.transpose().data, {1, 4, 2, 5, 3, 6})); CHECK(a.transpose().shape == (S{3, 2}));
    CHECK(vnear(a.transpose({0, 1}).data, {1, 2, 3, 4, 5, 6}));             // identity perm honored
    CHECK(vnear(a.transpose({1, 0}).data, {1, 4, 2, 5, 3, 6}));
    CHECK_THROWS(a.transpose({0, 0})); CHECK_THROWS(a.transpose({0, 2})); CHECK_THROWS(a.transpose({0}));
    M tp = t3.transpose({2, 0, 1});                                          // shape {4,2,3}
    CHECK(tp.shape == (S{4, 2, 3})); CHECK(near(tp.data[1], t3.data[4])); CHECK(near(tp.at({3, 1, 2}).data[0], 23));
    CHECK(t3.transpose().shape == (S{4, 3, 2})); CHECK(near(t3.transpose().at({3, 2, 1}).data[0], 23));
    CHECK(M({1, 2, 3}).transpose().shape == (S{3, 1}));                      // 1-D -> {n,1} (documented)
    {
        M big = M::arrange(70.f * 45.f).reshape({70, 45}); M bt = big.transpose(); bool ok = true;
        for (size_t i = 0; i < 70; i++) { for (size_t j = 0; j < 45; j++) ok = ok && bt.data[j * 70 + i] == big.data[i * 45 + j]; }
        CHECK(ok);
    }
    // B8 arrange
    CHECK(Matrix<int>::arrange(0, 5, 2).data == (std::vector<int>{0, 2, 4}));
    CHECK(Matrix<int>::arange(5).data == (std::vector<int>{0, 1, 2, 3, 4}));
    CHECK(vnear(M::arrange(0.f, 1.f, 0.25f).data, {0, .25f, .5f, .75f}));
    CHECK(vnear(M::arrange(5.f, 0.f, -2.f).data, {5, 3, 1}));
    CHECK(M::arrange(0.f, -3.f, 1.f).data.empty()); CHECK_THROWS(M::arrange(0.f, 1.f, 0.f));
    // B9 one_hot
    CHECK(vnear(M::one_hot(M({0, 2}), 3).data, {1, 0, 0, 0, 0, 1})); CHECK_THROWS(M::one_hot(M({3}), 3)); CHECK_THROWS(M::one_hot(M({-1}), 3));
    // B10 random in [0,1); B11 one shared generator across element types
    { M r = M::random({1000}); bool ok = true; for (float v : r.data) ok = ok && v >= 0.f && v < 1.f; CHECK(ok); }
    M::manual_seed(123); MD x1 = MD::randu({5}); Matrix<int>::manual_seed(123); MD x2 = MD::randu({5}); CHECK(x1 == x2);
    M::manual_seed(7); M n1 = M::randn({4}); M::manual_seed(7); M n2 = M::randn({4}); CHECK(n1 == n2);
    M::manual_seed(9); M q1 = M::random({4}); Matrix<double>::manual_seed(9); M q2 = M::random({4}); CHECK(q1 == q2);
    CHECK(M::he({4, 3}).shape == (S{4, 3})); CHECK_THROWS(M::he(shape_t{}));
    { Matrix<int> ri = Matrix<int>::randu(0, 5, {200}); bool ok = true; for (int v : ri.data) ok = ok && v >= 0 && v < 5; CHECK(ok); }
    CHECK(M::choice(3, M({0.f, 1.f, 0.f})).data[0] == 1.f);
    // B12 tril/triup
    CHECK(vnear(M::tril(M::ones({3, 2})).data, {1, 0, 1, 1, 1, 1}));         // non-square
    CHECK(vnear(M::triup(M::ones({2, 3})).data, {1, 1, 1, 0, 1, 1}));
    { M b = M::tril(M::ones({2, 2, 2})); CHECK(vnear(b.data, {1, 0, 1, 1, 1, 0, 1, 1})); }
    CHECK(vnear(M::tril(3).data, {1, 0, 0, 1, 1, 0, 1, 1, 1})); CHECK(vnear(M::triu(2).data, {1, 1, 0, 1}));
    CHECK_THROWS(M::tril(M({1, 2, 3})));
}

// ── matmul / dot (B13, B14) ──────────────────────────────────────────────────
static void test_matmul() {
    M a({{1, 2, 3}, {4, 5, 6}}), b({{1, 0}, {0, 1}, {1, 1}});
    M c = a.matmul(b);
    CHECK(c.shape == (S{2, 2})); CHECK(vnear(c.data, {4, 5, 10, 11})); CHECK(c.data.size() == 4);      // not padded
    // trimmed exactly even when total is not a multiple of 8
    M x({1, 2, 3, 4, 5, 6, 7}, {1, 7}); M y = M::ones({7, 1});
    CHECK(x.matmul(y).data.size() == 1); CHECK(near(x.matmul(y).data[0], 28));
    CHECK_THROWS(M::ones({2, 3}).matmul(M::ones({4, 2})));                                        // inner dims
    CHECK_THROWS(M({1, 2, 3}).matmul(M({1, 2, 3})));                                               // 1D@1D
    // 1-D promotion (numpy)
    CHECK(vnear(M({1, 2, 3}).matmul(b).data, {4, 5})); CHECK(M({1, 2, 3}).matmul(b).shape == S{2});
    CHECK(vnear(a.matmul(M({1, 1, 1})).data, {6, 15})); CHECK(a.matmul(M({1, 1, 1})).shape == S{2});
    // batch: [2,2,3] @ [3,2]  and  [2,2,3] @ [2,3,2]  and broadcast [1,2,3] @ [2,3,2]
    M A3 = M::arange(12.f).reshape({2, 2, 3});
    M B2 = M({{1, 0}, {0, 1}, {1, 1}});
    M r1 = A3.matmul(B2); CHECK(r1.shape == (S{2, 2, 2}));
    auto ref = [&](const M& A, size_t ab, const M& B, size_t bb) {      // [2x3]@[3x2] for batch ab / bb
        V out; for (size_t i = 0; i < 2; i++) for (size_t j = 0; j < 2; j++) { float s = 0; for (size_t k = 0; k < 3; k++) s += A.data[ab * 6 + i * 3 + k] * B.data[bb * 6 + k * 2 + j]; out.push_back(s); } return out; };
    { V e0 = ref(A3, 0, M::concat({B2.reshape({1, 3, 2})}, 0), 0), e1 = ref(A3, 1, M::concat({B2.reshape({1, 3, 2})}, 0), 0);
      CHECK(vnear(V(r1.data.begin(), r1.data.begin() + 4), e0)); CHECK(vnear(V(r1.data.begin() + 4, r1.data.end()), e1)); }
    M B3 = M::arange(12.f).reshape({2, 3, 2});
    M r2 = A3.matmul(B3); CHECK(r2.shape == (S{2, 2, 2}));
    CHECK(vnear(V(r2.data.begin(), r2.data.begin() + 4), ref(A3, 0, B3, 0))); CHECK(vnear(V(r2.data.begin() + 4, r2.data.end()), ref(A3, 1, B3, 1)));
    M A1 = M::arange(6.f).reshape({1, 2, 3});
    M r3 = A1.matmul(B3); CHECK(r3.shape == (S{2, 2, 2}));                                   // size-1 batch broadcast
    CHECK(vnear(V(r3.data.begin(), r3.data.begin() + 4), ref(A1, 0, B3, 0))); CHECK(vnear(V(r3.data.begin() + 4, r3.data.end()), ref(A1, 0, B3, 1)));
    CHECK_THROWS(M::ones({2, 2, 3}).matmul(M::ones({3, 3, 2})));                               // batch mismatch
    CHECK(M::ones({0, 3}).matmul(M::ones({3, 2})).shape == (S{0, 2}));
    // dot
    CHECK(near(M({1, 2, 3}).dot(M({4, 5, 6})).data[0], 32)); CHECK_THROWS(M({1, 2}).dot(M({1, 2, 3})));
    CHECK(near(a.dot(a).data[0], 91));                        // [DECISION] 2D.2D = flattened inner product
    CHECK_THROWS(a.dot(M({{1, 2, 3}})));                      // used to read out of bounds
    CHECK(vnear(A3.dot(B2).data, r1.data)); CHECK_THROWS(a.dot(M({1, 2, 3}))); CHECK_THROWS(A3.dot(M::ones({4, 2})));
    // integer matmul (generic path) and double
    Matrix<int> ia({{1, 2}, {3, 4}}); CHECK(ia.matmul(ia).data == (std::vector<int>{7, 10, 15, 22}));
    MD da({{1, 2}, {3, 4}}); CHECK(da.matmul(da).data == (std::vector<double>{7, 10, 15, 22}));
    CHECK_THROWS(Matrix<bool>({true, false}).matmul(Matrix<bool>({{true}, {false}})));
}

// ── constructors / storage / printing (B15, B16, B22, C1-C3) ─────────────────────
static void test_storage() {
    // B15 strict storage: no accepted tail padding
    CHECK_THROWS(M(V(8, 1.f), shape_t{7})); CHECK_THROWS((M(V(8, 1.f), {7}))); CHECK(M(V(7, 1.f), shape_t{7}).data.size() == 7);
    CHECK_THROWS(M({1, 2, 3}, {2, 2}));
    CHECK(M::ones({3, 3}).matmul(M::ones({3, 3})).data.size() == 9);
    // B16 0-D
    M s0 = M({1, 2, 3}).sum(0);
    CHECK(s0.shape.empty()); CHECK(s0.data.size() == 1); CHECK(near(s0.data[0], 6));
    { std::ostringstream os; os << s0; CHECK(os.str().find("6") != std::string::npos); }
    { std::ostringstream os; os << M(); CHECK(os.str().size() > 0); }
    { std::ostringstream os; os << M({{1, 2}, {3, 4}}); CHECK(os.str() == "[\n [1,2,]\n [3,4,]\n]"); }
    { std::ostringstream os; os << M({1, 2}); CHECK(os.str() == " [1,2,]\n"); }
    CHECK(near(M({1, 2, 3}).mean(0).data[0], 2)); CHECK(M({1, 2, 3}).mean(0).shape.empty());
    // B22 empty constructors
    CHECK(M(std::vector<std::vector<float>>{}).shape == (S{0, 0}));
    CHECK(M(std::initializer_list<std::initializer_list<float>>{}).data.empty());
    CHECK(M(std::initializer_list<std::initializer_list<std::initializer_list<float>>>{}).data.empty());
    CHECK_THROWS(M(std::vector<std::vector<float>>{{1, 2}, {3}}));
    CHECK_THROWS(M({{1, 2}, {3}}));
    CHECK_THROWS((M(std::vector<std::vector<float>>{{1, 2}, {3, 4}}, {3, 3})));
    M t = M({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}); CHECK(t.shape == (S{2, 2, 2})); CHECK(near(t.data[7], 8));
    CHECK_THROWS(M({{{1, 2}, {3, 4}}, {{5, 6}, {7}}}));
    // C1 const-correct: everything below is called on a const object
    const M c({{1, 2, 3}, {4, 5, 6}});
    (void)(c + c); (void)(c - c); (void)(c * c); (void)(c / c); (void)(-c); (void)(c == c); (void)c.row(0); (void)c.col(0); (void)c.at({0, 0});
    (void)c.sum(); (void)c.sum(0); (void)c.mean(); (void)c.mean(0); (void)c.variance(); (void)c.var(0); (void)c.std(0); (void)c.transpose(); (void)c.transpose({1, 0});
    (void)c.dot(c); (void)c.matmul(c.transpose()); (void)c.reshape({3, 2}); (void)c.flatten(); (void)c.slice_row(0, 1); (void)c.slice_cols(0, 1); (void)c.slice_axis(0, 1, 0);
    (void)c.sqrt(); (void)c.exponent(); (void)c.ln(); (void)c.cbrt(); (void)c.pow(2.f); (void)c.maximum(0.f); (void)c.get_data(); (void)c.get_size(); (void)c.get_ndims();
    { std::ostringstream os; c.print(os); os << c; }
    CHECK(vnear((c + c).data, {2, 4, 6, 8, 10, 12})); CHECK(vnear((c / c).data, {1, 1, 1, 1, 1, 1})); CHECK(c == c);
    // C2 rule of five: move keeps data, copy keeps gpu flag
    M m1 = M::ones({2, 2}); m1.gpu = true; M m2 = m1; CHECK(m2.gpu); M m3 = std::move(m1); CHECK(m3.shape == (S{2, 2}) && m3.gpu);
    M m4; m4 = m3; CHECK(m4 == m3); M m5; m5.copy_from(m3); CHECK(m5 == m3); M m6; m6.copy_from(&m3); CHECK(m6 == m3);
    M m7; const M cm = m3; m7.copy_from(cm); CHECK(m7 == m3); CHECK_THROWS(m7.copy_from((M*)nullptr));
    // C3 no stale cache after editing public members
    M e = M::ones({2, 3}); e.shape = {3, 2}; CHECK(e.strides() == (S{2, 1})); CHECK(e.get_ndims() == 2); e.data.push_back(1); e.data.push_back(1); e.shape = {2, 4};
    CHECK(e.get_size() == 8); M ez = M::ones({2, 2}); ez.zeros(); CHECK(vnear(ez.data, {0, 0, 0, 0})); ez.ones(); CHECK(vnear(ez.data, {1, 1, 1, 1}));
    M cl = M::ones({2}); cl.clear(); CHECK(cl.get_size() == 0 && cl.get_ndims() == 0);

    std::vector<float> out; M().flattenRecursive(std::vector<std::vector<float>>{{1, 2}, {3}}, out); CHECK(out.size() == 3);
    M gm = M::from(std::vector<float>{1, 2}); CHECK(gm.shape == S{2}); CHECK(M::from(3.f).data[0] == 3.f); CHECK(M::from(2).data[0] == 2.f);
    CHECK(M::from(M::ones({2})).shape == S{2}); CHECK(M::from({1.f, 2.f, 3.f}).shape == S{3});
    // pointer ctor is still there but explicit
    M pc(&m3); CHECK(pc == m3);
}

// ── broadcasting (B17, D1) and operators (B19, C5) ─────────────────────────────
static void test_broadcast() {
    M a({{1, 2, 3}, {4, 5, 6}});
    CHECK(vnear((a + M({10, 20, 30})).data, {11, 22, 33, 14, 25, 36}));
    CHECK(vnear((a * M({{2}, {3}})).data, {2, 4, 6, 12, 15, 18}));
    CHECK(((M({{1}, {2}}) + M({{10, 20, 30}})).shape == (S{2, 3})));
    CHECK(vnear((M({{1}, {2}}) + M({{10, 20, 30}})).data, {11, 21, 31, 12, 22, 32}));
    CHECK(vnear((a - M(1.f)).data, {0, 1, 2, 3, 4, 5})); CHECK(vnear((M({6, 6, 6}) / M({{1}, {2}})).data, {6, 6, 6, 3, 3, 3}));
    CHECK_THROWS(a + M({1, 2})); CHECK_THROWS(a + M::ones({3, 3}));
    CHECK(Broadcast<float>::broadcastTo(M({1, 2, 3}), {2, 3}).shape == (S{2, 3}));
    CHECK(Broadcast<float>::broadcastTo(M({{1}, {2}}), {2, 3}).data == (V{1, 1, 1, 2, 2, 2}));
    CHECK_THROWS(Broadcast<float>::broadcastTo(a, {3}));                      // lower rank target
    CHECK_THROWS(Broadcast<float>::broadcastTo(a, {2, 4}));                   // incompatible dim
    CHECK_THROWS(Broadcast<float>::broadcastTo(a, {3, 3}));
    { Broadcast<float> inst; CHECK(inst.broadcastTo(M({1}), {2, 2}).data == (V{1, 1, 1, 1})); }   // instance call still works
    { auto p = Broadcast<float>::broadcast(M({{1}, {2}}), M({1, 2, 3})); CHECK(p.first.shape == (S{2, 3}) && p.second.shape == (S{2, 3})); }
    CHECK(Broadcast<float>::computeBroadcastResultShape(M::ones({0, 3}), M::ones({1, 3})) == (S{0, 3}));      // max(0,1) = 0
    CHECK(Broadcast<float>::computeBroadcastResultShape(M::ones({1, 3}), M::ones({0, 3})) == (S{0, 3}));
    CHECK(Broadcast<float>::assertBroadcast(M::ones({2, 1}), M::ones({1, 3}))); CHECK(!Broadcast<float>::assertBroadcast(M::ones({2}), M::ones({3})));
    // compound ops
    M z = M::ones({2, 2});
    M& r = (z += 1.f); CHECK(&r == &z); CHECK(r.shape == (S{2, 2})); CHECK(vnear(z.data, {2, 2, 2, 2}));
    (z *= 3.f); (z -= 1.f); (z /= 5.f); CHECK(vnear(z.data, {1, 1, 1, 1}));
    z += M({1, 2}); CHECK(vnear(z.data, {2, 3, 2, 3}));                       // broadcast rhs to lhs
    z -= M({{1}, {1}}); CHECK(vnear(z.data, {1, 2, 1, 2})); z *= M({2, 2}); CHECK(vnear(z.data, {2, 4, 2, 4})); z /= M({2, 2}); CHECK(vnear(z.data, {1, 2, 1, 2}));
    CHECK_THROWS(z += M({1, 2, 3})); CHECK_THROWS(z += M::ones({3, 2}));
    // C5: scalar operators with a scalar of a different arithmetic type, both sides
    CHECK(vnear((2 * M({1, 2})).data, {2, 4})); CHECK(vnear((M({1, 2}) + 1).data, {2, 3})); CHECK(vnear((1 - M({1, 2})).data, {0, -1}));
    CHECK(vnear((6 / M({1, 2})).data, {6, 3})); CHECK(vnear((M({1, 2}) * 2.5).data, {2.5f, 5})); CHECK(vnear((M({4, 2}) / 2).data, {2, 1}));
    z += 1; CHECK(vnear(z.data, {2, 3, 2, 3}));
    CHECK_THROWS(M({1, 2}) / M({0, 1})); CHECK_THROWS(M({1, 2}) / 0.f);
    // sumGradForBroadcast
    M g = M::ones({2, 3});
    CHECK(vnear(sumGradForBroadcast(g, {3}).data, {2, 2, 2})); CHECK(sumGradForBroadcast(g, {3}).shape == S{3});
    CHECK(vnear(sumGradForBroadcast(g, {2, 1}).data, {3, 3})); CHECK(sumGradForBroadcast(g, {2, 1}).shape == (S{2, 1}));
    CHECK(vnear(sumGradForBroadcast(g, {1, 3}).data, {2, 2, 2})); CHECK(sumGradForBroadcast(g, {1, 3}).shape == (S{1, 3}));
    CHECK(sumGradForBroadcast(g, {2, 3}).shape == (S{2, 3})); CHECK(vnear(sumGradForBroadcast(g, {1}).data, {6}));
    CHECK_THROWS(sumGradForBroadcast(M({1, 2}), {2, 2}));
}

// ── stack / concat / misc (B18) ─────────────────────────────────────────────────
static void test_stack() {
    M a({{1, 2}, {3, 4}}), b({{5, 6}, {7, 8}});
    CHECK(vnear(M::concat({a, b}, 0).data, {1, 2, 3, 4, 5, 6, 7, 8})); CHECK(M::concat({a, b}, 0).shape == (S{4, 2}));
    CHECK(vnear(M::concat({a, b}, 1).data, {1, 2, 5, 6, 3, 4, 7, 8})); CHECK(M::concat({a, b}, 1).shape == (S{2, 4}));
    M c = M::concat({M::arange(8.f).reshape({2, 2, 2}), M::arange(4.f).reshape({2, 1, 2})}, 1);
    CHECK(c.shape == (S{2, 3, 2})); CHECK(vnear(c.data, {0, 1, 2, 3, 0, 1, 4, 5, 6, 7, 2, 3}));
    CHECK_THROWS(M::concat(std::vector<M>{}, 0)); CHECK_THROWS(M::concat({a, b}, 2)); CHECK_THROWS(M::concat({a, M::ones({3, 3})}, 0));
    CHECK_THROWS(M::concat({a, M::ones({2, 2, 2})}, 0)); CHECK(M::concat(std::initializer_list<M>{}, 0).data.empty());
    CHECK(vnear(M::stack({a, b}, 0).data, {1, 2, 3, 4, 5, 6, 7, 8})); CHECK(M::stack({a, b}, 0).shape == (S{4, 2}));
    CHECK(vnear(M::stack({a, b}, 1).data, {1, 2, 5, 6, 3, 4, 7, 8})); CHECK(M::stack({a, b}, 1).shape == (S{2, 4}));
    CHECK(vnear(M::stack({a, b}, 2).data, {1, 5, 2, 6, 3, 7, 4, 8})); CHECK(M::stack({a, b}, 2).shape == (S{2, 2, 2}));
    CHECK_THROWS(M::stack(std::vector<M>{}, 0)); CHECK_THROWS(M::stack({a, b}, 3)); CHECK_THROWS(M::stack({a, M::ones({3, 2})}, 0));
    CHECK_THROWS(M::stack({M({1, 2}), M({3, 4})}, 0));
    // elemsAt (embedding)
    M table({{1, 2}, {3, 4}, {5, 6}});
    CHECK(vnear(table.elemsAt(M({2, 0})).data, {5, 6, 1, 2})); CHECK(table.elemsAt(M({2, 0})).shape == (S{2, 2}));
    CHECK_THROWS(table.elemsAt(M({3}))); CHECK_THROWS(table.elemsAt(M({-1})));
    // reshape / misc
    CHECK_THROWS(a.reshape({3, 3})); CHECK(a.reshape({4}).shape == S{4}); CHECK(M::ravel(a).shape == S{4});
    CHECK(M::any(M({0, 0, 1}))); CHECK(!M::any(M({0, 0}))); CHECK(M::any(M({1, 5}), [](float v) { return v > 4; }));
    CHECK(!M::hasNaN(a)); CHECK(M::hasNaN(M({1.f, M::nan()}))); CHECK(std::isinf(M::inf()));
    CHECK(vnear(M::zeros({2}).data, {0, 0})); CHECK(vnear(M::ones({2}).data, {1, 1}));
    CHECK(vnear(M::sin(M({0.f})).data, {0}) && vnear(M::cos(M({0.f})).data, {1}) && vnear(M::tan(M({0.f})).data, {0}));
    CHECK(vnear(M({4, 9}).sqrt().data, {2, 3})); CHECK(vnear(M({8, 27}).cbrt().data, {2, 3})); CHECK(vnear(M({0.f}).exponent().data, {1}));
}

// ── element types other than float/double (F, C9) ─────────────────────────────
static void test_types() {
    // bool
    Matrix<bool> mb({{true, false}, {false, true}});
    CHECK(Matrix<float>::where(mb, 1.f, 2.f).data == (V{1, 2, 2, 1})); CHECK(Matrix<bool>::any(mb)); CHECK(mb.transpose().data == mb.data);
    CHECK((mb == true).data[0]); CHECK(mb.sum() == true); CHECK(mb.sum(0).data.size() == 2); CHECK(mb.reshape({4}).shape == S{4});
    { std::ostringstream os; os << mb; CHECK(os.str() == "[\n [1,0,]\n [0,1,]\n]"); }
    // int
    Matrix<int> mi({{1, 2}, {3, 4}});
    CHECK(mi.sum() == 10); CHECK(mi.sum(0).data == (std::vector<int>{4, 6})); CHECK(mi.mean(1).data == (std::vector<int>{1, 3}));
    CHECK(mi.maximum(2).data == (std::vector<int>{2, 2, 3, 4})); CHECK((mi * 2).data == (std::vector<int>{2, 4, 6, 8}));
    CHECK(Matrix<int>::eye(2).data == (std::vector<int>{1, 0, 0, 1})); CHECK(Matrix<int>::arrange(3).data == (std::vector<int>{0, 1, 2}));
    Matrix<int>::manual_seed(1); (void)Matrix<int>::randn({2}); (void)Matrix<int>::random({2}); (void)Matrix<int>::he({2, 2});
    // FP8 / FP4 / float16: results within quantisation error of the float result
    auto near_q = [](float got, float want, float rel) { return std::fabs(got - want) <= rel * std::fabs(want) + 1e-6f; };
    {
        using Q = fp8_e4m3;
        Matrix<Q> a({Q(1.f), Q(2.f), Q(3.f), Q(4.f)}, {2, 2});
        CHECK(near_q(float(a.sum()), 10.f, 0.07f));
        CHECK(near_q(float(a.mean().data[0]), 2.5f, 0.07f));
        Matrix<Q> mm = a.matmul(a);                                           // [[7,10],[15,22]] quantised
        CHECK(near_q(float(mm.data[0]), 7.f, 0.07f) && near_q(float(mm.data[3]), 22.f, 0.07f) && mm.shape == (S{2, 2}));
        CHECK(near_q(float(a.sqrt().data[3]), 2.f, 0.07f)); CHECK(near_q(float(a.ln().data[3]), std::log(4.f), 0.07f));
        CHECK(near_q(float(a.exponent().data[0]), std::exp(1.f), 0.07f));
        CHECK(!Matrix<Q>::hasNaN(a)); CHECK(Matrix<Q>::hasNaN(Matrix<Q>({std::numeric_limits<Q>::quiet_NaN()})));
        CHECK(float(Matrix<Q>::inf()) > 1e30f);
        Matrix<bool> gt = a > Q(2.f); CHECK(!gt.data[0] && !gt.data[1] && gt.data[2] && gt.data[3]);
        CHECK(float((a + a).data[3]) == 8.f); CHECK(float((-a).data[0]) == -1.f);
        Matrix<Q> ones = Matrix<Q>::ones({100});                               // accumulate in float, not in FP8: 100 is representable-ish
        CHECK(near_q(float(ones.sum()), 100.f, 0.07f));
        Matrix<Q>::manual_seed(3); Matrix<Q> rn = Matrix<Q>::randn({50}); CHECK(rn.data.size() == 50);
        Matrix<Q> ru = Matrix<Q>::random({50}); bool ok = true; for (auto v : ru.data) ok = ok && float(v) >= 0.f && float(v) <= 1.f; CHECK(ok);
        Matrix<Q> dd = a.dot(a); CHECK(near_q(float(dd.data[0]), 30.f, 0.07f));
        CHECK(near_q(float(a.var(0).data[0]), 1.f, 0.07f));
        { std::ostringstream os; os << a; CHECK(os.str().find("[1,2,]") != std::string::npos); }
    }
    {
        using Q = fp4_e2m1;
        Matrix<Q> a({Q(1.f), Q(2.f), Q(1.f), Q(1.5f)}, {2, 2});
        CHECK(float(a.sum()) == 3.f);   /* 5.5 accumulated in float, then saturates at fp4_e2m1 max (3.0) */ CHECK(float(a.matmul(a).data[0]) > 0.f);
        CHECK(float(std::numeric_limits<Q>::max()) == 3.f);
        CHECK(float(Matrix<Q>::inf()) > 1e30f);
    }
    {
        using Q = float16;
        Matrix<Q> a({Q(1), Q(2), Q(3), Q(4)}, {2, 2});
        CHECK(near_q(float(a.sum()), 10.f, 1e-3f)); CHECK(near_q(float(a.matmul(a).data[3]), 22.f, 1e-3f));
        CHECK(near_q(float(a.sqrt().data[3]), 2.f, 1e-3f)); CHECK(!Matrix<Q>::hasNaN(a));
        Matrix<Q>::manual_seed(4); CHECK(Matrix<Q>::randn({4}).data.size() == 4);
        { std::ostringstream os; os << a; CHECK(os.str().find("[1,2,]") != std::string::npos); }
    }
}

int main() {
    test_A(); test_B(); test_matmul(); test_storage(); test_broadcast(); test_stack(); test_types();
    std::cout << (g_n - g_fail) << "/" << g_n << " checks passed\n";
    return g_fail ? 1 : 0;
}
