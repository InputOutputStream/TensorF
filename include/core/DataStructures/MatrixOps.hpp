#pragma once
// MatrixOps.hpp — free functions / operators on Matrix<T>. Included at the end of Matrix.hpp
// (include Matrix.hpp, not this file). These are ordinary templates found by ADL.
//
// Scalar parameters are non-deduced (std::type_identity_t<T>), so 2*m, m+1, m+=1 work for Matrix<float>.

#include <ostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "Matrix.hpp"

template <typename E>
std::ostream& operator<<(std::ostream& out, const Matrix<E>& m)
{
    return m.print(out);
}

// ───────────────────────────── scalar arithmetic ─────────────────────────────

template <typename T> Matrix<T> operator*(std::type_identity_t<T> a, const Matrix<T>& rhs) { return Matrix<T>(vecmath::mul_s(rhs.data, a), rhs.shape); }
template <typename T> Matrix<T> operator*(const Matrix<T>& lhs, std::type_identity_t<T> a) { return Matrix<T>(vecmath::mul_s(lhs.data, a), lhs.shape); }
template <typename T> Matrix<T> operator/(const Matrix<T>& lhs, std::type_identity_t<T> a) { return Matrix<T>(vecmath::div_s(lhs.data, a), lhs.shape); }
template <typename T> Matrix<T> operator/(std::type_identity_t<T> a, const Matrix<T>& rhs) { return Matrix<T>(vecmath::s_div(a, rhs.data), rhs.shape); }
template <typename T> Matrix<T> operator+(const Matrix<T>& lhs, std::type_identity_t<T> a) { return Matrix<T>(vecmath::add_s(lhs.data, a), lhs.shape); }
template <typename T> Matrix<T> operator+(std::type_identity_t<T> a, const Matrix<T>& rhs) { return Matrix<T>(vecmath::add_s(rhs.data, a), rhs.shape); }
template <typename T> Matrix<T> operator-(const Matrix<T>& lhs, std::type_identity_t<T> a) { return Matrix<T>(vecmath::sub_s(lhs.data, a), lhs.shape); }
template <typename T> Matrix<T> operator-(std::type_identity_t<T> a, const Matrix<T>& rhs) { return Matrix<T>(vecmath::s_sub(a, rhs.data), rhs.shape); }

template <typename T>
Matrix<T> pow(const Matrix<T>& lhs, std::type_identity_t<T> a) { return lhs.pow(a); }

template <typename T>
Matrix<T> pow(const Matrix<T>& a, const Matrix<T>& b) { return a.pow(b); }

// ───────────────────────────── comparisons (B1, B20) ─────────────────────────────
// All comparisons return Matrix<bool> so where(m > 0, a, b) and Tensor's bool ops work.

namespace mxd {
template <typename T>
Matrix<bool> to_bool_matrix(const std::vector<uint8_t>& v, const shape_t& shape)
{
    return Matrix<bool>(std::vector<bool>(v.begin(), v.end()), shape);
}
} // namespace mxd

template <typename T> Matrix<bool> operator<(std::type_identity_t<T> a, const Matrix<T>& m)  { return mxd::to_bool_matrix<T>(vecmath::cmp_lt(a, m.data), m.shape); }  // a <  m
template <typename T> Matrix<bool> operator<(const Matrix<T>& m, std::type_identity_t<T> a)  { return mxd::to_bool_matrix<T>(vecmath::cmp_lt(m.data, a), m.shape); }  // m <  a
template <typename T> Matrix<bool> operator>(std::type_identity_t<T> a, const Matrix<T>& m)  { return mxd::to_bool_matrix<T>(vecmath::cmp_gt(a, m.data), m.shape); }  // a >  m
template <typename T> Matrix<bool> operator>(const Matrix<T>& m, std::type_identity_t<T> a)  { return mxd::to_bool_matrix<T>(vecmath::cmp_gt(m.data, a), m.shape); }  // m >  a
template <typename T> Matrix<bool> operator<=(std::type_identity_t<T> a, const Matrix<T>& m) { return mxd::to_bool_matrix<T>(vecmath::cmp_le(a, m.data), m.shape); }  // a <= m
template <typename T> Matrix<bool> operator<=(const Matrix<T>& m, std::type_identity_t<T> a) { return mxd::to_bool_matrix<T>(vecmath::cmp_le(m.data, a), m.shape); }  // m <= a
template <typename T> Matrix<bool> operator>=(std::type_identity_t<T> a, const Matrix<T>& m) { return mxd::to_bool_matrix<T>(vecmath::cmp_ge(a, m.data), m.shape); }  // a >= m
template <typename T> Matrix<bool> operator>=(const Matrix<T>& m, std::type_identity_t<T> a) { return mxd::to_bool_matrix<T>(vecmath::cmp_ge(m.data, a), m.shape); }  // m >= a

// ───────────────────────────── compound assignment ─────────────────────────────
// Matrix op= Matrix: same shape -> in place; otherwise rhs is broadcast to lhs.shape (validated).

#define MATRIX_COMPOUND_OP(OP, VFN, SFN)                                                 \
    template <typename T>                                                                \
    Matrix<T>& operator OP(Matrix<T>& lhs, const Matrix<T>& rhs) {                       \
        if (lhs.shape == rhs.shape) { vecmath::VFN(lhs.data, rhs.data); return lhs; }   \
        Matrix<T> rhs_bc = Broadcast<T>::broadcastTo(rhs, lhs.shape);                    \
        vecmath::VFN(lhs.data, rhs_bc.data);                                             \
        return lhs;                                                                      \
    }                                                                                    \
    template <typename T>                                                                \
    Matrix<T>& operator OP(Matrix<T>& lhs, std::type_identity_t<T> cte) {                \
        vecmath::SFN(lhs.data, cte);                                                     \
        return lhs;                                                                      \
    }

MATRIX_COMPOUND_OP(+=, add_inplace, add_inplace_s)
MATRIX_COMPOUND_OP(-=, sub_inplace, sub_inplace_s)
MATRIX_COMPOUND_OP(*=, mul_inplace, mul_inplace_s)
MATRIX_COMPOUND_OP(/=, div_inplace, div_inplace_s)

#undef MATRIX_COMPOUND_OP

// ───────────────────────────── autograd helper ─────────────────────────────

// Reduces a broadcast gradient back to `originalShape` (sums leading dims and dims that were size 1).
template <typename T>
Matrix<T> sumGradForBroadcast(const Matrix<T>& grad, const std::vector<size_t>& originalShape)
{
    Matrix<T> res = grad;
    if (res.shape.size() < originalShape.size())
        throw std::runtime_error("sumGradForBroadcast: gradient rank is lower than the original rank");

    while (res.shape.size() > originalShape.size())
        res = res.sum(0);

    for (int i = static_cast<int>(res.shape.size()) - 1; i >= 0; i--) {
        if (originalShape[i] == 1 && res.shape[i] > 1) {
            res = res.sum(static_cast<size_t>(i));          // removes the axis ...
            shape_t s = res.shape;
            s.insert(s.begin() + i, 1);                      // ... put it back as size 1
            res = Matrix<T>(res.data, s);
        }
    }

    if (res.shape != originalShape)
        res = Matrix<T>(res.data, originalShape);
    return res;
}
