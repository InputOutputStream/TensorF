#include "core/Types/types.hpp"
#ifdef ORIG
#include "core/DataStructures/Matrix.orig.hpp"
#else
#include "core/DataStructures/Matrix.hpp"
#endif
#include <cstdio>
using M = Matrix<float>;
static unsigned long long st = 88172645463325252ULL;
static float rnd(){ st ^= st<<13; st ^= st>>7; st ^= st<<17; return (float)((st>>11)%2001)/100.f - 10.f; }
static M gen(shape_t s){ size_t n=1; for(auto d:s) n*=d; std::vector<float> v(n); for(auto&x:v) x=rnd(); return M(v,s); }
static void pr(const char* name, const M& m){ std::printf("%s shape(",name); for(auto d:m.shape) std::printf("%zu,",d); std::printf(") n=%zu :",m.data.size()); for(size_t i=0;i<m.data.size()&&i<4000;i++) std::printf(" %.6g",m.data[i]); std::printf("\n"); }
int main(){
  M a=gen({5,7}), b=gen({7,3}), c=gen({5,7}), r=gen({7}), col=gen({5,1}), t3=gen({3,4,5}), t3b=gen({3,5,2}), t4=gen({2,3,4,5});
  pr("add",a+c); pr("sub",a-c); pr("mul",a*c); pr("neg",-a);
  pr("add_bcast_row",a+r); pr("mul_bcast_col",a*col); pr("sub_col_row",col-r); pr("add3_2d",t3+gen({4,5}));
  pr("div",a/(c+M(100.f)));
  pr("matmul2d",a.matmul(b)); pr("matmul3d_2d",t3.matmul(gen({5,6}))); pr("matmul3d_3d",t3.matmul(t3b)); pr("dot_batched",t3.dot(t3b));
  pr("dot2d_flat",a.dot(c)); pr("dot1d",r.dot(r));
  pr("T2",a.transpose()); pr("T3",t3.transpose()); pr("T3perm",t3.transpose({1,2,0})); pr("T4perm",t4.transpose({3,1,0,2}));
  pr("sum2_ax0",a.sum(0)); pr("sum2_ax1",a.sum(1)); pr("sum3_ax2",t3.sum(2)); pr("sum3_ax0",t3.sum(0)); pr("sum4_ax3",t4.sum(3)); pr("sum4_ax0",t4.sum(0));
  pr("mean2_ax0",a.mean(0)); pr("mean2_ax1",a.mean(1)); pr("var_ax1",a.var(1)); pr("std_ax0",a.std(0));
  pr("sqrt",a.maximum(0.f).sqrt()); pr("exp",c.exponent()); pr("cbrt",a.cbrt()); pr("ln",(a*a+M(1.f)).ln());
  pr("slice_row",a.slice_row(1,4)); pr("slice_cols",a.slice_cols(2,6)); pr("slice_axis",t3.slice_axis(1,3,2)); pr("row",a.row(2)); pr("col",a.col(3));
  pr("concat0",M::concat({a,c},0)); pr("concat1",M::concat({a,c},1)); pr("concat_3d",M::concat({t3,t3},1));
  pr("stack0",M::stack({a,c},0)); pr("stack1",M::stack({a,c},1)); pr("stack2",M::stack({a,c},2));
  pr("reshape",a.reshape({7,5})); pr("flatten",a.flatten()); pr("tril3",M::tril(3)); pr("triup4",M::triup(4)); pr("tril_sq",M::tril(gen({4,4}))); pr("triup_sq",M::triup(gen({4,4})));
  pr("zeros",M::zeros({2,3})); pr("ones",M::ones({2,3})); pr("maximum0",a.maximum(0.f)); pr("elemsAt",gen({6,3}).elemsAt(M({4,0,5,5},{2,2})));
  pr("onehot",M::one_hot(M({0,2,1}),3)); pr("pow2",a.pow(2.f)); pr("pow3",a.pow(3.f)); 
#ifndef ORIG
  pr("where_mask",a.at(Matrix<bool>(std::vector<bool>(35,true),{5,7})));
#endif

  M z=a; z+=c; pr("iadd",z); z-=c; pr("isub",z); z*=c; pr("imul",z); z+=r; pr("iadd_bc",z);
  pr("scalar_mul",a*2.0f); pr("scalar_rdiv",(c+M(100.f))*0.5f); pr("scalar_add",3.0f+a); pr("scalar_sub",a-1.5f);
  pr("expand",M::expand_dims(a,1)); pr("slice1",gen({5}).row(0));
  pr("sumGrad",sumGradForBroadcast(gen({4,5,3}), {5,3})); pr("sumGrad2",sumGradForBroadcast(gen({4,5,3}), {1,5,1}));
}
