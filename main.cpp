#include "Types/types.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor.hpp"
#include "Modules/Linear.hpp"
#include "Modules/Optimizer.hpp"
#include "Modules/Relu.hpp"

#include <iostream>
#include <vector>
#include <cassert>
using namespace std;


// TODO intergrate the gtest framework to the tsting chain

// ─── helpers ────────────────────────────────────────────────────────────────

void PASS(const string& name) { cout << "[PASS] " << name << "\n"; }
void FAIL(const string& name) { cout << "[FAIL] " << name << "\n"; }

// ─── 1D tests ───────────────────────────────────────────────────────────────

void test_1D_dot()
{
    // numpy: np.dot([1,2,3], [4,5,6]) = 32
    Matrix<float> a({1,2,3});
    Matrix<float> b({4,5,6});
    Matrix<float> res = a.dot(b);
    (res.data[0] == 32) ? PASS("1D dot") : FAIL("1D dot");
}

void test_1D_sum()
{
    // numpy: np.sum([1,2,3,4]) = 10
    Matrix<float> a({1,2,3,4});
    (a.sum() == 10) ? PASS("1D sum") : FAIL("1D sum");
}

void test_1D_arithmetic()
{
    // numpy: [1,2,3] + [4,5,6] = [5,7,9]
    Matrix<float> a({1,2,3});
    Matrix<float> b({4,5,6});
    Matrix<float> res = a + b;
    bool ok = res.data[0]==5 && res.data[1]==7 && res.data[2]==9;
    ok ? PASS("1D add") : FAIL("1D add");

    // numpy: [4,6,9] - [1,2,3] = [3,4,6]
    Matrix<float> c({4,6,9});
    Matrix<float> d({1,2,3});
    Matrix<float> res2 = c - d;
    bool ok2 = res2.data[0]==3 && res2.data[1]==4 && res2.data[2]==6;
    ok2 ? PASS("1D sub") : FAIL("1D sub");
}

// ─── 2D tests ───────────────────────────────────────────────────────────────

void test_2D_matmul()
{
    // numpy: [[1,2],[3,4]] @ [[1,2],[3,4]] = [[7,10],[15,22]]
    Matrix<float> a({{1,2},{3,4}});
    Matrix<float> res = a.matmul(a);
    bool ok = res.data[0]==7  && res.data[1]==10
           && res.data[2]==15 && res.data[3]==22;
    ok ? PASS("2D matmul square") : FAIL("2D matmul square");

    // numpy: [[1,2],[3,4],[5,6]] @ [[1,2,3],[4,5,6]]
    // = [[9,12,15],[19,26,33],[29,40,51]]   shape {3,3}
    Matrix<float> b({{1,2},{3,4},{5,6}});
    Matrix<float> c({{1,2,3},{4,5,6}});
    Matrix<float> res2 = b.matmul(c);
    bool ok2 = res2.data[0]==9  && res2.data[1]==12 && res2.data[2]==15
            && res2.data[3]==19 && res2.data[4]==26 && res2.data[5]==33
            && res2.data[6]==29 && res2.data[7]==40 && res2.data[8]==51;
    ok2 ? PASS("2D matmul rect") : FAIL("2D matmul rect");
}

void test_2D_transpose()
{
    // numpy: [[1,2,3,4],[5,6,7,8],[9,10,11,12]].T
    // = [[1,5,9],[2,6,10],[3,7,11],[4,8,12]]   shape {4,3}
    Matrix<float> m({{1,2,3,4},{5,6,7,8},{9,10,11,12}});
    Matrix<float> t = m.transpose({4,3});
    bool ok = t.data[0]==1 && t.data[1]==5  && t.data[2]==9
           && t.data[3]==2 && t.data[4]==6  && t.data[5]==10
           && t.data[6]==3 && t.data[7]==7  && t.data[8]==11
           && t.data[9]==4 && t.data[10]==8 && t.data[11]==12;
    ok ? PASS("2D transpose") : FAIL("2D transpose");
}

void test_2D_sum_axis0()
{
    // numpy: np.sum([[1,2,3],[4,5,6]], axis=0) = [5,7,9]
    Matrix<float> m({{1,2,3},{4,5,6}});
    Matrix<float> res = m.sum(0);
    bool ok = res.data[0]==5 && res.data[1]==7 && res.data[2]==9;
    ok ? PASS("2D sum axis=0") : FAIL("2D sum axis=0");
}

void test_2D_sum_axis1()
{
    // numpy: np.sum([[1,2,3],[4,5,6]], axis=1) = [6,15]
    Matrix<float> m({{1,2,3},{4,5,6}});
    Matrix<float> res = m.sum(1);
    bool ok = res.data[0]==6 && res.data[1]==15;
    ok ? PASS("2D sum axis=1") : FAIL("2D sum axis=1");
}

// ─── 3D tests ───────────────────────────────────────────────────────────────

void test_3D_matmul()
{
    // numpy:
    // a = np.arange(16).reshape(2,2,4)  -> [[[0..3],[4..7]],[[8..11],[12..15]]]
    // b = np.arange(16).reshape(2,4,2)  -> [[[0,1],[2,3],[4,5],[6,7]],[[8,9],[10,11],[12,13],[14,15]]]
    // np.matmul(a,b)[0,1,1] = 98
    // np.matmul(a,b).shape = (2,2,2)
    Matrix<float> a({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}, {2,2,4});
    Matrix<float> b({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}, {2,4,2});
    Matrix<float> res = a.matmul(b);
    // shape check
    bool shapeOk = res.shape[0]==2 && res.shape[1]==2 && res.shape[2]==2;
    // res[0,1,1] = sum(a[0,1,:] * b[0,:,1]) = 4*1+5*3+6*5+7*7 = 4+15+30+49 = 98
    bool valOk = res.data[3] == 98; // flat index of [0,1,1] in shape {2,2,2}
    (shapeOk && valOk) ? PASS("3D matmul") : FAIL("3D matmul");
}

void test_3D_sum_axis0()
{
    // numpy: 
    // a = [[[1,2],[3,4]],[[5,6],[7,8]]]  shape {2,2,2}
    // np.sum(a, axis=0) = [[6,8],[10,12]]  shape {2,2}
    Matrix<float> a({1,2,3,4,5,6,7,8}, {2,2,2});
    Matrix<float> res = a.sum(0);
    // std::cout << "test_3D_sum_axis0" << res <<"\n";
    bool shapeOk = res.shape[0]==2 && res.shape[1]==2;
    bool valOk = res.data[0]==6 && res.data[1]==8
              && res.data[2]==10 && res.data[3]==12;
    (shapeOk && valOk) ? PASS("3D sum axis=0") : FAIL("3D sum axis=0");
}

void test_3D_sum_axis1()
{
    // numpy:
    // a = [[[1,2],[3,4]],[[5,6],[7,8]]]  shape {2,2,2}
    // np.sum(a, axis=1) = [[4,6],[12,14]]  shape {2,2}
    Matrix<float> a({1,2,3,4,5,6,7,8}, {2,2,2});
    Matrix<float> res = a.sum(1);
    bool shapeOk = res.shape[0]==2 && res.shape[1]==2;
    bool valOk = res.data[0]==4  && res.data[1]==6
              && res.data[2]==12 && res.data[3]==14;
    (shapeOk && valOk) ? PASS("3D sum axis=1") : FAIL("3D sum axis=1");
}

void test_3D_sum_axis2()
{
    // numpy:
    // a = [[[1,2],[3,4]],[[5,6],[7,8]]]  shape {2,2,2}
    // np.sum(a, axis=2) = [[3,7],[11,15]]  shape {2,2}
    Matrix<float> a({1,2,3,4,5,6,7,8}, {2,2,2});
    Matrix<float> res = a.sum(2);
    bool shapeOk = res.shape[0]==2 && res.shape[1]==2;
    bool valOk = res.data[0]==3  && res.data[1]==7
              && res.data[2]==11 && res.data[3]==15;
    (shapeOk && valOk) ? PASS("3D sum axis=2") : FAIL("3D sum axis=2");
}

void test_3D_transpose()
{
    // numpy:
    // a = [[[1,2],[3,4]],[[5,6],[7,8]]]  shape {2,2,2}
    // np.transpose(a) shape = {2,2,2}
    // result[0,0,0]=1, [0,0,1]=5, [0,1,0]=3, [0,1,1]=7
    //        [1,0,0]=2, [1,0,1]=6, [1,1,0]=4, [1,1,1]=8
    Matrix<float> a({1,2,3,4,5,6,7,8}, {2,2,2});
    Matrix<float> res = a.transpose({2,2,2});
    bool ok = res.data[0]==1 && res.data[1]==5
           && res.data[2]==3 && res.data[3]==7
           && res.data[4]==2 && res.data[5]==6
           && res.data[6]==4 && res.data[7]==8;
    ok ? PASS("3D transpose") : FAIL("3D transpose");
}



// ─── 4D tests ───────────────────────────────────────────────────────────────

void test_4D_matmul()
{
    // numpy:
    // a = np.arange(48).reshape(2,3,2,4)
    // b = np.arange(60).reshape(2,3,4,5)
    // np.matmul(a,b).shape = (2,3,2,5)
    // np.matmul(a,b)[0,0,0,0] = 0*0+1*5+2*10+3*15 = 0+5+20+45 = 70
    // np.matmul(a,b)[1,2,1,4] = ?
    // a[1,2,1,:] = [44,45,46,47]
    // b[1,2,:,4] = [44*... let numpy compute: result[1,2,1,4] = 44*4+45*9+46*14+47*19 ... 
    // easier: just check shape and [0,0,0,0]

    std::vector<float> adata, bdata;
    for(int i=0; i<48; i++) adata.push_back(i);
    for(int i=0; i<120; i++) bdata.push_back(i);

    Matrix<float> a(adata, {2,3,2,4});
    Matrix<float> b(bdata, {2,3,4,5});
    Matrix<float> res = a.matmul(b);

    bool shapeOk = res.shape[0]==2 && res.shape[1]==3 
                && res.shape[2]==2 && res.shape[3]==5;
    // res[0,0,0,0] = a[0,0,0,:] . b[0,0,:,0] = 0*0+1*5+2*10+3*15 = 70
    bool valOk = res.data[0] == 70;
    (shapeOk && valOk) ? PASS("4D matmul") : FAIL("4D matmul");
}

void test_nonuniform_sum_axis0()
{
    // numpy:
    // a = np.arange(24).reshape(3,2,4)  -> 3 slices of shape {2,4}
    // np.sum(a, axis=0).shape = {2,4}
    // np.sum(a, axis=0) = [[24,27,30,33],[36,39,42,45]]
    std::vector<float> data;
    for(int i=0; i<24; i++) data.push_back(i);
    Matrix<float> a(data, {3,2,4});
    Matrix<float> res = a.sum(0);

    bool shapeOk = res.shape[0]==2 && res.shape[1]==4;
    bool valOk = res.data[0]==24 && res.data[1]==27
              && res.data[2]==30 && res.data[3]==33
              && res.data[4]==36 && res.data[5]==39
              && res.data[6]==42 && res.data[7]==45;
    (shapeOk && valOk) ? PASS("non-uniform sum axis=0 {3,2,4}") : FAIL("non-uniform sum axis=0 {3,2,4}");
}

void test_nonuniform_sum_axis1()
{
    // numpy:
    // a = np.arange(24).reshape(3,2,4)
    // np.sum(a, axis=1).shape = {3,4}
    // np.sum(a, axis=1) = [[4,6,8,10],[20,22,24,26],[36,38,40,42]]
    std::vector<float> data;
    for(int i=0; i<24; i++) data.push_back(i);
    Matrix<float> a(data, {3,2,4});
    Matrix<float> res = a.sum(1);

    bool shapeOk = res.shape[0]==3 && res.shape[1]==4;
    bool valOk = res.data[0]==4  && res.data[1]==6
              && res.data[2]==8  && res.data[3]==10
              && res.data[4]==20 && res.data[5]==22
              && res.data[6]==24 && res.data[7]==26
              && res.data[8]==36 && res.data[9]==38
              && res.data[10]==40 && res.data[11]==42;
    (shapeOk && valOk) ? PASS("non-uniform sum axis=1 {3,2,4}") : FAIL("non-uniform sum axis=1 {3,2,4}");
}

void test_nonuniform_sum_axis2()
{
    // numpy:
    // a = np.arange(24).reshape(3,2,4)
    // np.sum(a, axis=2).shape = {3,2}
    // np.sum(a, axis=2) = [[6,22],[38,54],[70,86]]  
    std::vector<float> data;
    for(int i=0; i<24; i++) data.push_back(i);
    Matrix<float> a(data, {3,2,4});
    Matrix<float> res = a.sum(2);    
    bool shapeOk = res.shape[0]==3 && res.shape[1]==2;
    bool valOk = res.data[0]==6  && res.data[1]==22
              && res.data[2]==38 && res.data[3]==54
              && res.data[4]==70 && res.data[5]==86;
    (shapeOk && valOk) ? PASS("non-uniform sum axis=2 {3,2,4}") : FAIL("non-uniform sum axis=2 {3,2,4}");
}

void test_nonuniform_transpose()
{
    // numpy:
    // a = np.arange(24).reshape(2,3,4)
    // np.transpose(a).shape = {4,3,2}
    // np.transpose(a)[0,0,0] = 0
    // np.transpose(a)[1,0,0] = 1
    // np.transpose(a)[0,1,0] = 4
    // np.transpose(a)[0,0,1] = 12
    std::vector<float> data;
    for(int i=0; i<24; i++) data.push_back(i);
    Matrix<float> a(data, {2,3,4});
    Matrix<float> res = a.transpose();
    bool shapeOk = res.shape[0]==4 && res.shape[1]==3 && res.shape[2]==2;
    // flat index of [1,0,0] in {4,3,2} = 1*6+0*2+0 = 6
    // flat index of [0,1,0] in {4,3,2} = 0*6+1*2+0 = 2
    // flat index of [0,0,1] in {4,3,2} = 0*6+0*2+1 = 1
    bool valOk = res.data[0]==0
              && res.data[6]==1
              && res.data[2]==4
              && res.data[1]==12;
    (shapeOk && valOk) ? PASS("non-uniform transpose {2,3,4}") : FAIL("non-uniform transpose {2,3,4}");
}

void test_3D_arithmetic()
{
    // numpy:
    // a = np.ones((2,3,4))
    // b = np.ones((2,3,4)) * 2
    // a + b = 3 everywhere,  a * b = 2 everywhere
    std::vector<float> ones(24, 1.0f);
    std::vector<float> twos(24, 2.0f);
    Matrix<float> a(ones, {2,3,4});
    Matrix<float> b(twos, {2,3,4});

    Matrix<float> add = a + b;
    Matrix<float> mul = a * b;

    bool addOk = true, mulOk = true;
    for(int i=0; i<24; i++)
    {
        if(add.data[i] != 3.0f) addOk = false;
        if(mul.data[i] != 2.0f) mulOk = false;
    }
    addOk ? PASS("3D add {2,3,4}") : FAIL("3D add {2,3,4}");
    mulOk ? PASS("3D mul {2,3,4}") : FAIL("3D mul {2,3,4}");
}

// ============================================================================
//  BROADCAST FORWARD TESTS
// ============================================================================

// numpy:
//   a = np.array([[1],[2],[3]])          shape (3,1)
//   b = np.array([[10,20,30]])           shape (1,3)
//   a + b
//   = [[11,21,31],
//      [12,22,32],
//      [13,23,33]]                       shape (3,3)
void test_broadcast_outer_product_add()
{
    Matrix<float> a({{1},{2},{3}});           // shape {3,1}
    Matrix<float> b({{10,20,30}});            // shape {1,3}
    Matrix<float> res = a + b;

    bool shapeOk = res.shape[0]==3 && res.shape[1]==3;
    bool valOk = res.data[0]==11 && res.data[1]==21 && res.data[2]==31
              && res.data[3]==12 && res.data[4]==22 && res.data[5]==32
              && res.data[6]==13 && res.data[7]==23 && res.data[8]==33;
    (shapeOk && valOk) ? PASS("broadcast {3,1}+{1,3}") : FAIL("broadcast {3,1}+{1,3}");
}

// numpy:
//   a = np.array([[1,2,3],[4,5,6]])      shape (2,3)
//   b = np.array([10,20,30])             shape (3,)
//   a + b
//   = [[11,22,33],
//      [14,25,36]]                       shape (2,3)
void test_broadcast_1D_to_2D()
{
    Matrix<float> a({{1,2,3},{4,5,6}});       // shape {2,3}
    Matrix<float> b({10,20,30});              // shape {3}
    Matrix<float> res = a + b;

    bool shapeOk = res.shape[0]==2 && res.shape[1]==3;
    bool valOk = res.data[0]==11 && res.data[1]==22 && res.data[2]==33
              && res.data[3]==14 && res.data[4]==25 && res.data[5]==36;
    (shapeOk && valOk) ? PASS("broadcast {2,3}+{3}") : FAIL("broadcast {2,3}+{3}");
}

// numpy:
//   a = np.array([[1,2],[3,4]])          shape (2,2)
//   b = np.array([[10]])                 shape (1,1)
//   a + b
//   = [[11,12],
//      [13,14]]                          shape (2,2)
void test_broadcast_scalar_matrix()
{
    Matrix<float> a({{1,2},{3,4}});           // shape {2,2}
    Matrix<float> b({{10}});                  // shape {1,1}
    Matrix<float> res = a + b;

    bool shapeOk = res.shape[0]==2 && res.shape[1]==2;
    bool valOk = res.data[0]==11 && res.data[1]==12
              && res.data[2]==13 && res.data[3]==14;
    (shapeOk && valOk) ? PASS("broadcast {2,2}+{1,1}") : FAIL("broadcast {2,2}+{1,1}");
}

// numpy:
//   a = np.array([[[1,2,3]],[[4,5,6]]])          shape (2,1,3)
//   b = np.arange(24).reshape(2,4,3)             shape (2,4,3)
//   (a + b)[0,0,:] = [0+1, 1+2, 2+3]   = [1,3,5]
//   (a + b)[0,3,:] = [9+1, 10+2, 11+3] = [10,12,14]
//   (a + b)[1,0,:] = [12+4,13+5,14+6]  = [16,18,20]
void test_broadcast_3D_middle_axis()
{
    std::vector<float> adata = {1,2,3,4,5,6};
    Matrix<float> a(adata, {2,1,3});          // shape {2,1,3}

    std::vector<float> bdata;
    for(int i=0; i<24; i++) bdata.push_back((float)i);
    Matrix<float> b(bdata, {2,4,3});          // shape {2,4,3}

    Matrix<float> res = a + b;

    bool shapeOk = res.shape[0]==2 && res.shape[1]==4 && res.shape[2]==3;
    // res[0,0,:] flat index 0,1,2
    bool v1 = res.data[0]==1 && res.data[1]==3 && res.data[2]==5;
    // res[0,3,:] flat index 9,10,11
    bool v2 = res.data[9]==10 && res.data[10]==12 && res.data[11]==14;
    // res[1,0,:] flat index 12,13,14
    bool v3 = res.data[12]==16 && res.data[13]==18 && res.data[14]==20;

    (shapeOk && v1 && v2 && v3) ? PASS("broadcast {2,1,3}+{2,4,3}") : FAIL("broadcast {2,1,3}+{2,4,3}");
}

// numpy:
//   a = np.array([[1,2,3],[4,5,6]])      shape (2,3)
//   b = np.array([[10,20,30]])           shape (1,3)
//   a * b = [[10,40,90],[40,100,180]]    shape (2,3)
void test_broadcast_multiply()
{
    Matrix<float> a({{1,2,3},{4,5,6}});       // shape {2,3}
    Matrix<float> b({{10,20,30}});            // shape {1,3}
    Matrix<float> res = a * b;

    bool shapeOk = res.shape[0]==2 && res.shape[1]==3;
    bool valOk = res.data[0]==10 && res.data[1]==40  && res.data[2]==90
              && res.data[3]==40 && res.data[4]==100 && res.data[5]==180;
    (shapeOk && valOk) ? PASS("broadcast multiply {2,3}*{1,3}") : FAIL("broadcast multiply {2,3}*{1,3}");
}

// ============================================================================
//  sumGradForBroadcast TESTS
// ============================================================================

// numpy:
//   grad = np.ones((3,3))               shape (3,3)
//   originalShape = (3,1)
//   np.sum(grad, axis=1, keepdims=True) = [[3],[3],[3]]   shape (3,1)
void test_sumgrad_col_vector()
{
    std::vector<float> ones(9, 1.0f);
    Matrix<float> grad(ones, {3,3});
    shape_t orig = {3,1};
    Matrix<float> res = sumGradForBroadcast(grad, orig);

    bool shapeOk = res.shape[0]==3 && res.shape[1]==1;
    bool valOk = res.data[0]==3 && res.data[1]==3 && res.data[2]==3;
    (shapeOk && valOk) ? PASS("sumGradForBroadcast {3,3}->{3,1}") : FAIL("sumGradForBroadcast {3,3}->{3,1}");
}

// numpy:
//   grad = np.ones((2,3))               shape (2,3)
//   originalShape = (3,)
//   np.sum(grad, axis=0) = [2,2,2]      shape (3,)
void test_sumgrad_1D_from_2D()
{
    std::vector<float> ones(6, 1.0f);
    Matrix<float> grad(ones, {2,3});
    shape_t orig = {3};
    Matrix<float> res = sumGradForBroadcast(grad, orig);

    bool shapeOk = res.shape[0]==3;
    bool valOk = res.data[0]==2 && res.data[1]==2 && res.data[2]==2;
    (shapeOk && valOk) ? PASS("sumGradForBroadcast {2,3}->{3}") : FAIL("sumGradForBroadcast {2,3}->{3}");
}

// numpy:
//   grad = np.ones((2,3))               shape (2,3)
//   originalShape = (2,3)               no broadcast happened
//   result should be grad unchanged
void test_sumgrad_no_broadcast()
{
    std::vector<float> ones(6, 1.0f);
    Matrix<float> grad(ones, {2,3});
    shape_t orig = {2,3};
    Matrix<float> res = sumGradForBroadcast(grad, orig);

    bool shapeOk = res.shape[0]==2 && res.shape[1]==3;
    bool valOk = true;
    for(int i=0; i<6; i++) if(res.data[i] != 1.0f) valOk = false;
    (shapeOk && valOk) ? PASS("sumGradForBroadcast no-op {2,3}->{2,3}") : FAIL("sumGradForBroadcast no-op {2,3}->{2,3}");
}

// numpy:
//   grad = np.ones((3,3))               shape (3,3)
//   originalShape = (1,3)               row was broadcast along axis 0
//   np.sum(grad, axis=0, keepdims=True) = [[3,3,3]]  shape (1,3)
void test_sumgrad_row_vector()
{
    std::vector<float> ones(9, 1.0f);
    Matrix<float> grad(ones, {3,3});
    shape_t orig = {1,3};
    Matrix<float> res = sumGradForBroadcast(grad, orig);

    bool shapeOk = res.shape[0]==1 && res.shape[1]==3;
    bool valOk = res.data[0]==3 && res.data[1]==3 && res.data[2]==3;
    (shapeOk && valOk) ? PASS("sumGradForBroadcast {3,3}->{1,3}") : FAIL("sumGradForBroadcast {3,3}->{1,3}");
}


// Sigmoid Test ───────────────────────────────────────────────────────────────────



void sigmoidTest()
{
    Tensor_t<double> w0(make_tensor<double>(2));
    Tensor_t<double> x0(make_tensor<double>(-1));
    Tensor_t<double> w1(make_tensor<double>(-3));
    Tensor_t<double> x1(make_tensor<double>(-2));
    Tensor_t<double> bias(make_tensor<double>(-3));

    Tensor_t<double> neg = make_tensor<double>(-1);
    Tensor_t<double> one = make_tensor<double>(1);

    Tensor_t<double> b = neg * (w0*x0 + w1*x1 + bias) ;

    Tensor_t<double> c = one / (one + b->exp());
 
    c->backward(Matrix<double>(1));

    std::cout << w0->grad;

    std::cout << x0->grad;

    std::cout << w1->grad;

    std::cout << x1->grad;

    std::cout << bias->grad;

}

template<typename T>
void nn(Tensor_t<T> input, Tensor_t<T> labels, int epochs)
{
    Linear<T> l1(input->data.shape[1], 2, true);
    Linear<T> l2(2, 1, true);

    std::vector<Tensor_t<T>> params(l1.parameters());
    auto param2 = l2.parameters(); 
    params.insert(params.end(), param2.begin(), param2.end());

    Optimizer<T> Op(params,0.1, SGD);

    for(int epoch = 0; epoch < epochs; epoch++)
    {
        // forward
        Tensor_t<T> a = l1.forward(input);
        Tensor_t<T> b = l2.forward(a);
        Tensor_t<T> out = b->sigmoid();

        // loss
        Tensor_t<T> loss = ((out-labels)^(T)2)/(T)labels->data.shape[0];

        if(epoch % 10000 == 0)
            std:: cout << "epoch: " <<epoch << " "<< "loss: "<< ((loss->sum())/(T)labels->data.shape[0])->data; 

        // backward
        loss->backward(Matrix<T>(1));

        Op.step();

    }

}

// ─── main ───────────────────────────────────────────────────────────────────

int main()
{
    // cout << "=== 1D ===\n";
    // test_1D_dot();
    // test_1D_sum();
    // test_1D_arithmetic();

    // cout << "\n=== 2D ===\n";
    // test_2D_matmul();
    // test_2D_transpose();
    // test_2D_sum_axis0();
    // test_2D_sum_axis1();

    // cout << "\n=== 3D ===\n";
    // test_3D_matmul();
    // test_3D_sum_axis0();
    // test_3D_sum_axis1();
    // test_3D_sum_axis2();
    // test_3D_transpose();
    
    // cout << "\n=== 4D & non-uniform ===\n";
    // test_4D_matmul(); 
    // test_nonuniform_sum_axis0();
    // test_nonuniform_sum_axis1();
    // test_nonuniform_sum_axis2(); 
    // test_nonuniform_transpose(); 
    // test_3D_arithmetic(); 

    // cout << "\n=== Broadcast forward ===\n";
    // test_broadcast_outer_product_add();
    // test_broadcast_1D_to_2D();
    // test_broadcast_scalar_matrix();
    // test_broadcast_3D_middle_axis();
    // test_broadcast_multiply();

    // cout << "\n=== sumGradForBroadcast ===\n";
    // test_sumgrad_col_vector();
    // test_sumgrad_1D_from_2D();
    // test_sumgrad_no_broadcast();
    // test_sumgrad_row_vector();

    // cout << "\n=== Sigmoid test ===\n";
    // sigmoidTest();

    cout << "\n=== nn test ===\n";
    Tensor_t<double> in = make_tensor<double>({{0,0},{0,1},{1,0},{1, 1}});
    Tensor_t<double> y = make_tensor<double>({{0}, {1}, {1}, {0}});

    nn(in, y, 100000);
    return 0;

}