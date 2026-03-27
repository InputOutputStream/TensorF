#include "types.hpp"
#include "header.hpp"
#include "Matrix.hpp"

using namespace std;

template <typename T>
void printTensor(vector<T> vect)
{
    cout<<"(";
    for(auto i : vect)
    {
        cout<<i<<",";
    }
    cout<<")"<<endl;
}


void sigmoidTest()
{
    Tensor_t<float> w0 = make_tensor(vector<float> {2});
    Tensor_t<float> x0 = make_tensor(vector<float> {-1});

    Tensor_t<float> w1 = make_tensor(vector<float> {-3});
    Tensor_t<float> x1 = make_tensor(vector<float> {-2});
    
    Tensor_t<float> bias = make_tensor(vector<float> {-3});


    Tensor_t<float> neg = make_tensor(vector<float>{-1});
    Tensor_t<float> one = make_tensor(vector<float>{1});


    Tensor_t<float> e = (float)-1 *(w0*x0 + w1*x1 + bias);
    Tensor_t<float> c = one/(one + e->exp());
 
    c->backward(vector<float>{1});

    printTensor(w0->grad);
    printTensor(x0->grad);
    
    printTensor(w1->grad);
    printTensor(x1->grad);

    printTensor(bias->grad);
    
}



int main()
{   
    // sigmoidTest();
    

    //  Matrix<float> m1(9.0f); 

    //  Matrix<float> m2(9.0f); 
     

    Matrix<float> m3({9.0f, 9.0f}); 

    Matrix<float> m4({9.0f, 9.0f}); 
    
    Matrix<float> m5({{1,8, 0},{1,6,8},{2,1,1}});

    Matrix<float> m6({{9,9,2}, {1,2,8},{1,0, 2}});


    Matrix<float> m7({{1,8}});

    Matrix<float> m8({{9,2}});

    // Matrix<float> m9({{{1,8},{1,8},{2,1}}, {1,8},{1,8},{2,1}}); 

    // Matrix<float> m10({{{9,2}, {1,8},{1, 2}}, {{9,2}, {1,8},{1, 2}}}); 

    // cout << m1<<m1.shape<<"\n\n";
    // cout << m2<<m2.shape<<"\n\n";
    // cout<<".....................\n\n";

    cout << m3<<m3.shape<<"\n\n";
    cout << m4<<m4.shape<<"\n\n";
    cout<<".....................\n\n";

    cout << m5<<m5.shape<<"\n\n";
    cout << m6<<m6.shape<<"\n\n";
    cout<<".....................\n\n";
    
    cout << m7<<m7.shape<<"\n\n";
    cout << m8<<m8.shape<<"\n\n";
    cout<<".....................\n\n";

    // cout << m9<<m9.shape<<endl;
    // cout << m10<<m10.shape<<endl;
    cout<<"Result zone.....................\n\n";

    Matrix<float> res = m5.matmul(m6);

    cout<<res;

    return 0;
}


