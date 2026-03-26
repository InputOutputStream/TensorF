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



//     Matrix(vector<T> val, vector<long> shape)

int main()
{   
    sigmoidTest();

    // Matrix<float> m1({1,2}, {1,2});

    // Matrix<float> m2({1,2}, {1, 2});

    // cout << m1<<endl;
    // cout<<".....................\n";
    // cout << m2<<endl;
    
    // Tensor_t<float> m3 = m1.dot(m2);

    // cout<<m3;

    return 0;
}


