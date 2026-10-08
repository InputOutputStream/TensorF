// Overload.hpp scope: operators exist ONLY for numeric element types.
#include "core/Overloads/Overload.hpp"
#include "core/Types/types.hpp"
#include <string>
#include <cassert>
#include <sstream>
struct Foo { int x; bool operator==(const Foo&) const = default; };
using VS = std::vector<std::string>; using VF = std::vector<Foo>; using Vf = std::vector<float>;

template<class A,class B> concept can_add = requires(A a,B b){ a + b; };
template<class A,class B> concept can_sub = requires(A a,B b){ a - b; };
template<class A,class B> concept can_mul = requires(A a,B b){ a * b; };
template<class A,class B> concept can_div = requires(A a,B b){ a / b; };
template<class A> concept can_neg = requires(A a){ -a; };
template<class A,class B> concept can_pluseq = requires(A a,B b){ a += b; };
template<class A,class B> concept can_eq = requires(A a,B b){ a == b; };
template<class A,class B> concept can_lt = requires(A a,B b){ a < b; };
template<class A> concept can_print = requires(std::ostream& o, A v){ o << v; };

// negative: non-numeric vectors get NO global arithmetic / stream operator
static_assert(!can_add<VS,VS> && !can_sub<VS,VS> && !can_mul<VS,VS> && !can_div<VF,VF>);
static_assert(!can_print<VS> && !can_print<VF>);
static_assert(!can_neg<VS> && !can_pluseq<VS,std::string>);
static_assert(!can_eq<VS,const char*>);        // the scalar == (returning vector<uint8_t>) is not applied to strings
static_assert(!can_lt<VS,std::string>);
// positive: numeric vectors keep every operator
static_assert(can_add<Vf,Vf> && can_sub<Vf,Vf> && can_mul<Vf,Vf> && can_div<Vf,Vf> && can_neg<Vf>);
static_assert(can_add<Vf,float> && can_mul<float,Vf> && can_sub<int,Vf> && can_pluseq<Vf,int> && can_pluseq<Vf,Vf>);
static_assert(can_eq<Vf,Vf> && can_eq<Vf,int> && can_lt<Vf,float> && can_print<Vf>);
static_assert(can_add<std::vector<fp8_e4m3>,std::vector<fp8_e4m3>> && can_pluseq<std::vector<fp8_e4m3>,std::vector<fp8_e4m3>>);
static_assert(can_eq<std::vector<uint8_t>,int>);

int main(){
    // vector<string>/vector<Foo> == still uses std's operator== (not hijacked)
    VS s1{"a","b"}, s2{"a","b"}, s3{"a"}; assert(s1 == s2); assert(!(s1 == s3));
    VF f1{{1},{2}}, f2{{1},{2}}; assert(f1 == f2);
    Vf a{1,2,3}, b{4,5,6};
    assert((a + b) == (Vf{5,7,9})); assert((2.f * a) == (Vf{2,4,6})); assert((a == 2.f) == (std::vector<uint8_t>{0,1,0})); assert(!(a == b));
    std::ostringstream os; os << a; assert(os.str() == "[1,2,3,]");
    bool threw=false; try { (void)(a + Vf{1}); } catch(const std::invalid_argument&){ threw=true; } assert(threw);
    threw=false; try { (void)(a / Vf{1,0,1}); } catch(const std::runtime_error&){ threw=true; } assert(threw);
    std::puts("scope OK");
}
