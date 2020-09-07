
// This file needs to be included into "Kernel.hh" after the base class definition

#ifndef KERNELSSK_HH
#define KERNELSSK_HH

#include <cmath>
#include <vector>

// T is the kernel retun type which must be the same as the computing type i.e. 
// (double or float)
// the SSK kerel will operate on std::string pointers as input data 
template <typename T>
class KernelSSK : public KernelBase < KernelSSK <T>, T > {

public:
  
  KernelSSK() : fSubSeqLength(3), fLambdaDecay(0.5) { }
 ~KernelSSK() { }

 template <typename... Args>
 void SetKernelParameters(Args... args) {
   SetSubSeqLenAndLambdaDecay(args...);
 }
 
 template <typename... Args>
 T EvaluateKernel(Args... args) {
   return EvaluateSSK(args...);
 }

  
 
private:
  
  T EvaluateSSK(const std::string* const* str1, const std::string* const* str2, size_t /*num*/) {
    T val = 1.0;
    std::cout << **str1 << " vs "<< **str2 << std::endl;
    return val;
  }
  
  void SetSubSeqLenAndLambdaDecay(size_t ssl, float ld) { 
    fSubSeqLength = ssl;
    fLambdaDecay  = ld;
  }


private:
  
  size_t   fSubSeqLength;
  float    fLambdaDecay;

};

#endif