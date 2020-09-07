

// This file needs to be included into "Kernel.hh" after the base class definition

#ifndef KERNELCHI2_HH
#define KERNELCHI2_HH

#include <cmath>
#include <vector>

/**
 * @brief  Chi2 Kernel implementation.
 *
 * The class implements the Chi2 kernel function in the form of  
 * \f[ 
       K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left[  - \frac{ \chi_{ij}^2 }{\gamma }   \right] 
       = \exp\left[ -\frac{\sum_{k=1}^{d} (x_k^i-x_k^j)^2}{\gamma (x_k^i+x_k^j)}  \right]
   \f]
 * where \f$ \mathbf{x}_i,\mathbf{x}_j \in \mathcal{D}=\{\mathbf{x}_l\}_{l=1}^{N}, 
 * \mathbf{x}_l \in \mathrm{R}^d\f$ is the input data set and \f$\gamma\f$ is 
 * the (bandwidth) kernel parameter.
 *
 * The two interface methods of the KernelBase base class have been implemented:
 *  - KernelBase::SetParameters will invoke the KernelRBF::SetKernelParameters 
 *     method to set the kernel function parameter (i.e. the \f$\gamma\f$ bandwidth)
 *     either as a scalar or vector (the corresponding KernelChi2::SetBandWidth 
 *     method will be invoked).
 *  - KernelBase::Evaluate will invoke the KernelChi2::EvaluateKernel method to evaluate
 *     the kernel function (the KernelChi2:EvaluateKernelChi2 method will be invoked)
 *
  \rst
  .. note:: The return type of the kernel evaluation (T) can be differeent from 
     the type of the kernel parameter and input data (TInpD) in the case of this 
     implementatin of the RBF kernel. Any pattern can be implemented if this is 
     not suitable here with the only restriction, that the return type 
     must be either `double` or `float` (since those will populate the Kernel 
     matrix). 
  
     Also note, that the CTR is assumed to have an empty parameter list!
  \endrst
 *   
 *   
 * @author M. Novak
 * @date   December 2020
 */

// T    : is the return type of the kernel function that must be either double or
//        float. This is used to evaluate the kernel matrix so the computation
//        data type is the same (second parameter of the base calss). 
// TInpD: is the input data type on which the kernel function will operate. It 
//        also defines the type of the Chi2 kernel parameter i.e. the bandwidth. 
template <typename T, typename TInpD>
class KernelChi2 : public KernelBase < KernelChi2 <T, TInpD>, T > {
public:
  /**@brief Constructor (must be without arguments). */
  KernelChi2() : fDim(0) {}
  /**@brief Destructor (nothing to do). */
 ~KernelChi2() { }
  
  /**
    * @brief Main method to set the kernel function parameters. 
    *
    * The base class KernelBase::SetParameters interface method invokes this 
    * method by passing the argument(s). The appropriate SetBandWidt method will 
    * be invoked within this call by selecting the implementation that matches 
    * the argument list (scalar or vector in this case).
    *
    * @param[in] args  input argument list. 
    */
  template <typename... Args>
  void SetKernelParameters(Args... args) {
    SetBandWidth(args...);
  }
  /**@brief The concrete implementation for setting the **scalar** kernel bandwidth. 
    *@param[in] bw  the scalar Chi2 kernel bandwidth \f$\gamma\f$
    */
  void SetBandWidth(TInpD bw) {
    fDim = 1;
    fInvBandWV.resize(fDim);
    fInvBandWV[0] = 1./bw;    
  }
  /**@brief The concrete implementation for setting the **vector** kernel bandwidth. 
    *@param[in] bw  the Chi2 kernel bandwidth vector \f$\boldsymbol{\gamma}\f$
    */
  void SetBandWidth(std::vector<TInpD>& bw) {
    fDim = bw.size();
    fInvBandWV.resize(fDim);
    for (std::size_t i=0; i<fDim; ++i) {
      fInvBandWV[i] = 1./bw[i];
    } 
  }
  
  /**
    * @brief Main method to evaluate the kernel function. 
    *
    * The base class KernelBase::Evaluate interface method invokes this 
    * method by passing the argument(s). The appropriate EvaluateKernelChi2 method
    * will be invoked within this call by selecting the implementation that matches 
    * the argument list (only one version is implemented).
    *
    * @param[in] args  input argument list (two input data). 
    */
  template <typename... Args>
  T EvaluateKernel(Args... args) {
    return EvaluateKernelChi2(args...);
  }
  /**
    * @brief The concrete implementation for evaluating the kernel function (both 
    *        with scalar or vector bandwidth). 
    *
    * @param[in] a  pointer to the first input data array
    * @param[in] b  pointer to the second input data array
    * @param[in] num  dimension of the input data
    */
  T EvaluateKernelChi2(const TInpD* a, const TInpD* b, std::size_t num) {
    assert ( fDim!=0 && " Dimension of the kernel parameters is 0!");
    assert ( (fDim==1 || (fDim>1 && fDim==num)) && " Dimension of the kernel parameters is higher than 1 but not equal to the data dimension !");
    return (fDim==1) ? EvaluateKernelChi21D(a,b,num) : EvaluateKernelChi2MultiD(a,b,num);
  }
  
  


private:
  /**@brief Evaluation of the kernel function when scalar bandwidth is used. */
  T EvaluateKernelChi21D(const TInpD* a, const TInpD* b, std::size_t num) {
      T dum = 0.0;
//#pragma clang loop vectorize(enable)
      for (std::size_t id=0; id<num; ++id) {
        const T dum1 = a[id] - b[id];
        const T dum2 = a[id] + b[id];
        if (dum1*dum2!=0.) {
          dum -= dum1*dum1/dum2;  
        }
      }
      return std::exp(0.5*dum*fInvBandWV[0]);
  }
  /**@brief Evaluation of the kernel function when vector bandwidth is used. */
  T EvaluateKernelChi2MultiD(const TInpD* a, const TInpD* b, std::size_t num) {
      T dum = 0.0;
//#pragma clang loop vectorize(enable)
      for (std::size_t id=0; id<num; ++id) {
        const T dum1 = a[id] - b[id];
        const T dum2 = a[id] + b[id];
        if (dum1*dum2!=0.) {
          dum -= dum1*dum1*fInvBandWV[id]/dum2;  
        }
      }
      return std::exp(0.5*dum);
  }


private:
/**
 * @name Data members
 */
// @{  
  /**@brief Dimension of the Chi2 kernel bandwidth parameter. */
  std::size_t    fDim;
  /**@brief The Chi2 kernel bandwidth parameter value(s). */
  std::vector<TInpD> fInvBandWV;
// @} // end Data members  group

};


#endif
