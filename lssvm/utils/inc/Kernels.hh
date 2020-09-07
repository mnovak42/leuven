
#ifndef KERNELS_HH
#define KERNELS_HH

/**
 * @brief  CRTP base class for Kernels.
 *
 * Since kernel function evaluations are called frequently, the cost of virtual
 * calls i.e. Dynamic Polymorphism is avoided by using Static Polymorphism based
 * on Curiously Recurring Template Pattern (CRTP). This base class provides the 
 * SetParameters and Evaluate interface methods in which the concrete base 
 * class implementations of the corresponding methods will be invoked for setting 
 * the kernel function parameter(s) and for evaluating the kernel function.
 * The KernelRBF class, that implements the Radial Basis Function (RBF) kernel,
 * is provided as an example for kernel implementations.
 *
 * @author M. Novak
 * @date   December 2020
 *
 */

template < class TDerived, typename TReturnType >
class KernelBase {
public:

  /**
    * @brief Interface method to set the kernel function parameters. 
    *
    * The derived class \f$ \texttt{SetKernelParameters}\f$ method will be 
    * invoked within this method to set the kernel function parameters.
    *
    * @param[in] args  input argument list. 
    */
  template <typename... Args>
  void SetParameters(Args... args) {
    // caling derived::SetKernelParameters method 
    static_cast<TDerived*>(this)->SetKernelParameters(args...);
  }

  /**
    * @brief Interface method to evaluate the kernel function with the given 
    *        input arguments. 
    *
    * The derived class \f$ \texttt{EvaluateKernel} \f$ method will be invoked 
    * within this method to evaluate the kernel function with the given input 
    * arguments.
    *
    * @param[in] args  input argument list. 
    */
  template <typename... Args>
  TReturnType Evaluate(Args... args) {
    // caling derived::EvaluateKernel method 
    return static_cast<TDerived*>(this)->EvaluateKernel(args...);
  }
};


#include "KernelRBF.tpp"
#include "KernelSSK.tpp"
#include "KernelChi2.tpp"

#endif
