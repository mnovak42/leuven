
#ifndef INCCHOLESKY_HH
#define INCCHOLESKY_HH

//#include "types.hh"
#include "Matrix.hh"

#include "types.hh"

#include <iostream>
#include <vector>


/**
 * @brief Incomplete (pivoted) Cholesky decomposition of the *kernel matrix*.
 *
 * **Description:**
 *
 * Given a set of input data in the input space 
 * \f$ \{ \mathbf{x}_i \}_{i=1}^{N} \mathbf{x}_i \in \mathbb{R}^d \f$ and a non-
 * linear mapping \f$ \varphi{(\cdot)}: \mathbb{R}^d\to\mathbb{R}^{n_h}\f$ 
 * through the corresponding kernel function 
 * \f$ \kappa(\mathbf{x}_i,\mathbf{x}_j) = \varphi{(\mathbf{x}_i)}^T\varphi{(\mathbf{x}_i)}\f$
 * to evaluate the inner products in the feature space, the Gram (or kernel) 
 * matrix \f$ \Omega_{ij} = \varphi{(\mathbf{x}_i)}^T\varphi{(\mathbf{x}_i)} = 
 * \kappa(\mathbf{x}_i,\mathbf{x}_j), \in \mathbb{R}^{(N\times N)} \f$. 
 * (The crrent implementation assumes a normalised kernel i.e. 
 *  \f$ \kappa(\varphi(\mathbf{x})_i, \varphi(\mathbf{x})_i) = \varphi(\mathbf{x})_i^T\varphi(\mathbf{x})_i =1 \f$
 *  which is used only in the initialisation of the diagonals and the error computation.)
 *
 * The incomplete Cholesky decomposition generates the \f$ G \in \mathbb{R}^{(R\times N)}, R \leq N\f$
 * upper triangualr matrix (or optionally its transpose) such that 
 * \f$ \Omega \approx \tilde{\Omega} = G^TG \f$. Actually, the transpose of the 
 * feature map matrix \f$ \Phi = [\varphi(\mathbf{x})_1^T,\varphi(\mathbf{x})_2^T,\dots,\varphi(\mathbf{x})_N^T] 
 * \in \mathbb{R}^{(N \times n_h)} \f$ is orthogonalised by geedily slecting the 
 * feature map at each step with the highest residual norm and normalising to the 
 * previously selected and orthogonalised vectors. During this QR decomposition 
 * of \f$ \Phi^T \f$, an orthonormal basis is greedily generated and at the 
 * \f$ k\f$-th steps the \f$ k\f$ columns of the \f$ Q \in \mathbb{R}^{(n_h \times k)} \f$
 * contains these orthonormal basis vectors while the columns of the matrix 
 * \f$ G \in \mathbb{R}^{(k \times N)}\f$ contains the projections of the  
 * \f$ \{\varphi(\mathbf{x})_1,\dots,\varphi(\mathbf{x})_N^T \f$ feature maps onto 
 * these basis. Therefore, the matrix \f$ Q^T\Phi^T \approx G \in \mathbb{R}^{(R \times N)}, R \leq N\f$
 * can be interpreted as a low dimensional representation of the feature map of the input 
 * data and \f$ \Omega = \Phi\Phi^T \approx [QG]^T[QG] = G^TG \f$ gives a low rank 
 * approximation of the kernel matrix.
   \rst 
     See more details at the :ref:`seclabel-ICHOLDoc` section of the documentation.
   \endrst
 *
 * **How to use:**
 *
 * The pointer to the input data matrix, the kernel and its parameters needs to 
 * be set before the decompositon: 
 * 
 * - ``kernel function``: the class is templated on the **kernel function** class 
 *    so the user needs to chose the kernel when instantiate the class object. 
 *    The IncCholesky::SetKernelParameters method (that will invoke the corresponding 
 *    interface method of the kernel function object) is provided by the class 
 *    to set the paraneters of the kernel.
 * - ``input data``: the user needs to set the pointer to the input data matrix 
 *     before invoking the decomposition. The input data matrix is assumed to 
 *     store the input data vectors (in the input space) **in row-major format** 
 *     (i.e. each individual data vector is memory continuos). 
 *     The IncCholesky::SetInputDataMatrix method is by the class to set the 
 *     input data matrix pointer.
 * - ``decompositon``: the IncCholesky::Decompose method can be invoked by the 
 *     user to perform the incomplete Cholesky decomposition after completing 
 *     all these above required settings. See more details on the termination 
 *     criteria at the documentation of the IncCholesky::Decompose method.
 * 
 * **Result**:
 * 
 * The incomplete Cholesky decompositon i.e. the \f$ G \in mathbb{R}^{(R\times N)}\f$ 
 * (or its traspose if it was required) upper triangualr matrix such that 
 * \f$ \Omega = \Phi\Phi^T \approx [QG]^T[QG] = G^TG \f$ is available after the 
 * decomposition (IncCholesky::Decompose) is completed.
 * A pointer to the matrix \f$ G \f$ can be obtained by the IncCholesky::GetICholMatrix
 * method while a reference to the vector containing the input data indices, 
 * as the result of the permutations done during the decomposition i.e. according 
 * to the columns of \f$ G\f$, can be obtained by the IncCholesky::GetPermutationVector 
 * method.
 *
 * @author M. Novak
 * @date   December 2020
 */


// templated on the kernel and on the data type (T) and on the inpud data type (TInp)
template < class TKernel, typename T,  typename TInputD>
class IncCholesky {

public:

  IncCholesky() { 
    fKernel        = new TKernel(); 
    fInputDataM    = nullptr;
    fICholM        = nullptr;
    fFinalResidual = 1.0; 
  }
 
 ~IncCholesky() { 
   delete fKernel; 
   // the fInputDataM is not owned by the class
   if ( fICholM ) {
     BLAS theBlas;
     theBlas.Free(*fICholM);
     delete fICholM;
   }
  }

  /**
   * @brief Public method to set the parameters of the kernel function object.
   * 
   * The type of the kernel function object is selected at instantiation of an
   * IncCholesky object since the class is templated on this type. This public 
   * method can be used to set the parameters of the kernel function object. 
   * Note, that the method will invoke the corresponding \f$ \texttt{SetKernelParameters}\f$
   * method of the kernel function obejct through the base class KernelBase::SetParameters 
   * interface method by passing all provided parameters as arguments.
   * 
   * @param[in] args input argument list. 
   */
  template < typename... Args >
  void SetKernelParameters(Args... args) {
    fKernel->SetParameters(args...);
  }

  /**
   * @brief Public method to set the input pointer to the input data matrix. 
   * 
   * @param[in] inDataM pointer to the input data matrix that stores the input 
   *     data vector in row-major (memory continous) order.
   */
  void SetInputDataMatrix(Matrix<TInputD, false>* inDataM) { fInputDataM = inDataM; }

  /**
   * @brief Public method to perform the incomplete Cholesky decomposition of 
   *        the kernel matrix.
   *
   * Note, that the pointer to the input data matrix as well as the kernel functon 
   * parameters needs to be set before invoking this method.
   *
   * The decomposition will be terminated when either the proided approximation 
   * error or maximum number of iteration ( i.e. selected data vectors or rank of 
   * the approximation) is reached. The incomplete Cholesky decomposition 
   * can be interpreted as a trace optimisation \f$ \| \Omega - \tilde{\Omega}\|_1 \f$
   * (where \f$ \| \|_1\f$ denotes the sum of singular values which in turn is eaual 
   * to the sum of eigenvalues and that is equal to the trace of the matrix since 
   * the matrices involved are symmetric, positive semi-definite) or 
   * in orther words \f$ \texttt{max} [ \texttt{trace} \{\tilde{\Omega}\} ]\f$ . 
   * Therefore, the approximaton error will be computed at each step as 
   * \f$ \eta = \texttt{trace} [ \Omega - \tilde{\Omega} ]/N \f$ i.e. normalised 
   * by \f$ \texttt{trace} \{ \Omega \} \f$ assuming normalised kernel i.e. 
   * \f$ \Omega_{ii}=1 \f$. 
   *
   * @param[in] tolError maximum tolerated error value (see above)
   * @param[in] maxIter maximum number of iteration (or maximum dimension of the 
   *                    underlying sub-space, or maximum rank of the approximation, 
   *                    maximum number of data vectors orthogonalised)   
   * @param[in] transpose the algorithm will compute the \f$ G \in \mathbb{R}^{(R\times N)}\f$
   *                      upper triangualr matrix. The transpose of this can be 
   *                      requested by the user if this parameter is set to be 
   *                      \f$\texttt{true}\f$ (default).
   */  
  void Decompose(double tolError, size_t maxIter, bool transpose=true);

  /**@brief Public method to obtain a pointer to the resulted incomplete Cholesky factor.*/
  Matrix<T>* GetICholMatrix()   const { return fICholM; }
  
  /**@brief Removes the ownership of the memory related to the incomplete Cholesky matrix*/
  void SetNullICholMatrixPrt() {fICholM = nullptr; }
  
  /**@brief Public method to obtain a reference to the vector containing the 
    *       input data indices as the result of the permutations done during the 
    *       decompositon.
    */
  const std::vector<size_t>& GetPermutationVector() const { return fPermutationVect; }
  
  double     GetFinalResidual() const { return fFinalResidual; }
  
private:

/**
 * @name Data members
 */
// @{
  /**@brief Pointer to the kernel object that implements the kernel function. */  
  TKernel*              fKernel;
  // vector that stores the current indexing
  std::vector<size_t>   fPermutationVect;
  // pointer to the input data (doesn't owned by the class)
  Matrix<TInputD, false>* fInputDataM;  
  // pointer to the incomplete Cholesky matrix (owned by the class; r)
  Matrix<T>*            fICholM; 
  // the final error: square root of the sum of squared residuals i.e. sum of 
  //                  the squared projections to the orthogonal complements
  double                fFinalResidual;
// @} // end Data members  group

};


#include "IncCholesky.tpp"

#endif