
#include <iostream>

#include <stdlib.h>
#include <stdio.h>

#include "sys/time.h"
#include "time.h"

#include "types.hh"
#include "Matrix.hh"

#define min(x,y) (((x) < (y)) ? (x) : (y))

#define N 5      // dimensions of the square input matrix 
#define NEIGEN 3 // number of requested eigenvalues/eigenvectors

// Define data type for the test(double or float)
typedef double DTYPE;
//typedef float DTYPE;

DTYPE INPM_A [][N] = {
  {  0.67, -0.20,  0.19, -1.06,  0.46},
  { -0.20,  3.82, -0.13,  1.06, -0.48},
  {  0.19, -0.13,  3.27,  0.11,  1.10},
  { -1.06,  1.06,  0.11,  5.86, -0.98},
  {  0.46, -0.48,  1.10, -0.98,  3.54}
};

/*
  LAPACKE_dsyevr Example.
  =======================

  Program computes the smallest eigenvalues and the corresponding
  eigenvectors of a real symmetric matrix A using the Relatively Robust
  Representations, where A is:

     0.67  -0.20   0.19  -1.06   0.46
    -0.20   3.82  -0.13   1.06  -0.48
     0.19  -0.13   3.27   0.11   1.10
    -1.06   1.06   0.11   5.86  -0.98
     0.46  -0.48   1.10  -0.98   3.54

   
  Example Program Results.
  ========================

  LAPACKE_dsyevr (column-major, high-level) Example Program Results
  
  The total number of eigenvalues found: 3
  
  Selected eigenvalues
     0.43   2.14   3.37
  
  Selected eigenvectors (stored columnwise)
    -0.98  -0.01  -0.08
     0.01   0.02  -0.93
     0.04  -0.69  -0.07
    -0.18   0.19   0.31
     0.07   0.69  -0.13
*/

int main() {
  //
  // The matrix has 5 eigenvalue eigenvector pairs. The eigenpairs, corresponding 
  // to the eigenvalue indices of 1,2,3 (ASCENDING: so the top indices are 4,3,2,
  // 1) will be requested and stored in the first NEIGEN elements of the 
  // EigenVals vector (first col/row of matrix) and the corresponding eigenvectors 
  // will be in the first NEIGEN cols of the EigenVects matrix.
  const int    whichEigen  =   2;  // given by indices below
  const DTYPE  minEigenVal = 1.0;
  const DTYPE  maxEigenVal = 3.0; 
  const bool   isUploA     = false; // lower triangular of A is filled

  
  // On the HOST
  // row-major matrix order (only in case of CBLAS WRAPPER: MKL-BLAS, Open-BLAS)
//  Matrix<DTYPE, false> A(N, N);
//  Matrix<DTYPE, false> EigenVals(1, N);
//  Matrix<DTYPE, false> EigenVects(N, NEIGEN);

  Matrix<DTYPE> A(N, N);
  Matrix<DTYPE> EigenVals(N, 1);
  Matrix<DTYPE> EigenVects(N, NEIGEN);

#if defined(USE_CUBLAS) && defined(ON_GPU)  
  //GPU: On the DEVICE (strictly col.-major order)
  Matrix<DTYPE> A_d(N, N);
  Matrix<DTYPE> EigenVals_d(N, 1);
  Matrix<DTYPE> EigenVects_d(N, NEIGEN);
#endif

  // for fData memory managmenet and BLAS routines
  BLAS theBlas;
  theBlas.SetNumThreads(1);
#if defined(USE_CUBLAS) && defined(ON_GPU)  
  //GPU: for Matrix::fData memory managmenet and BLAS routines on the GPU
  BLAS_gpu  theBlas_gpu;
#endif

  // allocate memory on the HOST 
  theBlas.Calloc(A);           //symmetric => only lower triangular will be filled
  theBlas.Malloc(EigenVals);
  theBlas.Malloc(EigenVects);
#if defined(USE_CUBLAS) && defined(ON_GPU)  
  #pragma message("-------- USING cuBLAS ----")
  // allocate memory on the DEVICE
  theBlas_gpu.Malloc(A_d);
  theBlas_gpu.Malloc(EigenVals_d);
  theBlas_gpu.Malloc(EigenVects_d);
#endif
  
  
  // NOTE: default value of isUploA=true => BUT here we will fill the lower part.
  // fill A: only the lower triangular part
  for (size_t ic=0; ic<N; ++ic) {
    for (size_t ir=ic; ir<N; ++ir) {
      A.SetElem(ir, ic, INPM_A[ir][ic]);
    }
  }

  printf (" Top left corner of matrix A: \n"); 
  for (size_t ir=0; ir<min(A.GetNumRows(),7); ++ir) {
      for (size_t ic=0; ic<min(A.GetNumCols(),7); ++ic) printf ("%12.2E", A.GetElem(ir,ic));
      printf ("\n"); 
  }


#if defined(USE_CUBLAS) && defined(ON_GPU)  
  //GPU: copy the A matrix to the GPU
  theBlas_gpu.CopyToGPU( A, A_d);
#endif
  
  
  size_t numEigenFound = 0;
  
  double duration;
  struct timeval start,finish;  
  gettimeofday(&start, NULL); 

#if defined(USE_CUBLAS) && defined(ON_GPU) 
        printf("\n ---- XSYEVDX (cuSOLVER: eigenvalue/vector of A sym.) starts on GPU\n");
        numEigenFound = theBlas_gpu.XSYEVDX(A_d, EigenVals_d, EigenVects_d, whichEigen, minEigenVal, maxEigenVal, isUploA);
#else
        printf("\n ---- XSYEVR (LAPACK-SOLVER: eigenvalue/vector of A sym.) starts on CPU\n");
        numEigenFound = theBlas.XSYEVR(A, EigenVals, EigenVects, whichEigen, minEigenVal, maxEigenVal, isUploA);
        //numEigenFound = theBlas.XSYEVR(A, EigenVals);
#endif

  gettimeofday(&finish, NULL); 
  duration = ((double)(finish.tv_sec-start.tv_sec)*1000000 + (double)(finish.tv_usec-start.tv_usec)) / 1000000;
  printf("\n\n == TIMING: %lf [s] \n\n",  duration);

#if defined(USE_CUBLAS) && defined(ON_GPU)
  //GPU: copy back the resulted Eiegnevalues and Eigenvectors
  theBlas_gpu.CopyFromGPU( EigenVals_d,  EigenVals);
  theBlas_gpu.CopyFromGPU(EigenVects_d, EigenVects);
#endif
  
  
  // Print results:
  // eigenvalues 
  DTYPE* eigenValues = EigenVals.GetPtrToBlock(0);
  printf("\n The [%lu, %lu] eigenvalues of matrix A: \n", (size_t)minEigenVal, (size_t)maxEigenVal); 
  for (size_t ie=0; ie<min(numEigenFound,7); ++ie) {
    printf ("%12.2E", eigenValues[ie]);
  }
  // eigenvectors (top left corner)  
  printf("\n The corresponding eigenvectors of matrix A: \n"); 
  for (size_t ir=0; ir<min(EigenVects.GetNumRows(),7); ++ir) {
    for (size_t ic=0; ic<min(EigenVects.GetNumCols(),7); ++ic) printf ("%12.2E", EigenVects.GetElem(ir,ic));
    printf ("\n"); 
  }


  // free allocated memory
  theBlas.Free(A);
  theBlas.Free(EigenVals);
  theBlas.Free(EigenVects);

#if defined(USE_CUBLAS) && defined(ON_GPU)
  //GPU: free memory allocated on the GPU
  theBlas_gpu.Free(A_d);
  theBlas_gpu.Free(EigenVals_d);
  theBlas_gpu.Free(EigenVects_d);
#endif
  
  return 0;
}
