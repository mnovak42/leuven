
#include <iostream>

#include <stdlib.h>
#include <stdio.h>

#include "sys/time.h"
#include "time.h"

#include "types.hh"
#include "Matrix.hh"

#define min(x,y) (((x) < (y)) ? (x) : (y))

#define N 5
#define NRHS 3

// Define data type for the test(double or float)
typedef double DTYPE;
//typedef float DTYPE;


/*
  LAPACKE_dsysv Example.
  =======================

  The program computes the solution to the system of linear equations
  with a real symmetric matrix A and multiple right-hand sides B,
  where A is the coefficient matrix:

    -5.86   3.99  -5.93  -2.82   7.69
     3.99   4.46   2.58   4.42   4.61
    -5.93   2.58  -8.52   8.57   7.69
    -2.82   4.42   8.57   3.72   8.07
     7.69   4.61   7.69   8.07   9.83

  and B is the right-hand side matrix:

     1.32  -6.33  -8.77
     2.22   1.69  -8.33
     0.12  -1.56   9.54
    -6.41  -9.49   9.56
     6.33  -3.67   7.48

  Example Program Results:
  ========================

   Solution
     1.17   0.52  -0.86
    -0.71   1.05  -4.90
    -0.63  -0.52   0.99
    -0.33   0.43   1.22
     0.83  -1.22   1.96

 */


int main() {
  // On the HOST
  // row-major matrix order (only in case of CBLAS WRAPPER: MKL-BLAS, Open-BLAS)
//  Matrix<DTYPE,false> A(M,N);
//  Matrix<DTYPE,false> SIGMA(min(M,N),1);
//  Matrix<DTYPE,false> U(M,M);
//  Matrix<DTYPE,false> VT(N,N);

  Matrix<DTYPE> A(N,N);
  Matrix<DTYPE> B(N,NRHS);

#if defined(USE_CUBLAS) && defined(ON_GPU)
  //GPU: On the DEVICE (strictly col.-major order)
  Matrix<DTYPE> A_d(N,N);
  Matrix<DTYPE> B_d(N,NRHS);
#endif

  // for fData memory managmenet and BLAS routines
  BLAS theBlas;
  theBlas.SetNumThreads(1);
#if defined(USE_CUBLAS) && defined(ON_GPU)
  //GPU: for Matrix::fData memory managmenet and BLAS routines on the GPU
  BLAS_gpu  theBlas_gpu;
#endif

  // allocate memory on the HOST (CPU version is for symetric GPU version is for
  // squared matrix A)
  theBlas.Calloc(A);
  theBlas.Malloc(B);
#if defined(USE_CUBLAS) && defined(ON_GPU)
#if CONFIG_VERBOSE
  #pragma message("-------- USING cuBLAS ----")
#endif
  // allocate memory on the DEVICE
  theBlas_gpu.Malloc(A_d);
  theBlas_gpu.Malloc(B_d);
#endif


  // NOTE: default value of isUplo=true => upper triangular of A is filled.
  // fill A: only the upper triangle
  // -5.86   3.99  -5.93  -2.82   7.69
  //  3.99   4.46   2.58   4.42   4.61
  // -5.93   2.58  -8.52   8.57   7.69
  // -2.82   4.42   8.57   3.72   8.07
  //  7.69   4.61   7.69   8.07   9.83
  A.SetElem(0,0,-5.86); A.SetElem(0,1, 3.99); A.SetElem(0,2,-5.93); A.SetElem(0,3,-2.82); A.SetElem(0,4, 7.69);
  A.SetElem(1,1, 4.46); A.SetElem(1,2, 2.58); A.SetElem(1,3, 4.42); A.SetElem(1,4, 4.61);
  A.SetElem(2,2,-8.52); A.SetElem(2,3, 8.57); A.SetElem(2,4, 7.69);
  A.SetElem(3,3, 3.72); A.SetElem(3,4, 8.07);
  A.SetElem(4,4, 9.83);
  // symmetric part is filled so complete in case of GPU by filling the lower part
#if defined(USE_CUBLAS) && defined(ON_GPU)
  for (size_t ir=0; ir<A.GetNumRows(); ++ir)
      for (size_t ic=0; ic<ir; ++ic)
         A.SetElem(ir,ic,A.GetElem(ic,ir));
#endif

  // fill B:
  //  1.32  -6.33  -8.77
  //  2.22   1.69  -8.33
  //  0.12  -1.56   9.54
  // -6.41  -9.49   9.56
  //  6.33  -3.67   7.48
  B.SetElem(0,0, 1.32); B.SetElem(0,1,-6.33); B.SetElem(0,2,-8.77);
  B.SetElem(1,0, 2.22); B.SetElem(1,1, 1.69); B.SetElem(1,2,-8.33);
  B.SetElem(2,0, 0.12); B.SetElem(2,1,-1.56); B.SetElem(2,2, 9.54);
  B.SetElem(3,0,-6.41); B.SetElem(3,1,-9.49); B.SetElem(3,2, 9.56);
  B.SetElem(4,0, 6.33); B.SetElem(4,1,-3.67); B.SetElem(4,2, 7.48);


  printf (" Top left corner of matrix A: \n");
  for (size_t ir=0; ir<min(A.GetNumRows(),7); ++ir) {
      for (size_t ic=0; ic<min(A.GetNumCols(),7); ++ic) printf ("%12.2E", A.GetElem(ir,ic));
      printf ("\n");
  }

  printf (" Top left corner of matrix B: \n");
  for (size_t ir=0; ir<min(B.GetNumRows(),7); ++ir) {
      for (size_t ic=0; ic<min(B.GetNumCols(),7); ++ic) printf ("%12.2E", B.GetElem(ir,ic));
      printf ("\n");
  }


#if defined(USE_CUBLAS) && defined(ON_GPU)
  //GPU: copy the A and B matrix to the GPU
  theBlas_gpu.CopyToGPU( A, A_d);
  theBlas_gpu.CopyToGPU( B, B_d);
#endif

  double duration;
  struct timeval start,finish;
  gettimeofday(&start, NULL);

#if defined(USE_CUBLAS) && defined(ON_GPU)
        printf("\n ---- XGESV ( ?GETRF+?GETRS LAPACK-SOLVER: AX = B) starts on GPU\n");
        theBlas_gpu.XGESV(A_d, B_d);
#else
        printf("\n ---- XSYSV (LAPACK-SOLVER: AX = B) starts on CPU\n");
        theBlas.XSYSV(A, B);
#endif

  gettimeofday(&finish, NULL);
  duration = ((double)(finish.tv_sec-start.tv_sec)*1000000 + (double)(finish.tv_usec-start.tv_usec)) / 1000000;
  printf("\n\n == TIMING: %lf [s] \n\n",  duration);

#if defined(USE_CUBLAS) && defined(ON_GPU)
  //GPU: copy back the resulted X (stored in B; A is destroyed)
  theBlas_gpu.CopyFromGPU(B_d, B);
#endif


  // Print results:

  // input matrix A (N,N): DISTROYED (overwriten)
//  printf (" Top left corner of matrix A: \n");
//  for (size_t ir=0; ir<min(A.GetNumRows(),7); ++ir) {
//      for (size_t ic=0; ic<min(A.GetNumCols(),7); ++ic) printf ("%12.2E", A.GetElem(ir,ic));
//      printf ("\n");
//  }

  // B is overwritten by the solution X (N,NRHS)
  printf ("\n B (solution X such that AX=B) matrix: \n");
  for (size_t ir=0; ir<min(B.GetNumRows(),7); ++ir) {
    for (size_t ic=0; ic<min(B.GetNumCols(),7); ++ic) printf ("%12.2E", B.GetElem(ir,ic));
    printf ("\n");
  }


  // free allocated memory
  theBlas.Free(A);
  theBlas.Free(B);

#if defined(USE_CUBLAS) && defined(ON_GPU)
  //GPU: free memory allocated on the GPU
  theBlas_gpu.Free(A_d);
  theBlas_gpu.Free(B_d);
#endif

  return 0;
}
