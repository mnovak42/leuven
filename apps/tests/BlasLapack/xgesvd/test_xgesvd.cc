
#include <iostream>

#include <stdlib.h>
#include <stdio.h>

#include "sys/time.h"
#include "time.h"

#include "types.hh"
#include "Matrix.hh"

#define min(x,y) (((x) < (y)) ? (x) : (y))

#define M 6
#define N 5

// Define data type for the test(double or float)
typedef double DTYPE;
//typedef float DTYPE;


/*
  LAPACKE_dgesvd Example.
  =======================

  Program computes the singular value decomposition of a general
  rectangular matrix A M-by-N:

    8.79   9.93   9.83   5.45   3.16
    6.11   6.91   5.04  -0.27   7.98
   -9.15  -7.93   4.86   4.85   3.01
    9.57   1.64   8.83   0.74   5.80
   -3.49   4.02   9.80  10.00   4.27
    9.84   0.15  -8.99  -6.02  -5.31

  Example Program Results: 
  ========================

   
   LAPACKE_dgesvd or LAPACKE_sgesvd (row/col-major) Example Program Results: 
   with JOBU='S' and JOBVT='S' i.e. the min(M,N) left and right singular vectors 
   are computed and returned as the min(M,N) cols of U (M-by-M) and the min(M,N)
   cols of V (N-by-N). Note, that not V but V^T is computed so actually these 
   min(M,N) cols of V are min(M,N) rows of the computed V^T.

   Singular values: min(M,N)
    27.47  22.64   8.56   5.99   2.01

   Left singular vectors (stored columnwise):
    -0.59   0.26   0.36   0.31   0.23
    -0.40   0.24  -0.22  -0.75  -0.36
    -0.03  -0.60  -0.45   0.23  -0.31
    -0.43   0.24  -0.69   0.33   0.16
    -0.47  -0.35   0.39   0.16  -0.52
     0.29   0.58  -0.02   0.38  -0.65

   Right singular vectors (stored rowwise):  
    -0.25  -0.40  -0.69  -0.37  -0.41
     0.81   0.36  -0.25  -0.37  -0.10
    -0.26   0.70  -0.22   0.39  -0.49
     0.40  -0.45   0.25   0.43  -0.62
    -0.22   0.14   0.59  -0.63  -0.44
 */
 

int main() {
  // On the HOST
  // row-major matrix order (only in case of CBLAS WRAPPER: MKL-BLAS, Open-BLAS)
//  Matrix<DTYPE,false> A(M,N);
//  Matrix<DTYPE,false> SIGMA(min(M,N),1);
//  Matrix<DTYPE,false> U(M,M);
//  Matrix<DTYPE,false> VT(N,N);

  Matrix<DTYPE> A(M,N);
  Matrix<DTYPE> SIGMA(min(M,N),1);
  Matrix<DTYPE> U(M,M);
  Matrix<DTYPE> VT(N,N);


#if defined(USE_CUBLAS) && defined(ON_GPU)  
  //GPU: On the DEVICE (strictly col.-major order)
  Matrix<DTYPE> A_d(M,N);
  Matrix<DTYPE> SIGMA_d(min(M,N),1);
  Matrix<DTYPE> U_d(M,M);
  Matrix<DTYPE> VT_d(N,N);
#endif

  // for fData memory managmenet and BLAS routines
  BLAS theBlas;
  theBlas.SetNumThreads(1);
  #if defined(USE_CUBLAS) && defined(ON_GPU)  
  //GPU: for Matrix::fData memory managmenet and BLAS routines on the GPU
  BLAS_gpu  theBlas_gpu;
  #endif

  // allocate memory on the HOST
  theBlas.Malloc(A);
  theBlas.Calloc(SIGMA);
  theBlas.Calloc(U);
  theBlas.Calloc(VT);
  #if defined(USE_CUBLAS) && defined(ON_GPU) && CONFIG_VERBOSE
    #pragma message("-------- USING cuBLAS ----")
  // allocate memory on the DEVICE
  theBlas_gpu.Malloc(A_d);
  theBlas_gpu.Calloc(SIGMA_d);
  theBlas_gpu.Calloc(U_d);
  theBlas_gpu.Calloc(VT_d);
  #endif
  
  
  // fill A:
  //  8.79   9.93   9.83   5.45   3.16
  //  6.11   6.91   5.04  -0.27   7.98
  // -9.15  -7.93   4.86   4.85   3.01
  //  9.57   1.64   8.83   0.74   5.80
  // -3.49   4.02   9.80  10.00   4.27
  //  9.84   0.15  -8.99  -6.02  -5.31
  A.SetElem(0,0, 8.79); A.SetElem(0,1, 9.93); A.SetElem(0,2, 9.83); A.SetElem(0,3, 5.45); A.SetElem(0,4, 3.16);
  A.SetElem(1,0, 6.11); A.SetElem(1,1, 6.91); A.SetElem(1,2, 5.04); A.SetElem(1,3,-0.27); A.SetElem(1,4, 7.98);
  A.SetElem(2,0,-9.15); A.SetElem(2,1,-7.93); A.SetElem(2,2, 4.86); A.SetElem(2,3, 4.85); A.SetElem(2,4, 3.01);
  A.SetElem(3,0, 9.57); A.SetElem(3,1, 1.64); A.SetElem(3,2, 8.83); A.SetElem(3,3, 0.74); A.SetElem(3,4, 5.80);
  A.SetElem(4,0,-3.49); A.SetElem(4,1, 4.02); A.SetElem(4,2, 9.80); A.SetElem(4,3,10.00); A.SetElem(4,4, 4.27);
  A.SetElem(5,0, 9.84); A.SetElem(5,1, 0.15); A.SetElem(5,2,-8.99); A.SetElem(5,3,-6.02); A.SetElem(5,4,-5.31);

  printf (" Top left corner of matrix A: \n"); 
  for (size_t ir=0; ir<min(A.GetNumRows(),7); ++ir) {
      for (size_t ic=0; ic<min(A.GetNumCols(),7); ++ic) printf ("%12.2E", A.GetElem(ir,ic));
      printf ("\n"); 
  }


#if defined(USE_CUBLAS) && defined(ON_GPU)  
  //GPU: copy the fData array content of A,U,VT and SIGMA to the GPU
  theBlas_gpu.CopyToGPU( A, A_d);
  theBlas_gpu.CopyToGPU( U, U_d);
  theBlas_gpu.CopyToGPU(VT, VT_d);
  theBlas_gpu.CopyToGPU(SIGMA, SIGMA_d);
#endif
  
  // NOTE: possible values for JOBU and JOBVT = `A`, `S`, `O` and `N` see doc.
  //
  // Compute the min(M,N) left and right signular vectors and store as cols of 
  // matrices U and V (note that V^T is returned)
  char JOBU  = 'S';
  char JOBVT = 'S';
  
  double duration;
  struct timeval start,finish;  
  gettimeofday(&start, NULL); 

  printf("\n ---- XDGESVD (LAPACK-SVD: A = U SIGMA V^T starts on ");
  #if defined(USE_CUBLAS) && defined(ON_GPU) 
        printf(" GPU!\n");
        theBlas_gpu.XGESVD(A_d, SIGMA_d, &JOBU, &JOBVT, U_d, VT_d);
  #else
        printf(" CPU!\n"); 
        theBlas.XGESVD(A, SIGMA, &JOBU, &JOBVT, U, VT);
  #endif

  gettimeofday(&finish, NULL); 
  duration = ((double)(finish.tv_sec-start.tv_sec)*1000000 + (double)(finish.tv_usec-start.tv_usec)) / 1000000;
  printf("\n\n == TIMING: %lf [s] \n\n",  duration);

  #if defined(USE_CUBLAS) && defined(ON_GPU)
  //GPU: copy back the resulted fData arraies from GPU to the host SIGMA, U, VT
  theBlas_gpu.CopyFromGPU(SIGMA_d,SIGMA);
  theBlas_gpu.CopyFromGPU(U_d,U);
  theBlas_gpu.CopyFromGPU(VT_d,VT);
  #endif
  
  
  // Print results:
  
  // input matrix A (M,N): DISTROYED 
//  printf (" Top left corner of matrix A: \n"); 
//  for (size_t ir=0; ir<min(A.GetNumRows(),7); ++ir) {
//      for (size_t ic=0; ic<min(A.GetNumCols(),7); ++ic) printf ("%12.2E", A.GetElem(ir,ic));
//      printf ("\n"); 
//  }

  printf ("\n SIGMA (diagonal) matrix: \n"); 
  for (size_t ir=0; ir<min(SIGMA.GetNumRows(),7); ++ir) {
    for (size_t ic=0; ic<min(SIGMA.GetNumCols(),7); ++ic) printf ("%12.3E", SIGMA.GetElem(ir,0));
    printf ("\n"); 
  }

  printf ("\n Top left corner of matrix U: \n"); 
  for (size_t ir=0; ir<min(U.GetNumRows(),7); ++ir) {
      for (size_t ic=0; ic<min(U.GetNumCols(),7); ++ic) printf ("%12.2E", U.GetElem(ir,ic));
      printf ("\n"); 
  }
  
  printf ("\n Top left corner of matrix VT: \n"); 
  for (size_t ir=0; ir<min(VT.GetNumRows(),7); ++ir) {
      for (size_t ic=0; ic<min(VT.GetNumCols(),7); ++ic) printf ("%12.2E", VT.GetElem(ir,ic));
      printf ("\n"); 
  }


  // free allocated memory
  theBlas.Free(A);
  theBlas.Free(SIGMA);
  theBlas.Free(U);
  theBlas.Free(VT);

#if defined(USE_CUBLAS) && defined(ON_GPU)
  //GPU: free memory allocated on the GPU
  theBlas_gpu.Free(A_d);
  theBlas_gpu.Free(SIGMA_d);
  theBlas_gpu.Free(U_d);
  theBlas_gpu.Free(VT_d);
#endif
  
  return 0;
}