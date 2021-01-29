
#include <iostream>

#include <stdlib.h>
#include <stdio.h>

#include "sys/time.h"
#include "time.h"

#include "types.hh"
#include "Matrix.hh"

#define min(x,y) (((x) < (y)) ? (x) : (y))

//#define m  25000
//#define n  15000
//#define k  1000

#define m  1000
#define n  1000
#define k  1000

// Define data type for the test(double or float)
//typedef double DTYPE;
typedef float DTYPE;

int main() {
  
  // On the HOST
//  Matrix<DTYPE,false> A(m,k);
//  Matrix<DTYPE,false> B(k,n);
//  Matrix<DTYPE,false> C(m,n);
  Matrix<DTYPE> A(m,k);
  Matrix<DTYPE> B(k,n);
  Matrix<DTYPE> C(m,n);


#if defined(USE_CUBLAS) && defined(ON_GPU)  
  //GPU: On the DEVICE
  Matrix<DTYPE> A_d(m,k);
  Matrix<DTYPE> B_d(k,n);
  Matrix<DTYPE> C_d(m,n);
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
  theBlas.Malloc(B);
  theBlas.Calloc(C);
#if defined(USE_CUBLAS) && defined(ON_GPU) && CONFIG_VERBOSE
  #pragma message("-------- USING cuBLAS ----")
  // allocate memory on the DEVICE
  theBlas_gpu.Malloc(A_d);
  theBlas_gpu.Malloc(B_d);
  theBlas_gpu.Calloc(C_d);
#endif
  
    
  // fill A

  const size_t nRowsA = A.GetNumRows();
  const size_t nColsA = A.GetNumCols();
  for (size_t ir = 0; ir < nRowsA; ++ir) {
    for (size_t ic = 0; ic < nColsA; ++ic) {
      A.SetElem(ir, ic, (DTYPE)(ir*nColsA+ic+1));
    }
  }
  // fill B
  const size_t nRowsB = B.GetNumRows();
  const size_t nColsB = B.GetNumCols();
  for (size_t ir = 0; ir < nRowsB; ++ir) {
    for (size_t ic = 0; ic < nColsB; ++ic) {
      B.SetElem(ir, ic, -(DTYPE)(ir*nColsA+ic+1));
    }
  }  

#if defined(USE_CUBLAS) && defined(ON_GPU)  
  //GPU: copy the fData array content of A, B, C to A_d, B_d and C_d (even if 
  //     C is a zero matrix now just to keep generality) 
  theBlas_gpu.CopyToGPU(A,A_d);
  theBlas_gpu.CopyToGPU(B,B_d);
  theBlas_gpu.CopyToGPU(C,C_d);
#endif

  double duration;
  struct timeval start,finish;  
  gettimeofday(&start, NULL); 

  printf("\n ---- XGEMM (Level 3 BLAS general matrix-matrix multiplication: C = \\alpha A^T B^T + \\beta C) starts on ");     
#if defined(USE_CUBLAS) && defined(ON_GPU) 
         printf(" GPU!\n");
        //GPU: invoke XGEMM (X={s=float,g=double}) : C = alpha A^T B^T + beta C 
        theBlas_gpu.XGEMM(A_d,B_d,C_d);
#else
        printf(" CPU!\n"); 
        //CPU: invoke XGEMM (X={s=float,g=double}) : C = alpha A^T B^T + beta C
        theBlas.XGEMM(A,B,C);
#endif

  gettimeofday(&finish, NULL); 
  duration = ((double)(finish.tv_sec-start.tv_sec)*1000000 + (double)(finish.tv_usec-start.tv_usec)) / 1000000;
  double gflops = 2.0 * m *n*k;
  gflops = gflops/duration*1.0e-6;
  printf("\n\n == TIMING: %dx%dx%d\t%lf s\t%lf MFLOPS \n\n", m, n, k, duration, gflops);
  
#if defined(USE_CUBLAS) && defined(ON_GPU)
  //GPU: copy back the resulted fData array of matrix C from GPU to the host
  theBlas_gpu.CopyFromGPU(C_d,C);
#endif
  
  printf (" Top left corner of matrix A: \n"); 
  for (size_t ir=0; ir<min(A.GetNumRows(),6); ++ir) {
      for (size_t ic=0; ic<min(A.GetNumCols(),6); ++ic) printf ("%12.0f", A.GetElem(ir,ic));
      printf ("\n"); 
  }

  printf ("\n Top left corner of matrix B: \n"); 
  for (size_t ir=0; ir<min(B.GetNumRows(),6); ++ir) {
      for (size_t ic=0; ic<min(B.GetNumCols(),6); ++ic) printf ("%12.0f", B.GetElem(ir,ic));
      printf ("\n"); 
  }
  
  printf ("\n Top left corner of matrix C: \n"); 
  for (size_t ir=0; ir<min(C.GetNumRows(),6); ++ir) {
      for (size_t ic=0; ic<min(C.GetNumCols(),6); ++ic) printf ("%15.5E", C.GetElem(ir,ic));
      printf ("\n"); 
  }
  

  // free allocated memory
  theBlas.Free(A);
  theBlas.Free(B);
  theBlas.Free(C);

#if defined(USE_CUBLAS) && defined(ON_GPU)
  //GPU: free memory allocated on the GPU
  theBlas_gpu.Free(A_d);
  theBlas_gpu.Free(B_d);
  theBlas_gpu.Free(C_d);
#endif

  return 0;
}