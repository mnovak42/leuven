#include <iostream>

#include <stdlib.h>
#include <stdio.h>

#include "sys/time.h"
#include "time.h"

#include "types.hh"
#include "Matrix.hh"

#define min(x,y) (((x) < (y)) ? (x) : (y))

#define M 6
#define N 2

// Define data type for the test(double or float)
typedef double DTYPE;
//typedef float DTYPE;


/*
  LAPACKE_dgeqrf Example.
  =======================

  Program computes the QR decomposition of a general rectangular matrix
  A M-by-N:  A = QR

    0.00   2.00
    2.00  -1.00
    2.00  -1.00
    0.00   1.50
    2.00  -1.00
    2.00  -1.00

  Example Program Results:
  ========================

  At output the matrix A will be overwritten to contain R and Q (in a form):
  - on and above the diagonal of A contain the min(M,N)-by-N upper trapezoidal
    matrix R (R is upper triangular if M>=N that is our case)
  - below the diagonal, with the vector TAU, contain the orthogonal matrix Q as
    a product of min(M,N) elementary reflectors (coefs in TAU)

  A:

  -4.000000   2.000000
   0.500000   2.500000
   0.500000   0.285714
   0.000000  -0.428571
   0.500000   0.285714
   0.500000   0.285714

  TAU:

   1.000000  1.400000
  ------------------------------------------------------------------------------

  Example2: with M = 4  < N = 5

            |   .500000   .500000  1.207107   .000000  1.707107 |
    A    =  |   .500000 -1.500000  -.500000  2.414214   .707107 |
            |   .500000   .500000   .207107   .000000   .292893 |
            |   .500000 -1.500000  -.500000  -.414214  -.707107 |


            | -1.000000  1.000000  -.207107 -1.000000  -1.000000 |
    A    =  |   .333333  2.000000  1.207107 -1.000000   1.000000 |
            |   .333333  -.200000   .707107   .000000   1.000000 |
            |   .333333   .400000   .071068 -2.000000  -1.000000 |

    TAU  =  |  1.500000  1.666667  1.989949   .000000 |
 */


 int main() {
   // On the HOST
   // row-major matrix order (only in case of CBLAS WRAPPER: MKL-BLAS, Open-BLAS)
 //  Matrix<DTYPE,false> A(M,N);
 //  Matrix<DTYPE,false> TAU(min(M,N),1);

   Matrix<DTYPE> A(M,N);
   Matrix<DTYPE> TAU(min(M,N),1);

#if defined(USE_CUBLAS) && defined(ON_GPU)
   //GPU: On the DEVICE (strictly col.-major order)
   Matrix<DTYPE> A_d(M,N);
   Matrix<DTYPE> TAU_d(min(M,N),1);
#endif

   // for fData memory managmenet and BLAS routines
   BLAS theBlas;
   theBlas.SetNumThreads(1);
#if defined(USE_CUBLAS) && defined(ON_GPU)
   //GPU: for Matrix::fData memory managmenet and BLAS routines on the GPU
   BLAS_gpu  theBlas_gpu;
#endif

   // allocate memory on the HOST
   theBlas.Malloc(  A);
   theBlas.Malloc(TAU);
#if defined(USE_CUBLAS) && defined(ON_GPU)
#if CONFIG_VERBOSE
     #pragma message("-------- USING cuBLAS ----")
#endif
   // allocate memory on the DEVICE
   theBlas_gpu.Malloc(  A_d);
   theBlas_gpu.Malloc(TAU_d);
#endif


   // fill A:
   // 0.00   2.00
   // 2.00  -1.00
   // 2.00  -1.00
   // 0.00   1.50
   // 2.00  -1.00
   // 2.00  -1.00
   A.SetElem(0,0, 0.0); A.SetElem(0,1, 2.0);
   A.SetElem(1,0, 2.0); A.SetElem(1,1,-1.0);
   A.SetElem(2,0, 2.0); A.SetElem(2,1,-1.0);
   A.SetElem(3,0, 0.0); A.SetElem(3,1, 1.5);
   A.SetElem(4,0, 2.0); A.SetElem(4,1,-1.0);
   A.SetElem(5,0, 2.0); A.SetElem(5,1,-1.0);

   printf (" Top left corner of matrix A: \n");
   for (size_t ir=0; ir<min(A.GetNumRows(),7); ++ir) {
       for (size_t ic=0; ic<min(A.GetNumCols(),7); ++ic) printf ("%12.2E", A.GetElem(ir,ic));
       printf ("\n");
   }


 #if defined(USE_CUBLAS) && defined(ON_GPU)
   //GPU: copy the fData array content of A,U,VT and SIGMA to the GPU
   theBlas_gpu.CopyToGPU(   A,   A_d);
   theBlas_gpu.CopyToGPU( TAU, TAU_d);
 #endif

   // Compute the QR factorization of the A, M x N marix: since M<N, the diagonal
   //  and above elements will contain the N x N upper-triangular matrix R

   double duration;
   struct timeval start,finish;
   gettimeofday(&start, NULL);

   printf("\n ---- XDGEQRF (LAPACK-QR: A = QR starts on ");
#if defined(USE_CUBLAS) && defined(ON_GPU)
         printf(" GPU!\n");
         theBlas_gpu.XGEQRF(A_d, TAU_d);
#else
         printf(" CPU!\n");
         theBlas.XGEQRF(A, TAU);
#endif

   gettimeofday(&finish, NULL);
   duration = ((double)(finish.tv_sec-start.tv_sec)*1000000 + (double)(finish.tv_usec-start.tv_usec)) / 1000000;
   printf("\n\n == TIMING: %lf [s] \n\n",  duration);

#if defined(USE_CUBLAS) && defined(ON_GPU)
   //GPU: copy back the resulted fData arraies from GPU to the host SIGMA, U, VT
   theBlas_gpu.CopyFromGPU(  A_d,  A);
   theBlas_gpu.CopyFromGPU(TAU_d,TAU);
#endif


   // Print results:

   // input matrix A (M,N) (overwritten)
   printf (" Top left corner of matrix A: \n");
   for (size_t ir=0; ir<min(A.GetNumRows(),7); ++ir) {
       for (size_t ic=0; ic<min(A.GetNumCols(),7); ++ic) printf ("%15.6E", A.GetElem(ir,ic));
       printf ("\n");
   }

   printf ("\n TAU (vector) matrix: \n");
   for (size_t ir=0; ir<min(TAU.GetNumRows(),7); ++ir) {
     for (size_t ic=0; ic<min(TAU.GetNumCols(),7); ++ic) printf ("%15.6E", TAU.GetElem(ir,ic));
     printf ("\n");
   }

   //
   // An extra part to check XORMQR by performing the multiplication of QR->A
   // 1. get matrix R from A: on and upper diagonal elements of A after the QR
   // 2. perform the multiplication QR (from the left, no-transpose)

   // 1. R is an min(M,N)-by-N matrix. Q is a M-by-min(M,N) matrix (not formed)
   //    When we perform the multiplictaion, the result is QR -> M-by-N matrix
   //    such that R is overwritten: so make GetBackA to be M-by-N and store the
   //    R matrix on its upper diagonal (not the most efficient but it's testing)

   Matrix<DTYPE> GetBackA(M,N);
   theBlas.Calloc(GetBackA);
   for (size_t ir=0; ir<min(M,N); ++ir)
     for (size_t ic=ir; ic<N; ++ic) {
       GetBackA.SetElem(ir, ic, A.GetElem(ir, ic));
     }
   // 2. perform the multiplication QR (from the left, no-transpose)
   theBlas.XORMQR(GetBackA, A, TAU);

   // input matrix GetBackA (M,N) (overwritten by XORMQR) that must be the as A
   printf ("\n Top left corner of matrix GetBackA (must be same as A = QR): \n");
   for (size_t ir=0; ir<min(GetBackA.GetNumRows(),7); ++ir) {
       for (size_t ic=0; ic<min(GetBackA.GetNumCols(),7); ++ic) printf ("%15.6E", GetBackA.GetElem(ir,ic));
       printf ("\n");
   }

   // free allocated memory
   theBlas.Free(A);
   theBlas.Free(TAU);
   theBlas.Free(GetBackA);

 #if defined(USE_CUBLAS) && defined(ON_GPU)
   //GPU: free memory allocated on the GPU
   theBlas_gpu.Free(A_d);
   theBlas_gpu.Free(TAU_d);
 #endif

   return 0;
 }
