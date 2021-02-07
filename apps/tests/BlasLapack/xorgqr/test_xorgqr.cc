#include <iostream>

#include <stdlib.h>
#include <stdio.h>

#include "sys/time.h"
#include "time.h"

#include "types.hh"
#include "Matrix.hh"

#if defined(USE_CUBLAS) && defined(ON_GPU)
#include "cuKers.h"
#endif

#define min(x,y) (((x) < (y)) ? (x) : (y))

#define M 6
#define N 2

// Define data type for the test(double or float)
typedef double DTYPE;
//typedef float DTYPE;


//
// ?orgqr can be used to form the matrix Q after a A=QR decomposition of a matrix
//  A: after QR-decomposition everything is stored in the original A and a Tau
//     vector. These can be used to explicitely form the Q matrix.

// 1. matrix A is QR-decomposed here by using XGEQRF
// 2. the upper triangular matrix R is formed from the resluted A
// 3. the matrix Q is explicitely formed by using XORGQR using the resulted
//    matrix A and vector Tau
// 4. the QxR multiplications is computed to see if we get back A

/*
1.
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

   Matrix<DTYPE> R(min(M,N), min(M,N));
   Matrix<DTYPE> ResA(M,N); // A as QxR


#if defined(USE_CUBLAS) && defined(ON_GPU)
   //GPU: On the DEVICE (strictly col.-major order)
   Matrix<DTYPE> A_d(M,N);
   Matrix<DTYPE> TAU_d(min(M,N),1);

   Matrix<DTYPE> R_d(min(M,N), min(M,N));
   Matrix<DTYPE> ResA_d(M,N);
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

   theBlas.Calloc(  R);
   theBlas.Malloc(ResA);

#if defined(USE_CUBLAS) && defined(ON_GPU)
#if CONFIG_VERBOSE
  #pragma message("-------- USING cuBLAS ----")
#endif
   // allocate memory on the DEVICE
   theBlas_gpu.Malloc(  A_d);
   theBlas_gpu.Malloc(TAU_d);

   theBlas_gpu.Calloc(  R_d);
   theBlas_gpu.Malloc(ResA_d);
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
   //GPU: copy the fData array content of A
   theBlas_gpu.CopyToGPU(   A,   A_d);
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
         // obtain the matrix R ( R supposed to have the size of min(M,N) where A is MxN)
         GetUpperTriangular(A_d.GetDataPtr(), R_d.GetDataPtr(), A_d.GetNumRows(), A_d.GetNumCols());
         // form the matrix Q explicitely (Q will be formed into A)
         theBlas_gpu.XORGQR(A_d, TAU_d); // Q -> A
         // compute the original matrix A by QxR
         theBlas_gpu.XGEMM(A_d, R_d, ResA_d, 1.0, 0.0);
         // copy results from device to host
         theBlas_gpu.CopyFromGPU(ResA_d, ResA);
 #else
         printf(" CPU!\n");
         theBlas.XGEQRF(A, TAU);
         // obtain the matrix R
         for (size_t ir=0; ir<min(M,N); ++ir) {
           for (size_t ic=ir; ic<N; ++ic) {
             R.SetElem(ir, ic, A.GetElem(ir, ic));
           }
         }
         // form the matrix Q explicitely (Q will be formed into A)
         theBlas.XORGQR(A, TAU); // Q -> A
         // compute the original matrix A by QxR
         theBlas.XGEMM(A, R, ResA);
 #endif

   gettimeofday(&finish, NULL);
   duration = ((double)(finish.tv_sec-start.tv_sec)*1000000 + (double)(finish.tv_usec-start.tv_usec)) / 1000000;
   printf("\n\n == TIMING: %lf [s] \n\n",  duration);

   // Print results:

   // A as QxR (overwritten)
   printf (" Top left corner of matrix ResA: \n");
   for (size_t ir=0; ir<min(ResA.GetNumRows(),7); ++ir) {
       for (size_t ic=0; ic<min(ResA.GetNumCols(),7); ++ic) printf ("%15.6E", ResA.GetElem(ir,ic));
       printf ("\n");
   }


   // free allocated memory
   theBlas.Free(A);
   theBlas.Free(TAU);
   theBlas.Free(R);
   theBlas.Free(ResA);

#if defined(USE_CUBLAS) && defined(ON_GPU)
   //GPU: free memory allocated on the GPU
   theBlas_gpu.Free(A_d);
   theBlas_gpu.Free(TAU_d);
   theBlas_gpu.Free(R_d);
   theBlas_gpu.Free(ResA_d);
#endif

   return 0;
 }
