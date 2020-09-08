#include <iostream>

#include <stdlib.h>
#include <stdio.h>

#include "sys/time.h"
#include "time.h"

#include "types.hh"
#include "Matrix.hh"

#include <cuda_runtime.h> //only for cuda_status

//
// IT'S A GPU ONLY A=QR factorization AND QR multiplication
//
// Same as test_xgeqrf but the back-subsitution i.e. QR multiplication is also 
// done on the GPU:
// - first the M-by-N matrix A and the min(M,N) Tau vector are created. The QR 
//   factorization is done on GPU by using XGEQRF. Before that, the matrix A is 
//   filled on CPU, transfered to GPU.
//   note: vector TAU would not be really needed on CPU. We have it to print it.   
// - then still on GPU, a matrix GetBackA will be created and filled with the R 
//   matrix i.e. top min(M,N)-by-N of the matrix A after the QR factorization. 
//   A kernel, GetUpperTriangular_2D is created for this step.
// - this GetBackA matrix will be multiplied on GPU by the Q mateirx (from the 
//   left, without transpose i.e. GetBackA = Q x GetBackA = QxR => A i.e. we 
//   should get back the initial matrix A 
// - the GetBackA matrix, that stores the result of the multiplication, is 
//   copied back (with the matrix A and vector TAU just to see them)
//

#define min(x,y) (((x) < (y)) ? (x) : (y))

#define M 6
#define N 2

// Define data type for the test(double or float)
typedef double DTYPE;
//typedef float DTYPE;



// a is an M-by-N matrix with M>=N 
// b is an N-by-N matrix 
// will write the upper triangular of a into the upper triangualr of b
template <class T>
__global__
void GetUpperTriangular_2D(T* a, T* b, int m, int n) {
   int ir= blockIdx.x * blockDim.x + threadIdx.x;
   int ic= blockIdx.y * blockDim.y + threadIdx.y;
   if (ir<m && ic<n && ic>=ir) {
//     printf("%d\%d%\t%lg\n",ir,ic,a[ic*m+ir]);
     int idx = ic*m + ir;
     b[ic*m+ir] = a[ic*m+ir]; 
   }
}



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
 
   Matrix<DTYPE> A(M,N);
   Matrix<DTYPE> GetBackA(M,N);
   Matrix<DTYPE> TAU(min(M,N),1);

   //GPU: On the DEVICE (strictly col.-major order)
   Matrix<DTYPE> A_d(M,N);
   Matrix<DTYPE> TAU_d(min(M,N),1);
   Matrix<DTYPE> GetBackA_d(M,N);

   // CPU BLAS is used only for memory managmenet of the host matrices
   BLAS  theBlas;
   //GPU: for Matrix::fData memory managmenet and BLAS routines on the GPU
   BLAS_gpu  theBlas_gpu;

   // allocate memory on the HOST
   theBlas.Malloc(  A);
   theBlas.Malloc(TAU);
   theBlas.Malloc(GetBackA);

   // allocate memory on the DEVICE
   theBlas_gpu.Malloc(  A_d);
   theBlas_gpu.Malloc(TAU_d);
   theBlas_gpu.Calloc(GetBackA_d);

//   theBlas_gpu.Calloc0(GetBackA_d.GetDataPtrAdrs(), GetBackA_d.GetSize());
   
   
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

   //GPU: copy the fData array content of A,U,VT and SIGMA to the GPU
   theBlas_gpu.CopyToGPU(   A,   A_d);
   
   // Compute 
   //  1. the QR factorization of the A, M x N marix: since M<N, the diagonal
   //     and above elements will contain the N x N upper-triangular matrix R
   //  2. form the R matrix by taking the result of the previous QR
   //  3. multiply this R by the Q to get back the initial matrix A:
   //       R is an min(M,N)-by-N matrix. Q is a M-by-min(M,N) matrix (not formed)
   //       When we perform the multiplictaion, the result is QR -> M-by-N matrix
   //       such that R is overwritten: so make GetBackA to be M-by-N and store the 
   //       R matrix on its upper diagonal (not the most efficient but it's testing)


   double duration;
   struct timeval start,finish;  
   gettimeofday(&start, NULL); 

   // 1.
   printf("\n ======== ALL the 3 computations (QR decomp., forming matrix R and QxR) done on the GPU ===== \n");
   printf("\n ---- XGEQRF (LAPACK-QR: A = QR starts on GPU");
   theBlas_gpu.XGEQRF(A_d, TAU_d);
   // 2.
   printf("\n ---- Forming the R matrix (M-by-N) but only upper triangular on GPU");
   dim3 numThreads(32,32,1);
   dim3 numBlocks( std::ceil( float(N)/numThreads.x ),  // row
                   std::ceil( float(N)/numThreads.y ),  // col
                   1
                  );
   GetUpperTriangular_2D<<< numBlocks, numThreads >>>(  A_d.GetDataPtr(), GetBackA_d.GetDataPtr(), M, N);  
   // Synchronize
   cudaDeviceSynchronize();
//   cuda_status = cudaDeviceSynchronize();
//   assert(cuda_status == cudaSuccess && "\n GetUpperTriangular_2D is NOT SUCCESS \n");
   
   // 3.
   printf("\n ---- XORMQR (LAPACK-?ORMQR: C = QC starts on GPU");
   theBlas_gpu.XORMQR(GetBackA_d, A_d, TAU_d);     
        
   gettimeofday(&finish, NULL); 
   duration = ((double)(finish.tv_sec-start.tv_sec)*1000000 + (double)(finish.tv_usec-start.tv_usec)) / 1000000;
   printf("\n\n == TIMING: %lf [s] \n\n",  duration);

   //GPU: copy back the resulted fData arraies from GPU to the host SIGMA, U, VT
   theBlas_gpu.CopyFromGPU(  A_d,   A);
   theBlas_gpu.CopyFromGPU(TAU_d, TAU);
   theBlas_gpu.CopyFromGPU(GetBackA_d, GetBackA);
   
   
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

   //GPU: free memory allocated on the GPU
   theBlas_gpu.Free(A_d);
   theBlas_gpu.Free(TAU_d);
   theBlas_gpu.Free(GetBackA_d);
   
   return 0;
 }  
