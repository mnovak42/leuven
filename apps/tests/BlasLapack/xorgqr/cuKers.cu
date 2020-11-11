
#include "cuKers.h"

#include <cuda_runtime.h>

#include <cstdio>

// a is an M-by-N matrix with M>=N 
// b is an N-by-N matrix i.e. min(M,N)xmin(M,N)
// will write the upper triangular of a into the upper triangualr of b
template < >
__global__
void GetUpperTriangular_2D(double* a, double* b, int m, int n) {
  int theMin = min(m,n);
  int ir= blockIdx.x * blockDim.x + threadIdx.x;
  int ic= blockIdx.y * blockDim.y + threadIdx.y;
  if (ic<theMin && ir<=ic) {
//     printf("%d\%d%\t%lg\n",ir,ic,a[ic*m+ir]);
//     int idx = ic*m + ir;
    b[ic*theMin+ir] = a[ic*m+ir]; 
   }
}

template < >
__global__
void GetUpperTriangular_2D(float* a, float* b, int m, int n) {
   int theMin = min(m,n);
   int ir= blockIdx.x * blockDim.x + threadIdx.x;
   int ic= blockIdx.y * blockDim.y + threadIdx.y;
   if (ic<theMin && ir<=ic) {
//     printf("%d\%d%\t%lg\n",ir,ic,a[ic*m+ir]);
//     int idx = ic*m + ir;
     b[ic*theMin+ir] = a[ic*m+ir]; 
   }
}

template < >
void GetUpperTriangular(double* a_d, double* b_d, int m, int n) {
  dim3 numThreads(32,32,1);
  dim3 numBlocks( std::ceil( float(m)/numThreads.x ),  // row
                  std::ceil( float(n)/numThreads.y ),  // col
                  1
                 );
  GetUpperTriangular_2D<<< numBlocks, numThreads >>> (a_d, b_d, m, n);  
  cudaDeviceSynchronize();  
}

template < >
void GetUpperTriangular(float* a_d, float* b_d, int m, int n) {
  dim3 numThreads(32,32,1);
  dim3 numBlocks( std::ceil( float(m)/numThreads.x ),  // row
                  std::ceil( float(n)/numThreads.y ),  // col
                  1
                 );
  GetUpperTriangular_2D<<< numBlocks, numThreads >>> (a_d, b_d, m, n);  
  cudaDeviceSynchronize();  
}

