
#include "cuKers.h"

#include <cuda_runtime.h>

// a is an M-by-N matrix with M>=N 
// b is an N-by-N matrix 
// will write the upper triangular of a into the upper triangualr of b
template < >
__global__
void GetUpperTriangular_2D(double* a, double* b, int m, int n) {
   int ir= blockIdx.x * blockDim.x + threadIdx.x;
   int ic= blockIdx.y * blockDim.y + threadIdx.y;
   if (ir<m && ic<n && ic>=ir) {
//     printf("%d\%d%\t%lg\n",ir,ic,a[ic*m+ir]);
//     int idx = ic*m + ir;
     b[ic*m+ir] = a[ic*m+ir]; 
   }
}

template < >
__global__
void GetUpperTriangular_2D(float* a, float* b, int m, int n) {
   int ir= blockIdx.x * blockDim.x + threadIdx.x;
   int ic= blockIdx.y * blockDim.y + threadIdx.y;
   if (ir<m && ic<n && ic>=ir) {
//     printf("%d\%d%\t%lg\n",ir,ic,a[ic*m+ir]);
//     int idx = ic*m + ir;
     b[ic*m+ir] = a[ic*m+ir]; 
   }
}

template < >
void GetUpperTriangular(double* a_d, double* b_d, int m, int n) {
  dim3 numThreads(32,32,1);
  dim3 numBlocks( std::ceil( float(n)/numThreads.x ),  // row
                  std::ceil( float(n)/numThreads.y ),  // col
                  1
                 );
  GetUpperTriangular_2D<<< numBlocks, numThreads >>> (a_d, b_d, m, n);  
  cudaDeviceSynchronize();  
}

template < >
void GetUpperTriangular(float* a_d, float* b_d, int m, int n) {
  dim3 numThreads(32,32,1);
  dim3 numBlocks( std::ceil( float(n)/numThreads.x ),  // row
                  std::ceil( float(n)/numThreads.y ),  // col
                  1
                 );
  GetUpperTriangular_2D<<< numBlocks, numThreads >>> (a_d, b_d, m, n);  
  cudaDeviceSynchronize();  
}

