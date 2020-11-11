
#include "CUBLAS.hh"

#include <cuda_runtime.h>
#include <cmath>

CUBLAS::CUBLAS() { /* nothing to do */ }


// template specialisationfor double and float
template < >
cublasStatus_t 
CUBLAS::xgemm(cublasHandle_t handl, cublasOperation_t trA, cublasOperation_t trB, 
              int m, int n, int k, double* alpha, double* A_d, int ldA, 
              double* B_d, int ldB, double* beta, double* C_d, int ldC) {
  return cublasDgemm(handl, trA, trB, m, n, k, alpha, A_d, ldA, B_d, ldB, beta, 
                     C_d, ldC);
}
template < >
cublasStatus_t 
CUBLAS::xgemm(cublasHandle_t handl, cublasOperation_t trA, cublasOperation_t trB, 
              int m, int n, int k, float* alpha, float* A_d, int ldA, 
              float* B_d, int ldB, float* beta, float* C_d, int ldC) {
  return cublasSgemm(handl, trA, trB, m, n, k, alpha, A_d, ldA, B_d, ldB, beta, 
                     C_d, ldC);
}

//
// ====  ?GESVD ==
//
// template specialisation for double and float
template < >
int CUBLAS::XGESVDBufferSize(cusolverDnHandle_t handle, int m, int n, double) {
  int lwork = 0;
  cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
  cusolver_status = cusolverDnDgesvd_bufferSize(handle, m, n, &lwork);
  assert(cusolver_status == CUSOLVER_STATUS_SUCCESS && "\n CUBLAS::XGESVDBufferSize double \n");
  return lwork;
}
template < >
int CUBLAS::XGESVDBufferSize(cusolverDnHandle_t handle, int m, int n, float) {
  int lwork = 0;
  cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
  cusolver_status = cusolverDnSgesvd_bufferSize(handle, m, n, &lwork);
  assert(cusolver_status == CUSOLVER_STATUS_SUCCESS && "\n CUBLAS::XGESVDBufferSize float \n");
  return lwork;
}

template < > 
cusolverStatus_t 
CUBLAS::xgesvd(cusolverDnHandle_t handle, signed char jobu, signed char jobvt, 
               int m, int n, double *A_d, int ldA, double* S_d, double* U_d, 
               int ldU, double* VT_d, int ldVT, double* work_d, int lwork, 
               double* rwork_d, int* info_d) {
  // Call SVD
  return cusolverDnDgesvd(handle, jobu, jobvt, m, n, A_d, ldA, S_d, U_d, ldU, 
                          VT_d, ldVT, work_d, lwork, rwork_d, info_d);
}
template < > 
cusolverStatus_t 
CUBLAS::xgesvd(cusolverDnHandle_t handle, signed char jobu, signed char jobvt, 
               int m, int n, float* A_d, int ldA, float* S_d, float* U_d, 
               int ldU, float* VT_d, int ldVT, float* work_d, int lwork, 
               float* rwork_d, int* info_d) {
  // Call SVD
  return cusolverDnSgesvd(handle, jobu, jobvt, m, n, A_d, ldA, S_d, U_d, ldU, 
                          VT_d, ldVT, work_d, lwork, rwork_d, info_d);
}


//
// ====  ?GEQRF ==
//
// template specialisation for double and float

template < >
int 
CUBLAS::XGEQRFBufferSize(cusolverDnHandle_t handle, int m, int n, double* A_d, int lda) {
  int lwork = 0;
  cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
  cusolver_status = cusolverDnDgeqrf_bufferSize(handle, m, n, A_d, lda, &lwork);
  assert(cusolver_status == CUSOLVER_STATUS_SUCCESS && "\n CUBLAS::XGEQRFBufferSize double \n");
  return lwork;  
}
template < >
int 
CUBLAS::XGEQRFBufferSize(cusolverDnHandle_t handle, int m, int n, float* A_d, int lda) {
  int lwork = 0;
  cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
  cusolver_status = cusolverDnSgeqrf_bufferSize(handle, m, n, A_d, lda, &lwork);
  assert(cusolver_status == CUSOLVER_STATUS_SUCCESS && "\n CUBLAS::XGEQRFBufferSize float \n");
  return lwork;  
}

template < > 
cusolverStatus_t 
CUBLAS::xgeqrf(cusolverDnHandle_t handle, int m, int n, double *A_d, int ldA, 
               double* TAU_d, double* work_d, int lwork, int* info_d) {
  // Call QR
  return cusolverDnDgeqrf(handle, m, n, A_d, ldA, TAU_d, work_d, lwork, info_d);
}
template < > 
cusolverStatus_t 
CUBLAS::xgeqrf(cusolverDnHandle_t handle, int m, int n, float *A_d, int ldA, 
               float* TAU_d, float* work_d, int lwork, int* info_d) {
  // Call QR
  return cusolverDnSgeqrf(handle, m, n, A_d, ldA, TAU_d, work_d, lwork, info_d);
}



//
// ====  ?ORMQR ==
//
// template specialisation for double and float

template < >
int CUBLAS::XORMQRBufferSize(cusolverDnHandle_t handle, cublasSideMode_t side,
                            cublasOperation_t trans, int m, int n, int k, 
                            const double* A_d, int ldA, const double* TAU_d, 
                            const double* C_d, int ldC) {
  int lwork = 0;
  cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
  cusolver_status = cusolverDnDormqr_bufferSize(handle, side, trans, m, n, k, A_d, 
                                                ldA, TAU_d, C_d, ldC, &lwork);
  assert(cusolver_status == CUSOLVER_STATUS_SUCCESS && "\n CUBLAS::ORMQRBufferSize double \n");
  return lwork;  
}
template < >
int CUBLAS::XORMQRBufferSize(cusolverDnHandle_t handle, cublasSideMode_t side,
                            cublasOperation_t trans, int m, int n, int k, 
                            const float* A_d, int ldA, const float* TAU_d, 
                            const float* C_d, int ldC) {
  int lwork = 0;
  cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
  cusolver_status = cusolverDnSormqr_bufferSize(handle, side, trans, m, n, k, A_d, 
                                                ldA, TAU_d, C_d, ldC, &lwork);
  assert(cusolver_status == CUSOLVER_STATUS_SUCCESS && "\n CUBLAS::ORMQRBufferSize float \n");
  return lwork;  
}

template < > 
cusolverStatus_t 
CUBLAS::xormqr(cusolverDnHandle_t handle, cublasSideMode_t side, 
               cublasOperation_t trans, int m, int n, int k, const double* A_d, 
               int ldA, const double* TAU_d, double* C_d, int ldC, 
               double* work_d, int lwork, int* info_d) {
  //
  // Call multiplication by Q (A)
  return cusolverDnDormqr(handle, side, trans, m, n, k, A_d, ldA, TAU_d, C_d, ldC, 
                          work_d, lwork, info_d);
}
template < > 
cusolverStatus_t 
CUBLAS::xormqr(cusolverDnHandle_t handle, cublasSideMode_t side,
               cublasOperation_t trans, int m, int n, int k, const float* A_d, 
               int ldA, const float* TAU_d, float* C_d, int ldC, 
               float* work_d, int lwork, int* info_d) {
  // Call multiplication by Q (A)
  return cusolverDnSormqr(handle, side, trans, m, n, k, A_d, ldA, TAU_d, C_d, ldC, 
                          work_d, lwork, info_d);
}



//
// ====  ?ORGQR ==
//
// template specialisation for double and float

template < >
int CUBLAS::XORGQRBufferSize(cusolverDnHandle_t handle, int m, int n, int k, 
                            const double* A_d, int ldA, const double* TAU_d) {
  int lwork = 0;
  cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
  cusolver_status = cusolverDnDorgqr_bufferSize(handle, m, n, k, A_d, ldA, TAU_d, &lwork);
  assert(cusolver_status == CUSOLVER_STATUS_SUCCESS && "\n CUBLAS::ORGQRBufferSize double \n");
  return lwork;  
}
template < >
int CUBLAS::XORGQRBufferSize(cusolverDnHandle_t handle, int m, int n, int k, 
                            const float* A_d, int ldA, const float* TAU_d) {
  int lwork = 0;
  cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
  cusolver_status = cusolverDnSorgqr_bufferSize(handle, m, n, k, A_d, ldA, TAU_d, &lwork);
  assert(cusolver_status == CUSOLVER_STATUS_SUCCESS && "\n CUBLAS::ORGQRBufferSize float \n");
  return lwork;  
}

template < > 
cusolverStatus_t 
CUBLAS::xorgqr(cusolverDnHandle_t handle, int m, int n, int k, double* A_d, 
               int ldA, const double* TAU_d, double* work_d, int lwork, int* info_d) {
  //
  // Form Q (A)
  return cusolverDnDorgqr(handle, m, n, k, A_d, ldA, TAU_d, work_d, lwork, info_d);
}
template < > 
cusolverStatus_t 
CUBLAS::xorgqr(cusolverDnHandle_t handle, int m, int n, int k, float* A_d, 
               int ldA, const float* TAU_d, float* work_d, int lwork, int* info_d) {
  //
  // Form Q (A)
  return cusolverDnSorgqr(handle, m, n, k, A_d, ldA, TAU_d, work_d, lwork, info_d);
}



//
// ====  ?GETRF ==
//
// template specialisation for double and float

template < >
int 
CUBLAS::XGETRFBufferSize(cusolverDnHandle_t handle, int m, int n, double* A_d, int ldA) {
  int lwork = 0;
  cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
  cusolver_status = cusolverDnDgetrf_bufferSize(handle, m, n, A_d, ldA, &lwork);
  assert(cusolver_status == CUSOLVER_STATUS_SUCCESS && "\n CUBLAS::XGETRFBufferSize double \n");
  return lwork;  
}
template < >
int 
CUBLAS::XGETRFBufferSize(cusolverDnHandle_t handle, int m, int n, float* A_d, int ldA) {
  int lwork = 0;
  cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
  cusolver_status = cusolverDnSgetrf_bufferSize(handle, m, n, A_d, ldA, &lwork);
  assert(cusolver_status == CUSOLVER_STATUS_SUCCESS && "\n CUBLAS::XGETRFBufferSize float \n");
  return lwork;  
}

template < > 
cusolverStatus_t 
CUBLAS::xgetrf(cusolverDnHandle_t handle, int m, int n, double* A_d, int ldA,
               double* work_d, int* ipiv_d, int* info_d) {
  // Call LU of A M-by-N
  return cusolverDnDgetrf(handle, m, n, A_d, ldA, work_d, ipiv_d, info_d);
}
template < > 
cusolverStatus_t 
CUBLAS::xgetrf(cusolverDnHandle_t handle, int m, int n, float* A_d, int ldA,
               float* work_d, int* ipiv_d, int* info_d) {
  // Call LU of A M-by-N
  return cusolverDnSgetrf(handle, m, n, A_d, ldA, work_d, ipiv_d, info_d);
}




//
// ====  ?GETRS ==
//
// template specialisation for double and float

template < > 
cusolverStatus_t 
CUBLAS::xgetrs(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, 
               const double* A_d, int ldA, const int* ipiv_d, double* B_d, int ldB, 
               int* info_d) {
  // Solve AX=B after the LU of A M-by-N
  return cusolverDnDgetrs(handle, trans, n, nrhs, A_d, ldA, ipiv_d, B_d, ldB, info_d);
}
template < > 
cusolverStatus_t 
CUBLAS::xgetrs(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, 
               const float* A_d, int ldA, const int* ipiv_d, float* B_d, int ldB, 
               int* info_d) {
  // Solve AX=B after the LU of A M-by-N
  return cusolverDnSgetrs(handle, trans, n, nrhs, A_d, ldA, ipiv_d, B_d, ldB, info_d);
}


// 
// ==== ?syevdx ==
//
// template specialisation for double and float
template < >
int 
CUBLAS::XSYEVDXBufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, 
      cusolverEigRange_t range, cublasFillMode_t uplo, int n, const double *A_d, 
      int ldA, double vL, double vU, int iL, int iU, int *m, const double *W_d) {
  int lwork = 0;
  cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
  cusolver_status = cusolverDnDsyevdx_bufferSize(handle, jobz, range, uplo, n, 
                                      A_d, ldA, vL, vU, iL, iU, m, W_d, &lwork);
  assert(cusolver_status == CUSOLVER_STATUS_SUCCESS && "\n CUBLAS::XSYEVDXBufferSize double \n");
  return lwork;  
}
template < >
int 
CUBLAS::XSYEVDXBufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, 
      cusolverEigRange_t range, cublasFillMode_t uplo, int n, const float *A_d, 
      int ldA, float vL, float vU, int iL, int iU, int *m, const float *W_d) {
  int lwork = 0;
  cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
  cusolver_status = cusolverDnSsyevdx_bufferSize(handle, jobz, range, uplo, n, 
                                      A_d, ldA, vL, vU, iL, iU, m, W_d, &lwork);
  assert(cusolver_status == CUSOLVER_STATUS_SUCCESS && "\n CUBLAS::XSYEVDXBufferSize double \n");
  return lwork;  
}

template < > 
cusolverStatus_t 
CUBLAS::xsyevdx(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range, 
                cublasFillMode_t uplo, int n, double *A_d, int ldA, 
                double vL, double vU, int iL, int iU, int *m, double *W_d,
                double* work_d, int lwork, int* info_d) {
  // Call dsyevdx on to compute selected eigenvalues and (Optionally) eigenvectors
  return cusolverDnDsyevdx(handle, jobz, range, uplo, n, A_d, ldA, vL, vU, iL, iU, 
                           m, W_d, work_d, lwork, info_d);
}
template < > 
cusolverStatus_t 
CUBLAS::xsyevdx(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range, 
                cublasFillMode_t uplo, int n, float *A_d, int ldA, 
                float vL, float vU, int iL, int iU, int *m, float *W_d,
                float* work_d, int lwork, int* info_d) {
  // Call ssyevdx on to compute selected eigenvalues and (Optionally) eigenvectors
  return cusolverDnSsyevdx(handle, jobz, range, uplo, n, A_d, ldA, vL, vU, iL, iU, 
                           m, W_d, work_d, lwork, info_d);
}



__global__
void GetUpperTriangular2D(double* a_d, double* b_d, int m, int n) {
  int theMin = n;
  int ir= blockIdx.x * blockDim.x + threadIdx.x;
  int ic= blockIdx.y * blockDim.y + threadIdx.y;
  if (ic<theMin && ir<=ic) {
//     printf("%d\%d%\t%lg\n",ir,ic,a[ic*m+ir]);
    b_d[ic*theMin+ir] = a_d[ic*m+ir]; 
   }
}


__global__
void GetUpperTriangular2D(float* a_d, float* b_d, int m, int n) {
  int theMin = n;
  int ir= blockIdx.x * blockDim.x + threadIdx.x;
  int ic= blockIdx.y * blockDim.y + threadIdx.y;
  if (ic<theMin && ir<=ic) {
//     printf("%d\%d%\t%lg\n",ir,ic,a[ic*m+ir]);
    b_d[ic*theMin+ir] = a_d[ic*m+ir]; 
   }
}



template < >
void CUBLAS::GetUpperTriangular(Matrix<double>& A_d, Matrix<double>& B_d) {
  int M = A_d.GetNumRows();
  int N = A_d.GetNumCols();
  assert (M>=N &&  B_d.GetNumRows()==B_d.GetNumCols() &&  B_d.GetNumCols()==N && "\n CUBLAS::GetUpperTriangular: A should be MxN with M>=N  and B should be NxN \n");  
  dim3 numThreads(32,32,1);
  dim3 numBlocks( std::ceil( float(N)/numThreads.x ),  // row
                  std::ceil( float(N)/numThreads.y ),  // col
                  1
                 );
  GetUpperTriangular2D<<< numBlocks, numThreads >>> (A_d.GetDataPtr(), B_d.GetDataPtr(), M, N);  
  cudaDeviceSynchronize();  
}


template < >
void CUBLAS::GetUpperTriangular(Matrix<float>& A_d, Matrix<float>& B_d) {
  int M = A_d.GetNumRows();
  int N = A_d.GetNumCols();
  assert (M>=N &&  B_d.GetNumRows()==B_d.GetNumCols() &&  B_d.GetNumCols()==N && "\n CUBLAS::GetUpperTriangular: A should be MxN with M>=N  and B should be NxN \n");  
  dim3 numThreads(32,32,1);
  dim3 numBlocks( std::ceil( float(N)/numThreads.x ),  // row
                  std::ceil( float(N)/numThreads.y ),  // col
                  1
                 );
  GetUpperTriangular2D<<< numBlocks, numThreads >>> (A_d.GetDataPtr(), B_d.GetDataPtr(), M, N);  
  cudaDeviceSynchronize();  
}


