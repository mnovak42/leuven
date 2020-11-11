
//
// CUBLAS function templates.
// This file is included into CUBLAS.hh
//

#include <cuda_runtime.h>

template < class T >
void CUBLAS::Malloc ( Matrix<T>& m ) {
  cudaError_t cudaError;
  cudaError = cudaMalloc ( m.GetDataPtrAdrs(), m.GetSize()*sizeof( T ) );
  assert ( cudaError==0 && cudaGetErrorString ( cudaError ) );
  _unused (cudaError);
}


template<class T>
void CUBLAS::Calloc ( Matrix<T>& m ) {
  cudaError_t cudaError;
  cudaError = cudaMalloc(  m.GetDataPtrAdrs(), m.GetSize()*sizeof( T ) );
  assert ( cudaError==cudaSuccess && cudaGetErrorString ( cudaError) );
  if ( cudaError==cudaSuccess ) cudaMemset( m.GetDataPtr(), 0.0, m.GetSize()*sizeof( T ));
}


template<class T>
void   CUBLAS::Free(Matrix<T>& m) {
  cudaFree((void*)m.GetDataPtr());
}


template<class T>
void CUBLAS::CopyToGPU(Matrix<T>& m_h, Matrix<T>& m_d) {
  cudaError_t cudaError;
  cudaError = cudaMemcpy(m_d.GetDataPtr(), m_h.GetDataPtr(), m_h.GetSize()*sizeof( T ), cudaMemcpyHostToDevice);
  assert ( cudaError==0 && cudaGetErrorString ( cudaError ) );
  _unused (cudaError);
}

template<class T>
void CUBLAS::CopyFromGPU(Matrix<T>& m_d, Matrix<T>& m_h) {
  cudaError_t cudaError;
  cudaError = cudaMemcpy(m_h.GetDataPtr(), m_d.GetDataPtr(), m_h.GetSize()*sizeof( T ), cudaMemcpyDeviceToHost);
  assert ( cudaError==0 && cudaGetErrorString ( cudaError ) );  
  _unused(cudaError);
}

template<class T>
void CUBLAS::CopyOnGPU(T* from_d, T* to_d, size_t size) {
  cudaError_t cudaError;
  cudaError = cudaMemcpy(to_d, from_d, size, cudaMemcpyDeviceToDevice);
  assert ( cudaError==0 && cudaGetErrorString ( cudaError ) );  
  _unused(cudaError);
}


//
template < class T >
void CUBLAS::XGEMM(Matrix<T>& A_d, Matrix<T>& B_d, Matrix<T>& C_d, T alpha, 
                   T beta, bool isTransA, bool isTransB) {
  // all Matrix should have the same Col-major order !!! true by design
  // Specifies whether to transpose matrix A. 
  cublasOperation_t theTransA = (isTransA) ? CUBLAS_OP_T : CUBLAS_OP_N;
  // Specifies whether to transpose matrix B.
  cublasOperation_t theTransB = (isTransB) ? CUBLAS_OP_T : CUBLAS_OP_N;
  // Number of rows in matrices A and C.  
  int theM       = (int) ((isTransA) ? A_d.GetNumCols() :  A_d.GetNumRows()); 
  // Number of cols in matrices B and C. 
  int theN       = (int) ((isTransA) ? B_d.GetNumRows() :  B_d.GetNumCols()); 
  // Number of cols in matrix A; number of rows in matrix B.
  int theK       = (int) ((isTransA) ? A_d.GetNumRows() :  A_d.GetNumCols()); 
  // The size of the first dimention of matrix A[m][k]; m (col-major)
  int theLDA     = (int) A_d.GetNumRows();
  // The size of the first dimention of matrix B[k][n]; k (col-major)
  int theLDB     = (int) B_d.GetNumRows();
  // The size of the first dimention of matrix C[m][n]; m (col-major)
  int theLDC     = (int) C_d.GetNumRows();
  // Create a CUBLAS handle
  cublasHandle_t handle;
  cublasCreate( &handle );
  //
  // invoke XDGEMM (X={s=float,g=double}) : C = alpha A^T B^T + beta C  
  // https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemm:
  /* 
  cublasStatus_t cublasDgemm(cublasHandle_t handle,
                         cublasOperation_t transa, cublasOperation_t transb,
                         int m, int n, int k,
                         const double *alpha,
                         const double *A, int lda,
                         const double *B, int ldb,
                         const double *beta,
                         double *C, int ldc)
  */
  cublasStatus_t cuBlasStatus;
  cuBlasStatus = xgemm(handle, theTransA, theTransB, theM, theN, theK, &alpha, 
                       A_d.GetDataPtr(), theLDA, B_d.GetDataPtr(), theLDB, &beta, 
                       C_d.GetDataPtr(), theLDC);
  assert ( cuBlasStatus==CUBLAS_STATUS_SUCCESS && "\n cuBlasStatus is NOT SUCCESS \n");
  _unused (cuBlasStatus);
  // Destroy the CUBLAS handle
  cublasDestroy(handle);
}




template < class T >
void CUBLAS::XGESVD(Matrix<T>& A_d,  Matrix<T>& SIGMA_d, const char* JOBU, 
                    const char* JOBVT, Matrix<T>& U_d, Matrix<T>& VT_d) {
  // Get dimensions of matrices 
  int M   = A_d.GetNumRows();
  int N   = A_d.GetNumCols();    
  int LDA = M;
  // dimension of U is M-by-M but the used part depends on JOBU:
  //  - 'A'  => U M-by-M i.e. all the M col-s required
  //  - 'S'  => U M-by-min(M,N) i.e. the min(M,N) col-s required
  //  - 'N' or O'  => not referenced (i.e. matrix U can be anything)
  int LDU  = U_d.GetNumRows(); 
  // dimensions of VT is N-by-N but the used part depends on JOBVT: 
  //  - 'A'  => V^T N-by-N i.e. all the N row-s required
  //  - 'S'  => V^T min(M,N)-by-N i.e. the min(M,N) row-s required
  //  - 'N' or O'  => not referenced (i.e. matrix T can be anything)
  int LDVT = VT_d.GetNumRows();
  // get JOBU and JOBVT
  signed char jobu  = *JOBU;
  signed char jobvt = *JOBVT;
  //
  // Create a cusolverDnHandle_t handle (and check)
  cusolverDnHandle_t handle;
  cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS; 
  cusolver_status = cusolverDnCreate( &handle );
  assert(CUSOLVER_STATUS_SUCCESS == cusolver_status && "\n CUBLAS::XGESVD-1 is NOT SUCCESS \n");
  //
  // Query work space of SVD (and check) ....D/S
  T dum = 0.;
  int lwork = XGESVDBufferSize(handle, M, N, dum);
  //
  // Allocate workspace on DEVICE
  T*  work_d = NULL;
  T* rwork_d = NULL;
  cudaError_t cuda_status = cudaSuccess;
  cuda_status = cudaMalloc((void**)&work_d, sizeof(T)*lwork);
  assert(cudaSuccess == cuda_status && "\n CUBLAS::XGESVD-2 is NOT SUCCESS \n");
  cuda_status = cudaMalloc((void**)&rwork_d, sizeof(T)*std::min(M,N)-1);
  assert(cudaSuccess == cuda_status && "\n CUBLAS::XGESVD-3 is NOT SUCCESS \n");
  //
  // On DEVICE memory pointers
  int*  info_d = NULL;
  cuda_status = cudaMalloc ((void**)&info_d, sizeof(int));
  assert(cudaSuccess == cuda_status && "\n CUBLAS::XGESVD-4 is NOT SUCCESS \n");
  //
  // Call cuslover-Dense-geSVD and compute SVD
  cusolver_status = xgesvd(handle, jobu, jobvt, M, N, A_d.GetDataPtr(), LDA, 
                          SIGMA_d.GetDataPtr(), U_d.GetDataPtr(), LDU,
                          VT_d.GetDataPtr(), LDVT, work_d, lwork, rwork_d, info_d);
  // Synchronize
  cuda_status = cudaDeviceSynchronize();
  assert(cusolver_status == CUSOLVER_STATUS_SUCCESS && "\n CUBLAS::XGESVD-5 is NOT SUCCESS \n");
  assert(cuda_status == cudaSuccess && "\n CUBLAS::XGESVD-6 is NOT SUCCESS \n");
  //
  // free allocated memory
  if (work_d)  cudaFree(work_d);
  if (rwork_d) cudaFree(rwork_d);                 
  if (info_d)  cudaFree(info_d);
}

template < class T >
void CUBLAS::XGESVD(Matrix<T>& A_d,  Matrix<T>& SIGMA_d) {
  // Get dimensions of matrices 
  int M   = A_d.GetNumRows();
  int N   = A_d.GetNumCols();    
  int LDA = M;
  // U and VT are not referenced
  int LDU  = 1;
  int LDVT = 1;
  T*  U_d  = NULL;
  T*  VT_d = NULL;
  // get JOBU and JOBVT
  signed char jobu  = 'O';
  signed char jobvt = 'N';
  //
  // Create a cusolverDnHandle_t handle (and check)
  cusolverDnHandle_t handle;
  cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS; 
  cusolver_status = cusolverDnCreate( &handle );
  assert(CUSOLVER_STATUS_SUCCESS == cusolver_status && "\n CUBLAS::XGESVD-1 is NOT SUCCESS \n");
  //
  // Query work space of SVD (and check) ....D/S
  T dum = 0.;
  int lwork = XGESVDBufferSize(handle, M, N, dum);
  //
  // Allocate workspace on DEVICE
  T*  work_d = NULL;
  T* rwork_d = NULL;
  cudaError_t cuda_status = cudaSuccess;
  cuda_status = cudaMalloc((void**)&work_d, sizeof(T)*lwork);
  assert(cudaSuccess == cuda_status && "\n CUBLAS::XGESVD-2 is NOT SUCCESS \n");
  cuda_status = cudaMalloc((void**)&rwork_d, sizeof(T)*std::min(M,N)-1);
  assert(cudaSuccess == cuda_status && "\n CUBLAS::XGESVD-3 is NOT SUCCESS \n");
  //
  // On DEVICE memory pointers
  int*  info_d = NULL;
  cuda_status = cudaMalloc ((void**)&info_d, sizeof(int));
  assert(cudaSuccess == cuda_status && "\n CUBLAS::XGESVD-4 is NOT SUCCESS \n");
  //
  // Call cuslover-Dense-geSVD and compute SVD
  cusolver_status = xgesvd(handle, jobu, jobvt, M, N, A_d.GetDataPtr(), LDA, 
                          SIGMA_d.GetDataPtr(), U_d, LDU, VT_d, LDVT, work_d, 
                          lwork, rwork_d, info_d);
  // Synchronize
  cuda_status = cudaDeviceSynchronize();
  assert(cusolver_status == CUSOLVER_STATUS_SUCCESS && "\n CUBLAS::XGESVD-5 is NOT SUCCESS \n");
  assert(cuda_status == cudaSuccess && "\n CUBLAS::XGESVD-6 is NOT SUCCESS \n");
  //
  // free allocated memory
  if (work_d)  cudaFree(work_d);
  if (rwork_d) cudaFree(rwork_d);                 
  if (info_d)  cudaFree(info_d);
}



template < class T >
void CUBLAS::XGEQRF(Matrix<T>& A_d,  Matrix<T>& TAU_d) {
  // Get dimensions of matrices 
  int M   = A_d.GetNumRows();
  int N   = A_d.GetNumCols();    
  int LDA = M;
  //
  // Create a cusolverDnHandle_t handle (and check)
  cusolverDnHandle_t handle;
  cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS; 
  cusolver_status = cusolverDnCreate( &handle );
  assert(CUSOLVER_STATUS_SUCCESS == cusolver_status && "\n CUBLAS::XGEQRF-1 is NOT SUCCESS \n");
  //
  // Query work space of QR (and check) ....D/S
  int lwork = XGEQRFBufferSize(handle, M, N, A_d.GetDataPtr(), LDA);
  //
  // Allocate workspace on DEVICE
  T*  work_d = NULL;
  cudaError_t cuda_status = cudaSuccess;
  cuda_status = cudaMalloc((void**)&work_d, sizeof(T)*lwork);
  assert(cudaSuccess == cuda_status && "\n CUBLAS::XGEQRF-2 is NOT SUCCESS \n");
  //
  // On DEVICE memory pointers
  int*  info_d = NULL;
  cuda_status = cudaMalloc ((void**)&info_d, sizeof(int));
  assert(cudaSuccess == cuda_status && "\n CUBLAS::XGEQRF-3 is NOT SUCCESS \n");
  //
  // Call cuslover-Dense-geQRF and compute QR factorization
  cusolver_status = xgeqrf(handle, M, N, A_d.GetDataPtr(), LDA, TAU_d.GetDataPtr(), 
                           work_d, lwork, info_d);
  // Synchronize
  cuda_status = cudaDeviceSynchronize();
  assert(cusolver_status == CUSOLVER_STATUS_SUCCESS && "\n CUBLAS::XGEQRF-4 is NOT SUCCESS \n");
  assert(cuda_status == cudaSuccess && "\n CUBLAS::XGEQRF-5 is NOT SUCCESS \n");
  //
  // free allocated memory
  if (work_d)  cudaFree(work_d);
  if (info_d)  cudaFree(info_d);
}


template < class T >
void CUBLAS::XORMQR(Matrix<T>& C_d,  Matrix<T>& A_d, Matrix<T>& TAU_d, 
                    bool isQLeft, bool isTransQ) {
  // get dimensions of matric C 
  int M   = C_d.GetNumRows();
  int N   = C_d.GetNumCols();
  // number of elementary refractors stored in Q (and TAU) 
  int K   = std::min(A_d.GetNumRows(), A_d.GetNumCols());
  // leading dimension of matrix A and C
  int LDA = A_d.GetNumRows();
  int LDC = C_d.GetNumRows();
  // Specifies whether to transpose matrix Q (A_d). 
  cublasOperation_t theTransQ = (isTransQ) ? CUBLAS_OP_T      : CUBLAS_OP_N;
  // Specifies whether to multiply matrix C with matrix Q (A_d) from the left.
  cublasSideMode_t  theSideQ  = (isQLeft)  ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;
  //
  // Create a cusolverDnHandle_t handle (and check)
  cusolverDnHandle_t handle;
  cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS; 
  cusolver_status = cusolverDnCreate( &handle );
//  printf("\n%s\n",cusolverGetErrorString(cusolver_status));
//  PrintCuSolverStatus(cusolver_status);
  assert(CUSOLVER_STATUS_SUCCESS == cusolver_status && "\n CUBLAS::XORMQR-1 is NOT SUCCESS \n");

  //
  // Query work space (and check) ....D/S
  int lwork = XORMQRBufferSize(handle, theSideQ, theTransQ, M, N, K, A_d.GetDataPtr(), 
                               LDA, TAU_d.GetDataPtr(), C_d.GetDataPtr(), LDC);
  //
  // Allocate workspace on DEVICE
  T*  work_d = NULL;
  cudaError_t cuda_status = cudaSuccess;
  cuda_status = cudaMalloc((void**)&work_d, sizeof(T)*lwork);
  assert(cudaSuccess == cuda_status && "\n CUBLAS::XORMQR-2 is NOT SUCCESS \n");
  //
  // On DEVICE memory pointers
  int*  info_d = NULL;
  cuda_status = cudaMalloc ((void**)&info_d, sizeof(int));
  assert(cudaSuccess == cuda_status && "\n CUBLAS::XORMQR-3 is NOT SUCCESS \n");
  //
  // Call cuslover-Dense-ormQR and compute C = QC or Q^TC or CQ or CQ^T
  cusolver_status = xormqr(handle, theSideQ, theTransQ, M, N, K, A_d.GetDataPtr(), 
                           LDA, TAU_d.GetDataPtr(), C_d.GetDataPtr(), LDC,
                           work_d, lwork, info_d);
  // Synchronize
  cuda_status = cudaDeviceSynchronize();
  assert(cusolver_status == CUSOLVER_STATUS_SUCCESS && "\n CUBLAS::XORMQR-4 is NOT SUCCESS \n");
  assert(cuda_status == cudaSuccess && "\n CUBLAS::XORMQR-5 is NOT SUCCESS \n");
  //
  // free allocated memory
  if (work_d)  cudaFree(work_d);
  if (info_d)  cudaFree(info_d);
}


template < class T >
void CUBLAS::XORGQR(Matrix<T>& A_d, Matrix<T>& TAU_d) {
  // get dimensions of matrix A_d 
  int M   = std::max(A_d.GetNumRows(), A_d.GetNumCols());
  int N   = std::min(A_d.GetNumRows(), A_d.GetNumCols());
  // number of elementary refractors stored in Q (and TAU) 
  int K   = N;
  // leading dimension of matrix A
  int LDA = A_d.GetNumRows();
  // Create a cusolverDnHandle_t handle (and check)
  cusolverDnHandle_t handle;
  cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS; 
  cusolver_status = cusolverDnCreate( &handle );
//  printf("\n%s\n",cusolverGetErrorString(cusolver_status));
//  PrintCuSolverStatus(cusolver_status);
  assert(CUSOLVER_STATUS_SUCCESS == cusolver_status && "\n CUBLAS::XORGQR-1 is NOT SUCCESS \n");

  //
  // Query work space (and check) ....D/S
  int lwork = XORGQRBufferSize(handle, M, N, K, A_d.GetDataPtr(), LDA, TAU_d.GetDataPtr());
  //
  // Allocate workspace on DEVICE
  T*  work_d = NULL;
  cudaError_t cuda_status = cudaSuccess;
  cuda_status = cudaMalloc((void**)&work_d, sizeof(T)*lwork);
  assert(cudaSuccess == cuda_status && "\n CUBLAS::XORGQR-2 is NOT SUCCESS \n");
  //
  // On DEVICE memory pointers
  int*  info_d = NULL;
  cuda_status = cudaMalloc ((void**)&info_d, sizeof(int));
  assert(cudaSuccess == cuda_status && "\n CUBLAS::XORGQR-3 is NOT SUCCESS \n");
  //
  // Call cuslover-Dense-orgQR and form the matrix Q from a previous A = QR, that results in the inout A and Tau
  cusolver_status = xorgqr(handle, M, N, K, A_d.GetDataPtr(), LDA, TAU_d.GetDataPtr(),
                           work_d, lwork, info_d);
  // Synchronize
  cuda_status = cudaDeviceSynchronize();
  assert(cusolver_status == CUSOLVER_STATUS_SUCCESS && "\n CUBLAS::XORGQR-4 is NOT SUCCESS \n");
  assert(cuda_status == cudaSuccess && "\n CUBLAS::XORGQR-5 is NOT SUCCESS \n");
  //
  // free allocated memory
  if (work_d)  cudaFree(work_d);
  if (info_d)  cudaFree(info_d);
}


template < class T >
void CUBLAS::XGESV(Matrix<T>& A_d,  Matrix<T>& B_d, bool isTransA) {
  // get dimensions of matrix A (should be squared i.e. M=N) 
  int M    = A_d.GetNumRows();
  int N    = A_d.GetNumCols();
  int NRHS = B_d.GetNumCols();
  // leading dimension of matrix A and B
  int LDA  = A_d.GetNumRows();
  int LDB  = B_d.GetNumRows();
  // Specifies whether to transpose matrix A 
  cublasOperation_t theTransA = (isTransA) ? CUBLAS_OP_T : CUBLAS_OP_N;
  //
  // Create a cusolverDnHandle_t handle (and check)
  cusolverDnHandle_t handle;
  cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS; 
  cusolver_status = cusolverDnCreate( &handle );
//  printf("\n%s\n",cusolverGetErrorString(cusolver_status));
//  PrintCuSolverStatus(cusolver_status);
  assert(CUSOLVER_STATUS_SUCCESS == cusolver_status && "\n CUBLAS::XGESV-1 is NOT SUCCESS \n");
  //
  //  ------------------------------------------------------------------------
  //  The LU decompositon of the input matrix A using XGETRF
  //  ------------------------------------------------------------------------
  //
  // Query work space (and check) ....D/S
  int lwork = XGETRFBufferSize(handle, M, N, A_d.GetDataPtr(), LDA);
  //
  // Allocate workspace on DEVICE
  T*  work_d = NULL;
  cudaError_t cuda_status = cudaSuccess;
  cuda_status = cudaMalloc((void**)&work_d, sizeof(T)*lwork);
  assert(cudaSuccess == cuda_status && "\n CUBLAS::XGESVR-2 is NOT SUCCESS \n");
  //
  // On DEVICE memory pointers
  int*  info_d = NULL;
  int*  ipiv_d = NULL;
  cuda_status = cudaMalloc ((void**)&info_d, sizeof(int));
  assert(cudaSuccess == cuda_status && "\n CUBLAS::XGESV-3 is NOT SUCCESS \n");
  cuda_status = cudaMalloc ((void**)&ipiv_d, sizeof(int)*std::min(M,N));
  assert(cudaSuccess == cuda_status && "\n CUBLAS::XGESV-4 is NOT SUCCESS \n");

  //
  // Call cuslover-Dense-getrf for the LU decomposition of A (squared N-by-N)
  cusolver_status = xgetrf(handle, M, N, A_d.GetDataPtr(), LDA, work_d, ipiv_d, info_d);
  // Synchronize
  cuda_status = cudaDeviceSynchronize();
  assert(cusolver_status == CUSOLVER_STATUS_SUCCESS && "\n CUBLAS::XGESV-5 is NOT SUCCESS \n");
  assert(cuda_status == cudaSuccess && "\n CUBLAS::XGESV-6 is NOT SUCCESS \n");
  //
  //  ------------------------------------------------------------------------
  //  The solution of AX=B using XGETRS (after the LU decompositon of A)
  //  ------------------------------------------------------------------------
  //
  // Call cuslover-Dense-getrs for solving AX=B after the LU decomposition of the 
  // squared matrix A (N-by-N)
  cusolver_status = xgetrs(handle, theTransA, N, NRHS, A_d.GetDataPtr(), LDA, 
                           ipiv_d, B_d.GetDataPtr(), LDB, info_d);
  // Synchronize
  cuda_status = cudaDeviceSynchronize();
  assert(cusolver_status == CUSOLVER_STATUS_SUCCESS && "\n CUBLAS::XGESV-6 is NOT SUCCESS \n");
  assert(cuda_status == cudaSuccess && "\n CUBLAS::XGESV-7 is NOT SUCCESS \n");
  //
  // free allocated memory
  if (work_d)  cudaFree(work_d);
  if (info_d)  cudaFree(info_d);
  if (ipiv_d)  cudaFree(ipiv_d);
}


template < class T >
int CUBLAS::XSYEVDX(Matrix<T>& A_d,  Matrix<T>& EIGENVALS_d, 
                    int whichEigenValue, T minEigenVal, T maxEigenVal, bool isUploA) {
  // Only eigenvlaues are requested in this method (i.e. no eigenvectors)
  cusolverEigMode_t   jobz = CUSOLVER_EIG_MODE_NOVECTOR;
  // Which eigenvalues: all only a range (defined by min/max values or indices)
  cusolverEigRange_t range =  (whichEigenValue==0 ? CUSOLVER_EIG_RANGE_ALL 
                            : (whichEigenValue==1 ? CUSOLVER_EIG_RANGE_V   
                                                  : CUSOLVER_EIG_RANGE_I));
  cublasFillMode_t   uplo  =  isUploA ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
  // Get dimensions of matrix A (NxN symmetric) 
  int   N = A_d.GetNumRows();    
  // Col major (but square anyway) 
  int LDA = N;
  // If the range of the required eignevalues are given as min/max index:
  //  - determine lower/upper indices of the required eigenvlaues
  int  iL = (whichEigenValue==2) ? static_cast<int>(minEigenVal) : 0;
  int  iU = (whichEigenValue==2) ? static_cast<int>(maxEigenVal) : 0;
  // If the range of the required eignevalues are given as min/max values:
  //  - determine lower/upper values of the required eigenvlaues
  T    vL = (whichEigenValue==1) ? minEigenVal : 0.;
  T    vU = (whichEigenValue==1) ? maxEigenVal : 0.;
  // All the N eigenvalues are required otherwise...
  // Number of eigenvectors found (on exit)
  int   M = 0;
  // Create a cusolverDnHandle_t handle (and check)
  cusolverDnHandle_t handle;
  cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS; 
  cusolver_status = cusolverDnCreate( &handle );
  assert(CUSOLVER_STATUS_SUCCESS == cusolver_status && "\n CUBLAS::XSYEVDX-1 is NOT SUCCESS \n");
  //
  // Query work space of D/S-SYEVDX 
  int lwork = XSYEVDXBufferSize(handle, jobz, range, uplo, N, A_d.GetDataPtr(), 
                                LDA, vL, vU, iL, iU, &M, EIGENVALS_d.GetDataPtr());
  // Allocate workspace on DEVICE
  T*  work_d = NULL;
  cudaError_t cuda_status = cudaSuccess;
  cuda_status = cudaMalloc((void**)&work_d, sizeof(T)*lwork);
  assert(cudaSuccess == cuda_status && "\n CUBLAS::XSYEVDX-2 is NOT SUCCESS \n");
  //
  // Further on DEVICE memory pointers
  int*  info_d = NULL;
  cuda_status = cudaMalloc ((void**)&info_d, sizeof(int));
  assert(cudaSuccess == cuda_status && "\n CUBLAS::XSYEVDX-3 is NOT SUCCESS \n");
  //
  // Call eigensolver cuslover-Dense-<D/S>-syevdx and compute the eigenvalues
  cusolver_status = xsyevdx(handle, jobz, range, uplo, N, A_d.GetDataPtr(), LDA, 
                            vL, vU, iL, iU, &M, EIGENVALS_d.GetDataPtr(),
                            work_d, lwork, info_d);
  // Synchronize
  cuda_status = cudaDeviceSynchronize();
  assert(cusolver_status == CUSOLVER_STATUS_SUCCESS && "\n CUBLAS::XSYEVDX-4 is NOT SUCCESS \n");
  assert(cuda_status == cudaSuccess && "\n CUBLAS::XSYEVDX-5 is NOT SUCCESS \n");
  //
  // free allocated memory
  if (work_d)  cudaFree(work_d);
  if (info_d)  cudaFree(info_d);
  //
  return M;
}


template < class T >
int 
CUBLAS::XSYEVDX(Matrix<T>& A_d,  Matrix<T>& EIGENVALS_d, Matrix<T>& EIGENVECTS_d, 
                int whichEigenValue, T minEigenVal, T maxEigenVal, bool isUploA) {
  // Both eigenvlaues and eigenvectors are requested in this method
  cusolverEigMode_t   jobz = CUSOLVER_EIG_MODE_VECTOR;
  // Which eigenvalues: all only a range (defined by min/max values or indices)
  cusolverEigRange_t range =  (whichEigenValue==0 ? CUSOLVER_EIG_RANGE_ALL 
                            : (whichEigenValue==1 ? CUSOLVER_EIG_RANGE_V   
                                                  : CUSOLVER_EIG_RANGE_I));
  cublasFillMode_t   uplo  =  isUploA ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
  // Get dimensions of matrix A (NxN symmetric) 
  int   N = A_d.GetNumRows();    
  // Col major (but square anyway) 
  int LDA = N;
  // If the range of the required eignevalues are given as min/max index:
  //  - determine lower/upper indices of the required eigenvlaues
  int  iL = (whichEigenValue==2) ? static_cast<int>(minEigenVal) : 0;
  int  iU = (whichEigenValue==2) ? static_cast<int>(maxEigenVal) : 0;
  // If the range of the required eignevalues are given as min/max values:
  //  - determine lower/upper values of the required eigenvlaues
  T    vL = (whichEigenValue==1) ? minEigenVal : 0.;
  T    vU = (whichEigenValue==1) ? maxEigenVal : 0.;
  // All the N eigenvalues are required otherwise...
  // Number of eigenvectors found (on exit)
  int   M = 0;
  //
  // Make sure that the EIGENVECTS matrix has proper dimensions
  assert ( N==EIGENVECTS_d.GetNumRows() && "\n*** CUBLAS::XSYEVDX: EIGENVECTS matrix should have same number of row as the input square matrix A. ***\n");                  
  //assert ( whichEigenValue==0 && N==EIGENVECTS_d.GetNumCols() && "\n*** CUBLAS::XSYEVDX: EIGENVECTS matrix should have enough cols to store the N eigenvectors.(all eigens was required). ***\n");                  
  //assert ( whichEigenValue==2 && (iU-iL+1)>=EIGENVECTS_d.GetNumCols() && "\n*** CUBLAS::XSYEVDX: EIGENVECTS matrix should have enough cols to store the iU-iL+1 eigenvectors. ***\n");                  
  //
  // Create a cusolverDnHandle_t handle (and check)
  cusolverDnHandle_t handle;
  cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS; 
  cusolver_status = cusolverDnCreate( &handle );
  assert(CUSOLVER_STATUS_SUCCESS == cusolver_status && "\n CUBLAS::XSYEVDX-1 is NOT SUCCESS \n");
  //
  // Query work space of D/S-SYEVDX 
  int lwork = XSYEVDXBufferSize(handle, jobz, range, uplo, N, A_d.GetDataPtr(), 
                                LDA, vL, vU, iL, iU, &M, EIGENVALS_d.GetDataPtr());
  // Allocate workspace on DEVICE
  T*  work_d = NULL;
  cudaError_t cuda_status = cudaSuccess;
  cuda_status = cudaMalloc((void**)&work_d, sizeof(T)*lwork);
  assert(cudaSuccess == cuda_status && "\n CUBLAS::XSYEVDX-2 is NOT SUCCESS \n");
  //
  // Further on DEVICE memory pointers
  int*  info_d = NULL;
  cuda_status = cudaMalloc ((void**)&info_d, sizeof(int));
  assert(cudaSuccess == cuda_status && "\n CUBLAS::XSYEVDX-3 is NOT SUCCESS \n");
  //
  // Call eigensolver cuslover-Dense-<D/S>-syevdx and compute the eigenvalues
  cusolver_status = xsyevdx(handle, jobz, range, uplo, N, A_d.GetDataPtr(), LDA, 
                            vL, vU, iL, iU, &M, EIGENVALS_d.GetDataPtr(),
                            work_d, lwork, info_d);                          
  // Synchronize
  cuda_status = cudaDeviceSynchronize();
  assert(cusolver_status == CUSOLVER_STATUS_SUCCESS && "\n CUBLAS::XSYEVDX-4 is NOT SUCCESS \n");
  assert(cuda_status == cudaSuccess && "\n CUBLAS::XSYEVDX-5 is NOT SUCCESS \n");
  //
  // free allocated memory
  if (work_d)  cudaFree(work_d);
  if (info_d)  cudaFree(info_d);
  //
  // Take the orthonormal eigenevtors (from the overwritten cols. of the input 
  // A matrix and write into the cols of the EIGENVECTS matrix (note, that it's 
  // an on device copy!)
  cudaError_t cudaError;
  cudaError = cudaMemcpy(EIGENVECTS_d.GetDataPtr(), A_d.GetDataPtr(), N*M*sizeof(T), cudaMemcpyDeviceToDevice);
  assert ( cudaError==0 && cudaGetErrorString ( cudaError ) );
//
  return M;
}
