
#ifndef CUBLAS_HH
#define CUBLAS_HH

#include "definitions.hh"

#include "Matrix.hh"


// We will use the most flexible cuBLAS API. See more on the (3) API-s at 
// https://docs.nvidia.com/cuda/cublas/index.html#using-the-cublas-api

// https://docs.nvidia.com/cuda/cublas/index.html#appendix-a-using-the-cublas-legacy-api
#include <cublas_v2.h>
// https://docs.nvidia.com/cuda/cusolver/index.html
#include <cusolverDn.h>



// We will use the most flexible cuBLAS API here. See more on the (3) API-s at 
// https://docs.nvidia.com/cuda/cublas/index.html#using-the-cublas-api
// NOTE: ALL BLAS functions are supported by this API

// cuBLASXt API: the cublasXt API of cuBLAS exposes a multi-GPU capable Host 
// interface : when using this API the application only needs to allocate the 
// required matrices on the Host memory space. This API supports only the 
// compute-intensive BLAS3 functions.

// cuBLASLt API: a new lightweight library dedicated to GEneral Matrix-to-matrix 
// Multiply (GEMM) operations with lots of flexibility. 


// kernel for getting upper triangular of a matrix
__global__
void GetUpperTriangular2D(double* a_d, double* b_d, int m, int n);
__global__
void GetUpperTriangular2D(float* a_d, float* b_d, int m, int n);



class CUBLAS {
public:
  CUBLAS();
  
  // allocate/free memory on the DEVICE (GPU) for the fData array of the Matrix 
  // structure: only col-major so 2nd template arg. (true by default) no need 
  template < class T >
  void Malloc(Matrix<T>& m);
  template < class T >
  void Calloc(Matrix<T>& m);
  template < class T >
  void Calloc0 (T **m, size_t size );
  template < class T >
  void Free(Matrix<T>& m);

  // copy the Matrix fData data array from/to HOST (_h) to/from DEVICE (_d)
  template < class T >
  void CopyToGPU(Matrix<T>& m_h, Matrix<T>& m_d);
  template < class T >
  void CopyFromGPU(Matrix<T>& m_d, Matrix<T>& m_h);
  template < class T >
  void CopyOnGPU(T* from_d, T* to_h, size_t size);

  void SetNumThreads(int/*nthreads*/) { /* not MT */}


/*
  // a is an M-by-N matrix with M>=N 
  // b is an N-by-N matrix 
  // will write the upper triangular of a into the upper triangualr of b
  template < class T >
  __global__
  void GetUpperTriangular2D(T* a_d, T* b_d, int m, int n);
*/
  // A_d is supposed to be an MxN matrix with M>=N
  // B_d is supposed to be an NxN i.e. N=min(M,N)
  // This fuction takes the upper triangular part of A and copies to the upper 
  // triangular of B. Lower triangular of B is untouched.
  template <class T>
  void GetUpperTriangular(Matrix<T>& A_d, Matrix<T>& B_d);


  //
  // BLAS Level 3
  //
  // invoke XDGEMM (X={s=float,g=double}) : C = alpha A^T B^T + beta C
  // NOTE: fData arrays of matrices A, B and C should have been allocated on 
  //       the DEVICE memory (e.g. with Malloc ar Calloc) and the content of the 
  //       host matrices A and B should have already been copied to the DEVICE     
  template < class T >
  void   XGEMM(Matrix<T>& A_d, Matrix<T>& B_d, Matrix<T>& C_d, T alpha=1., 
               T beta=1., bool isTransA=false, bool isTransB=false);  
  template < class T >
  cublasStatus_t xgemm(cublasHandle_t handle, cublasOperation_t trA, 
              cublasOperation_t trB, int m, int n, int k, T* alpha, T* A_d, 
              int ldA, T* B_d, int ldB, T* beta, T* C_d, int ldC);
              
  
  void PrintCuSolverStatus(cusolverStatus_t status) {
    switch (status) {
      case CUSOLVER_STATUS_SUCCESS:
        return;
      case CUSOLVER_STATUS_NOT_INITIALIZED:
        printf("cuSolver has not been initialized\n");
        break;
      case CUSOLVER_STATUS_ALLOC_FAILED:
        printf("cuSolver allocation failed\n");
        break;
      case CUSOLVER_STATUS_INVALID_VALUE:
        printf("cuSolver invalid value error\n");
        break;
      case CUSOLVER_STATUS_ARCH_MISMATCH:
        printf("cuSolver architecture mismatch error\n");
        break;
      case CUSOLVER_STATUS_MAPPING_ERROR:
        printf("cuSolver mapping error\n");
        break;
      case CUSOLVER_STATUS_EXECUTION_FAILED:
        printf("cuSolver execution failed\n");
        break;
      case CUSOLVER_STATUS_INTERNAL_ERROR:
        printf("cuSolver internal error\n");
        break;
      case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        printf("cuSolver matrix type not supported error\n");
        break;
      case CUSOLVER_STATUS_NOT_SUPPORTED:
        printf("cuSolver not supported error\n");
        break;
      case CUSOLVER_STATUS_ZERO_PIVOT:
        printf("cuSolver zero pivot error\n");
        break;
      case CUSOLVER_STATUS_INVALID_LICENSE:
        printf("cuSolver invalid license error\n");
        break;
      default:
        printf("Unknown cuSolver error\n");
    }
  }
  
  
  // https://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-gesvd            
  template < class T >
  void    XGESVD(Matrix<T>& A_a,  Matrix<T>& SIGMA_d, const char* JOBU, 
              const char* JOBVT, Matrix<T>& U_d, Matrix<T>& VT_d);
  template < class T >
  void    XGESVD(Matrix<T>& A_d,  Matrix<T>& SIGMA_d);
  template < class T >
  int     XGESVDBufferSize(cusolverDnHandle_t handle, int m, int n, T dum);
  template < class T > 
  cusolverStatus_t xgesvd(cusolverDnHandle_t handle, signed char jobu,
                          signed char jobvt, int m, int n, T *A_d, int ldA, 
                          T* S_d, T* U_d, int ldU, T* VT_d, int ldVT, T* work_d, 
                          int lwork, T* rwork_d, int* info_d);
  
  // https://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-geqrf
  template < class T >
  void    XGEQRF(Matrix<T>& A_d,  Matrix<T>& TAU_d);
  template < class T >
  int     XGEQRFBufferSize(cusolverDnHandle_t handle, int m, int n, T* A, int lda);
  template < class T > 
  cusolverStatus_t xgeqrf(cusolverDnHandle_t handle, int m, int n, T *A_d, int ldA, 
                          T* TAU_d, T* work_d, int lwork, int* info_d);
  
  
  // https://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-ormqr
  template < class T >
  void    XORMQR(Matrix<T>& C_d,  Matrix<T>& A_d, Matrix<T>& TAU_d, 
                 bool isQLeft=true, bool isTransQ=false);
  template < class T >
  int     XORMQRBufferSize(cusolverDnHandle_t handle, cublasSideMode_t side,
                           cublasOperation_t trans, int m, int n, int k, 
                           const T* A_d, int ldA, const T* TAU_d, const T* C_d, 
                           int ldC);
  template < class T > 
  cusolverStatus_t xormqr(cusolverDnHandle_t handle, cublasSideMode_t side,
                          cublasOperation_t trans, int m, int n, int k, 
                          const T* A_d, int ldA, const T* TAU_d, T* C_d,
                          int ldC, T* work_d, int lwork, int* info_d);


  // https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDN-lt-t-gt-orgqr
  template < class T >
  void    XORGQR(Matrix<T>& A_d, Matrix<T>& TAU_d);
  template < class T >
  int     XORGQRBufferSize(cusolverDnHandle_t handle, int m, int n, int k, 
                           const T* A_d, int ldA, const T* TAU_d);
  template < class T > 
  cusolverStatus_t xorgqr(cusolverDnHandle_t handle, int m, int n, int k, 
                          T* A_d, int ldA, const T* TAU_d, 
                          T* work_d, int lwork, int* info_d);
  
  
  
  
  
  // There is no way to get the Lapack ?sysv to solve system of equations with 
  // symmetric coefficient matrix: cusolverDn?sytrf is available for the Bunch-
  // Kaufman factorization of a symmetric indefinite matrix that corresponds 
  // to the Lapack ?sytrf but there is no cusolverDn?sytrs or cusolverDn?sytrs2
  //
  // We could try to use the ?potrf and ?potrs that are positive definite matrix
  // (since we will use the kernel matrix as coefficient matrix which is PD) but
  // due to instabilities, those would fail with some (some minor of the coef. 
  // matrix is not PD) error. This is actually why the ?sysv (?sytrf+?sytrs)
  // is used even in the CPU version instead of the ?posv (?potrf+?potrs).
  //
  // So we will use the genral solver on GPU which is the combination of the 
  // cusolverDn?getrf (LU decomposition of a general M-by-N matrix) and 
  // cusolverDn?getrs to solve the system of equations. These will be combined 
  // in the XGESV driver (that corresponds to the Lapack ?gesv driver).
  // 
  // https://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-getrf
  // https://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-getrs
  
  // Solves AX=B for general M-by-N matrix A, that will be destroyed and the 
  // results will be stored in the columns of B
  template < class T >
  void    XGESV(Matrix<T>& A_d,  Matrix<T>& B_d, bool isTransA=false);

  // XGTRF for LU of M-by-N geneal matrix
  template < class T >
  int     XGETRFBufferSize(cusolverDnHandle_t handle, int m, int n, T* A_d, int ldA);
  template < class T > 
  cusolverStatus_t xgetrf(cusolverDnHandle_t handle, int m, int n, T* A_d, int ldA,
                          T* work_d, int* ipiv_d, int* info_d);

  // XGTRS for solving AX=B after an LU of the M-by-N geneal matrix A
  template < class T > 
  cusolverStatus_t xgetrs(cusolverDnHandle_t handle, cublasOperation_t trans, 
                          int n, int nrhs, const T* A_d, int ldA, 
                          const int* ipiv_d, T* B_d, int ldB, int* info_d);


  // https://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-syevdx
  // same as lapack d-s-syevr 
  // eigenvectors only
  template < class T >
  int    XSYEVDX(Matrix<T>& A_d,  Matrix<T>& EIGENVALS_d, 
                 int whichEigenValue=0, T minEigenVal=0., T maxEigenVal=0., bool isUploA=true);
  template < class T >
  int    XSYEVDX(Matrix<T>& A,  Matrix<T>& EIGENVALS, Matrix<T>& EIGENVECTS, 
                 int whichEigenValue=0, T minEigenVal=0., T maxEigenVal=0., bool isUploA=true);
  template < class T >
  int     XSYEVDXBufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, 
                            cusolverEigRange_t range, cublasFillMode_t uplo, 
                            int n, const T* A_d, int ldA, T vL, T vU, int iL, 
                            int iU, int *m, const T* W_d);
  template < class T > 
  cusolverStatus_t xsyevdx(cusolverDnHandle_t handle, cusolverEigMode_t jobz, 
                           cusolverEigRange_t range, cublasFillMode_t uplo, 
                           int n, T* A_d, int ldA, T vL, T vU, int iL, int iU, 
                           int *m, T* W_d, T* work_d, int lwork, int* info_d);

};

#include "CUBLAS.tpp"

#endif  // CUBLAS_HH