#ifndef CUDAUTILS_HH
#define CUDAUTILS_HH

#ifdef USE_CUBLAS

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cstdio>


// cuda error handling
#define cudaErrchk(ans) { cuAssert((ans), __FILE__, __LINE__); }
  inline void cuAssert(cudaError_t code, const char *file, int line, bool abort=true){
      if (code != cudaSuccess) {
          fprintf(stderr,"CUAassert: %s %s %d\n",
          cudaGetErrorString(code), file, line);
          if (abort) exit(code);
      }
  }


// cuBLAS
static const char* cuBlasGetSatusEnum(cublasStatus_t status);
#define cuBlasErrchk(ans) { cuBlasAssert((ans), __FILE__, __LINE__); }
  inline void cuBlasAssert(cublasStatus_t status, const char *file, int line, bool abort=true){
      if (status != CUBLAS_STATUS_SUCCESS) {
          fprintf(stderr,"CUBlasAssert: %s %s %d\n",
          cuBlasGetSatusEnum(status), file, line);
          if (abort) exit(status);
      }
  }
  static const char* cuBlasGetSatusEnum(cublasStatus_t status) {
    switch (status) {
      case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
      case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";
      case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";
      case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";
      case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";
      case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";
      case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";
      case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
      case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";
      case CUBLAS_STATUS_LICENSE_ERROR:
        return "CUBLAS_STATUS_LICENSE_ERROR";
      default:
        return "<unknown cublasStatus_t>";
    }
  }


// cuSOLVER error handling
static const char* cuSolverGetSatusEnum(cusolverStatus_t status);
#define cuSolverErrchk(ans) { cuSolverAssert((ans), __FILE__, __LINE__); }
  inline void cuSolverAssert(cusolverStatus_t status, const char *file, int line, bool abort=true){
      if (status != CUSOLVER_STATUS_SUCCESS) {
          fprintf(stderr,"CUSolverAssert: %s %s %d\n",
          cuSolverGetSatusEnum(status), file, line);
          if (abort) exit(status);
      }
  }
  static const char* cuSolverGetSatusEnum(cusolverStatus_t status) {
    switch (status) {
      case CUSOLVER_STATUS_SUCCESS:
        return "CUSOLVER_STATUS_SUCCESS";
      case CUSOLVER_STATUS_NOT_INITIALIZED:
        return "CUSOLVER_STATUS_NOT_INITIALIZED";
      case CUSOLVER_STATUS_ALLOC_FAILED:
        return "CUSOLVER_STATUS_ALLOC_FAILED";
      case CUSOLVER_STATUS_INVALID_VALUE:
        return "CUSOLVER_STATUS_INVALID_VALUE";
      case CUSOLVER_STATUS_ARCH_MISMATCH:
        return "CUSOLVER_STATUS_ARCH_MISMATCH";
      case CUSOLVER_STATUS_MAPPING_ERROR:
        return "CUSOLVER_STATUS_MAPPING_ERROR";
      case CUSOLVER_STATUS_EXECUTION_FAILED:
        return "CUSOLVER_STATUS_EXECUTION_FAILED";
      case CUSOLVER_STATUS_INTERNAL_ERROR:
        return "CUSOLVER_STATUS_INTERNAL_ERROR";
      case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
      case CUSOLVER_STATUS_NOT_SUPPORTED:
        return "CUSOLVER_STATUS_NOT_SUPPORTED ";
      case CUSOLVER_STATUS_ZERO_PIVOT:
        return "CUSOLVER_STATUS_ZERO_PIVOT";
      case CUSOLVER_STATUS_INVALID_LICENSE:
        return "CUSOLVER_STATUS_INVALID_LICENSE";
      default:
        return "<unknown cusolverStatus_t>";
    }
  }

#endif // USE_CUBLAS

#endif // CUDAUTILS_HH
