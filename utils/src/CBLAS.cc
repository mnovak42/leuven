
#include "CBLAS.hh"

#include <cstdio>


CBLAS::CBLAS(int verbose) : kMKLAlignment(64) { SetNumThreads(1, verbose); }


void CBLAS::SetNumThreads(int nthreads, int verbose) {
#if USE_MKL_BLAS
//  mkl_set_num_threads(nthreads);
  mkl_set_num_threads_local(nthreads);
  if (verbose>1)
    printf ("   ---> Using Intel(R) MKL BLAS on %i threads.\n",  mkl_get_max_threads());
#elif USE_OPEN_BLAS
  openblas_set_num_threads(nthreads);
  if (verbose>1)
    printf ("   ---> Using Open BLAS on %i threads.\n",  openblas_get_num_threads());
#elif USE_ATLAS_BLAS
  // ATLAS do not support dynamic setting of number of therads
  // It uses a fixed number that was determined at build/optimisation time.
  if (verbose>1);
    printf ("   ---> Using ATLAS BLAS on a fixed number of threads (determined at its build).\n");
#else   // should not happen (but it can ...)
  // NETLIB BLAS do not support multi threading. Since the NETLIB-BLAS option is 
  // also used as a wildcard option to use any BLAS-LAPACK implementation we 
  // cannot be sure what is the number of therads used.
  if (verbose>1);
    printf ("   ---> Using NETLIB BLAS or ANY BLAS on ? thread(s) (do not know for sure).\n");
#endif  
}



// ========================================================================== //
// ==================== CBLAS-XGEMM:  ======================================= //
// 
// Calling the CPU BLAS 'cblas' interface method for 'dgemm' or 'sgemm'
// 
// computes: C = alpha A^T B^T + beta C; 
// specialisation for double and float
// 
template < >
void CBLAS::xgemm(CBLAS_ORDER ord, CBLAS_TRANSPOSE trA, CBLAS_TRANSPOSE trB, 
                  int m, int n, int k, double alpha, double* A, int ldA, 
                  double* B, int ldB, double beta, double* C, int ldC) {
  cblas_dgemm(ord, trA, trB, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);
}
template < >
void CBLAS::xgemm(CBLAS_ORDER ord, CBLAS_TRANSPOSE trA, CBLAS_TRANSPOSE trB, 
                  int m, int n, int k, float alpha, float* A, int ldA, 
                  float* B, int ldB, float beta, float* C, int ldC) {
  cblas_sgemm(ord, trA, trB, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);
}



// ========================================================================== //
// ==================== LAPACKE-XGESVD: SVD ================================= //
//
// Calling the LAPAKE 'lapack-C' interface method for 'dgesvd' or 'sgesvd'
//
// specialisation for double and float 
//
template< >
int CBLAS::xgesvd(int ord, char jobu, char jobvt, int m, int n, double* A, 
                  int ldA, double* SIGMA, double* U, int ldU, double* VT, 
                  int ldVT, double* sup) { 
  return LAPACKE_dgesvd(ord, jobu, jobvt, m, n, A, ldA, SIGMA, U, ldU, VT, ldVT, 
                        sup);                     
}
template< >
int CBLAS::xgesvd(int ord, char jobu, char jobvt, int m, int n, float* A, 
                  int ldA, float* SIGMA, float* U, int ldU, float* VT, 
                  int ldVT, float* sup) { 
  return LAPACKE_sgesvd(ord, jobu, jobvt, m, n, A, ldA, SIGMA, U, ldU, VT, ldVT, 
                        sup);                     
}


// ========================================================================== //
// ==================== LAPACKE-XGEQRF: QR  ================================= //
//
// Calling the LAPACKE 'lapack-C' interface method for 'dgeqrf' or 'sgeqrf'
//
// specialisation for double and float 
//
template< >
int CBLAS::xgeqrf(int ord, int m, int n, double* A, int ldA, double* TAU) { 
  return LAPACKE_dgeqrf(ord, m, n, A, ldA, TAU);                     
}
template< >
int CBLAS::xgeqrf(int ord, int m, int n, float* A, int ldA, float* TAU) { 
  return LAPACKE_sgeqrf(ord, m, n, A, ldA, TAU);                     
}



// ========================================================================== //
// ==================== LAPACKE-XORMQR: Mutiply by Q ======================== //
//
// Calling the LAPACKE 'lapack-C' interface method for 'dormqr' or 'sormqr'
//
// specialisation for double and float 
//
template< >
int CBLAS::xormqr(int ord, char side, char trans, int m, int n, int k, double* A, 
                  int ldA, const double* TAU, double* C, int ldC) { 
  return LAPACKE_dormqr(ord, side, trans, m, n, k, A, ldA, TAU, C, ldC);
}
template< >
int CBLAS::xormqr(int ord, char side, char trans, int m, int n, int k, float* A, 
                  int ldA, const float* TAU, float* C, int ldC) { 
  return LAPACKE_sormqr(ord, side, trans, m, n, k, A, ldA, TAU, C, ldC);
}


// ========================================================================== //
// ==================== LAPACKE-XORGQR: Form the matrix Q =================== //
//
// Calling the LAPACKE 'lapack-C' interface method for 'dorgqr' or 'sorgqr'
//
// specialisation for double and float 
//
template< >
int CBLAS::xorgqr(int ord, int m, int n, int k, double* A, int ldA, const double* TAU) { 
  return LAPACKE_dorgqr(ord, m, n, k, A, ldA, TAU);
}
template< >
int CBLAS::xorgqr(int ord, int m, int n, int k, float* A, int ldA, const float* TAU) { 
  return LAPACKE_sorgqr(ord, m, n, k, A, ldA, TAU);
}


// ========================================================================== //
// ==================== LAPACKE-XSYSV: SOLVE AX=B WITH A NxN SYMMETRIC ====== //
//
// Calling the LAPACKE 'lapack-C' interface method for 'dsysv' or 'ssysv'
//
// specialisation for double and float 
//
template< >
int CBLAS::xsysv(int ord, char uplo, int n, int nrhs, double* A, int ldA, 
                 int* ipiv, double* B, int ldB) { 
  return LAPACKE_dsysv(ord, uplo, n, nrhs, A, ldA, ipiv, B, ldB);
}
template< >
int CBLAS::xsysv(int ord, char uplo, int n, int nrhs, float* A, int ldA, 
                 int* ipiv, float* B, int ldB) { 
  return LAPACKE_ssysv(ord, uplo, n, nrhs, A, ldA, ipiv, B, ldB);
}


// ========================================================================== //
// ===== LAPACKE-XSYEVR: SELECTED EIGENVALUES/VECTORS OF A NxN SYMMETRIC ==== //
//
// Calling the LAPACKE 'lapack-C' interface method for 'dsyevr' or 'ssyevr'
//
// specialisation for double and float 
//
template< >
int CBLAS::xsyevr(int ord, char jobz, char range, char uplo, int n, double* A, 
                  int ldA, double vL, double vU, int iL, int iU, double tol, 
                  int* m, double* W, double *Z, int ldZ, int* isuppz) {
  return LAPACKE_dsyevr(ord, jobz, range, uplo, n, A, ldA, vL, vU, iL, iU, tol, 
                        m, W, Z, ldZ, isuppz);           
}
template< >
int CBLAS::xsyevr(int ord, char jobz, char range, char uplo, int n, float* A, 
                  int ldA, float vL, float vU, int iL, int iU, float tol, 
                  int* m, float* W, float *Z, int ldZ, int* isuppz) {
  return LAPACKE_ssyevr(ord, jobz, range, uplo, n, A, ldA, vL, vU, iL, iU, tol, 
                        m, W, Z, ldZ, isuppz);           
}




