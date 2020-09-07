
#include "FBLAS.hh"

#include <cstdio>


FBLAS::FBLAS(int verbose) : kMKLAlignment(64) { SetNumThreads(1, verbose); }


void FBLAS::SetNumThreads(int nthreads, int verbose) {
#if USE_MKL_BLAS
  mkl_set_num_threads(nthreads);
  if (verbose>1)
    printf ("   ---> Using Intel(R) MKL BLAS on %i threads.\n",  mkl_get_max_threads());
#elif USE_OPEN_BLAS
  openblas_set_num_threads(nthreads);
  if (verbose>1)
    printf ("   ---> Using Open BLAS on %i threads.\n",  openblas_get_num_threads());
#elif USE_ATLAS_BLAS
  // ATLAS do not support dynamic setting of number of therads
  // It uses a fixed number that was determined at build/optimisation time.
  (void)nthreads;
  if (verbose>1)
    printf ("   ---> Using ATLAS BLAS on a fixed number of threads (determined at its build).\n");
#else   // should not happen (but it can ...)
  // NETLIB BLAS do not support multi threading. Since the NETLIB-BLAS option is 
  // also used as a wildcard option to use any BLAS-LAPACK implementation we 
  // cannot be sure what is the number of therads used.
  (void)nthreads;
  if (verbose>1)
    printf ("   ---> Using NETLIB BLAS or ANY BLAS on ? thread(s) (do not know for sure).\n");
#endif  
}





// ========================================================================== //
// ==================== FBLAS-XGEMM:  ======================================= //
// 
// Calling the declared C methods for 'dgemm_' or 'sgemm_' (direct Fortan link.)
// 
// computes: C = alpha A^T B^T + beta C; 
// specialisation for double and float
template < >             
void FBLAS::xgemm(char* trA, char* trB, int* m, int* n, int* k, double* alpha, 
                  double* A, int* ldA, double* B, int* ldB, double* beta, 
                  double* C, int* ldC) {
  dgemm_(trA, trB, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);
}
template < >             
void FBLAS::xgemm(char* trA, char* trB, int* m, int* n, int* k, float* alpha, 
                  float* A, int* ldA, float* B, int* ldB, float* beta, 
                  float* C, int* ldC) {
  sgemm_(trA, trB, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);
}



// ========================================================================== //
// ====================== FBLAS-XGESVD: SVD ================================= //
//
// Calling the declared C methods for 'dgesvd_' or 'sgesvd_'(direct Fortan link.)
//
// specialisation for double and float 
//
template < >
void FBLAS::xgesvd(char* jobu, char* jobvt, int* m, int* n, double* A, int* ldA, 
                   double* SIGMA, double* U, int* ldU, double* VT, int* ldVT, 
                   double* work, int* lwork, int* info) {
  dgesvd_(jobu, jobvt, m, n, A, ldA, SIGMA, U, ldU, VT, ldVT, work, lwork, info);
}
template < >
void FBLAS::xgesvd(char* jobu, char* jobvt, int* m, int* n, float* A, int* ldA, 
                   float* SIGMA, float* U, int* ldU, float* VT, int* ldVT, 
                   float* work, int* lwork, int* info) {
  sgesvd_(jobu, jobvt, m, n, A, ldA, SIGMA, U, ldU, VT, ldVT, work, lwork, info);
}



// ========================================================================== //
// ==================== LAPACK-XGEQRF: QR  ================================= //
//
// Calling the C methods for 'dgeqrf_' or 'sgeqrf_' (direct Fortan link.)
//
// specialisation for double and float 
//
template< >
void FBLAS::xgeqrf(int* m, int* n, double* A, int* ldA, double* TAU, 
                   double* work, int* lwork, int* info) { 
  dgeqrf_(m, n, A, ldA, TAU, work, lwork, info);
}
template< >
void FBLAS::xgeqrf(int* m, int* n, float* A, int* ldA, float* TAU, 
                   float* work, int* lwork, int* info) { 
  sgeqrf_(m, n, A, ldA, TAU, work, lwork, info);
}


// ========================================================================== //
// ==================== LAPACK-XORMQR: Mutiply by Q ======================== //
//
// Calling the C methods for 'dormqr_' or 'sormqr_' (direct Fortan link.)
//
// specialisation for double and float 
//
template< >
void FBLAS::xormqr(char* side, char* trans, int* m, int* n, int* k, double* A, 
                   int* ldA, double* TAU, double* C, int* ldC,
                   double* work, int* lwork, int* info) { 
  dormqr_(side, trans, m, n, k, A, ldA, TAU, C, ldC, work, lwork, info);
}
template< >
void FBLAS::xormqr(char* side, char* trans, int* m, int* n, int* k, float* A, 
                   int* ldA, float* TAU, float* C, int* ldC,
                   float* work, int* lwork, int* info) { 
  sormqr_(side, trans, m, n, k, A, ldA, TAU, C, ldC, work, lwork, info);
}


// ========================================================================== //
// ==================== LAPACK-XORGQR: Form the matrix Q ==================== //
//
// Calling the C methods for 'dorgqr_' or 'sorgqr_' (direct Fortan link.)
//
// specialisation for double and float 
//
template< >
void FBLAS::xorgqr(int* m, int* n, int* k, double* A, int* ldA, double* TAU,
                   double* work, int* lwork, int* info) { 
  dorgqr_(m, n, k, A, ldA, TAU, work, lwork, info);
}
template< >
void FBLAS::xorgqr(int* m, int* n, int* k, float* A, int* ldA, float* TAU,
                   float* work, int* lwork, int* info) { 
  sorgqr_(m, n, k, A, ldA, TAU, work, lwork, info);
}


// ========================================================================== //
// ==================== LAPACK-XSYSV: SOLVE AX=B WITH A NxN SYMMETRIC ====== //
//
// Calling the C methods for 'dsysv_' or 'ssysv_' (direct Fortan link.)
//
// specialisation for double and float 
//
template< >
void FBLAS::xsysv(char* uplo, int* n, int* nrhs, double* A, int* ldA, int* ipiv, 
                  double* B, int* ldB,
                  double* work, int* lwork, int* info) { 
  dsysv_(uplo, n, nrhs, A, ldA, ipiv, B, ldB, work, lwork, info);
}
template< >
void FBLAS::xsysv(char* uplo, int* n, int* nrhs, float* A, int* ldA, int* ipiv, 
                  float* B, int* ldB,
                  float* work, int* lwork, int* info) { 
  ssysv_(uplo, n, nrhs, A, ldA, ipiv, B, ldB, work, lwork, info);
}



// ========================================================================== //
// ===== LAPACK-XSYEVR: SELECTED EIGENVALUES/VECTORS OF A NxN SYMMETRIC ==== //
//
// Calling the C methods for 'dsyevr_' or 'ssyevr_' (direct Fortan link.)
//
// specialisation for double and float 
//
template< >
void FBLAS::xsyevr(char* jobz, char* range, char* uplo, int* n, double* A, int* ldA, 
                   double* vL, double* vU, int* iL, int* iU, double* tol, int* m, 
                   double* W, double* Z, int* ldZ, int* isuppz, double* work, 
                   int* lwork, int* iwork, int* liwork, int* info) {
  dsyevr_(jobz, range, uplo, n, A, ldA, vL, vU, iL, iU, tol, m, W, Z, ldZ, 
          isuppz, work, lwork, iwork, liwork, info);           
}
template< >
void FBLAS::xsyevr(char* jobz, char* range, char* uplo, int* n, float* A, int* ldA, 
                   float* vL, float* vU, int* iL, int* iU, float* tol, int* m, 
                   float* W, float* Z, int* ldZ, int* isuppz, float* work, 
                   int* lwork, int* iwork, int* liwork, int* info) {
  ssyevr_(jobz, range, uplo, n, A, ldA, vL, vU, iL, iU, tol, m, W, Z, ldZ, 
          isuppz, work, lwork, iwork, liwork, info);          
}






