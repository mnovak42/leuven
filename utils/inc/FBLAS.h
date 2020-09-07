
//
// FBLAS C-external function declarations
// This file is included into FBLAS.hh

// -------------------------------------------------------------------------- //
extern "C" {
    void dgemm_(char* transA, char* transB, int* ms, int* ns, int* ks, 
                double* alpha, double* A, int* ldA, 
                double* B, int* ldB, double* beta,
                double* C, int* ldC);  
}

extern "C" {
    void sgemm_(char* transA, char* transB, int* ms, int* ns, int* ks, 
                float* alpha, float* A, int* ldA, 
                float* B, int* ldB, float* beta,
                float* C, int* ldC);  
}


// -------------------------------------------------------------------------- //
extern "C" {
    void dgesvd_(char* jobU, char* jobVT, int* m, int* n, double* A, int* ldA,
                 double* S, double* U, int* ldU, double* VT, int* ldVT, 
                 double* work, int* lwork, int* info);
  
}

extern "C" {
    void sgesvd_(char* jobU, char* jobVT, int* m, int* n, float* A, int* ldA,
                 float* S, float* U, int* ldU, float* VT, int* ldVT, 
                 float* work, int* lwork, int* info);
  
}


// -------------------------------------------------------------------------- //
extern "C" {
    void dgeqrf_(int* m, int* n, double* A, int* ldA, double* TAU, 
                 double* work, int* lwork, int* info);
  
}

extern "C" {
    void sgeqrf_(int* m, int* n, float* A, int* ldA, float* TAU,
                 float* work, int* lwork, int* info);
  
}


// -------------------------------------------------------------------------- //
extern "C" {
    void dormqr_(char* side, char* trans, int* m, int* n, int* k, double* A, 
                 int* ldA, double* TAU, double* C, int* ldC,
                 double* work, int* lwork, int* info);
  
}

extern "C" {
    void sormqr_(char* side, char* trans, int* m, int* n, int* k, float* A, 
                 int* ldA, float* TAU, float* C, int* ldC,
                 float* work, int* lwork, int* info);
  
}


// -------------------------------------------------------------------------- //
extern "C" {
    void dorgqr_(int* m, int* n, int* k, double* A, int* ldA, double* TAU,
                 double* work, int* lwork, int* info);
  
}

extern "C" {
    void sorgqr_(int* m, int* n, int* k, float* A, int* ldA, float* TAU,
                 float* work, int* lwork, int* info);
  
}


// -------------------------------------------------------------------------- //
extern "C" {
    void dsysv_(char* uplo, int* n, int* nrhs, double* A, int* ldA, int* ipiv, 
                double* B, int* ldB,
                double* work, int* lwork, int* info);  
}

extern "C" {
    void ssysv_(char* uplo, int* n, int* nrhs, float* A, int* ldA, int* ipiv, 
                float* B, int* ldB,
                float* work, int* lwork, int* info);  
}


// -------------------------------------------------------------------------- //
extern "C" {
    void dsyevr_(char* jobz, char* range, char* uplo, int* n, double* A, int* ldA, 
                 double* vL, double* vU, int* iL, int* iU, double* tol, int* m, 
                 double* W, double *Z, int* ldZ, int* isuppz, double* work, 
                 int* lwork, int* iwork, int* liwork, int* info);
}

extern "C" {
    void ssyevr_(char* jobz, char* range, char* uplo, int* n, float* A, int* ldA, 
                 float* vL, float* vU, int* iL, int* iU, float* tol, int* m, 
                 float* W, float *Z, int* ldZ, int* isuppz, float* work, 
                 int* lwork, int* iwork, int* liwork, int* info);
}
