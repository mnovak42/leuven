
// -------------------------------------------------------------------------- //
//                                                                            //
// There are two wrappers for using different BLAS implemetations:            // 
//  - CBLAS that is used USE_CBLAS is defined (i.e. compile with -DUSE_CBLAS) //
//  - FBLAS that is used USE_FBLAS is defined (i.e. compile with -DUSE_FBLAS) // 
//                                                                            //
// FBLAS is a wrapper for using different BLAS implementations. BLAS routines //
// are called DIRECTLY through their Fortarn interface. Since all different   //
// BLAS implemetations supports this Fortran style interface it's possible to //
// use any of the BLAS implementations through this wrapper. The consequence  //
// of using directly the Fortran interface is that the Fortran style ordering,//
// COLUMN-MAJOR ordering needs to be used and cannot be changed.              //
//                                                                            //
// Note, that some BLAS implementations (MKL, OpenBLAS) include the C-BLAS    //
// C-interface that makes possible to use either ROW- or COLUMN-MAJOR order.  //
// In order to provide this flexibility, an other wrapper (called CBLAS ) is  //
// also available. The CBLAS wrapper will call the BLAS functions of the given//
// implementation through its built-in C-BLAS interface. This makes possible  //
// to use either ROW- or COLUMN-MAJOR order with the restriction that not all //
// BLAS implementations can be used through this CBALS interface (only those  //
// that have their C-BLAS interface).                                         //
// -------------------------------------------------------------------------- //


#ifndef FBLAS_HH
#define FBLAS_HH

#include "definitions.hh"

#include "Matrix.hh"

class FBLAS {
  public:
    FBLAS(int verbose=0);
    
    // allocate/free the Matrix structure fData data array using malloc, calloc
    // and free. 
    template<class T, bool isColMajor>
    void Malloc(Matrix<T,isColMajor>& m);
    template<class T, bool isColMajor>
    void Calloc(Matrix<T,isColMajor>& m);
    template<class T, bool isColMajor>
    void Free(Matrix<T,isColMajor>& m);

    // only for MT supported BLAS (MKL, OpenBLAS)
    void SetNumThreads(int nthreads, int verbose=0);

    //
    // BLAS Level 3
    //
    // XDGEMM (X={s=float,g=double}) : C = alpha A^T B^T + beta C 
    template < class T >
    void   XGEMM(Matrix<T>& A, Matrix<T>& B, Matrix<T>& C, T alpha=1., T beta=1.,
                 bool isTransA=false, bool isTransB=false); 
    // note: it's equivalent to XGEMM(Matrix<T, ture>& A...) because 2nd template 
    //       argument has a default 'true' value
    template < class T >             
    void   xgemm(char* trA, char* trB, int* m, int* n, int* k, T* alpha, T* A, 
                int* ldA, T* B, int* ldB, T* beta, T* C, int* ldC);
                  

    //
    // LAPACK: 
    //
    // LAPACK-DGESVD and LAPACK-SGESVD for SVD: a general and a special version
    template < class T >
    void   XGESVD(Matrix<T>& A,  Matrix<T>& SIGMA, const char* JOBU, 
                  const char* JOBVT, Matrix<T>& U, Matrix<T>& VT);
    // special version with JOBU='O' and JOBVT=N''
    template < class T >
    void   XGESVD(Matrix<T>& A, Matrix<T>& SIGMA);
    //
    template < class T >
    void   xgesvd(char* jobu, char* jobvt, int* m, int* n, T* A, int* ldA, 
                  T* SIGMA, T* U, int* ldU, T* VT, int* ldVT, 
                  T* work, int* lwork, int* info);

    //
    // LAPACK-DGEQRF and LAPACK-SGEQRF for A=QR factorization of general matrix A
    template < class T >
    void    XGEQRF(Matrix<T>& A,  Matrix<T>& TAU);
    template < class T >
    void    xgeqrf(int* m, int* n, T* A, int* ldA, T* TAU, 
                   T* work, int* lwork, int* info);

    //
    // LAPACK-DORMQR and LAPACK-SORMQR for multipying a matrix with Q from QR
    template < class T >
    void    XORMQR(Matrix<T>& C,  Matrix<T>& A, Matrix<T>& TAU, 
                   bool isQLeft=true, bool isTransQ=false);
    template < class T >
    void    xormqr(char* side, char* trans, int* m, int* n, int* k, T* A, 
                   int* ldA, T* TAU, T* C, int* ldC,
                   T* work, int* lwork, int* info);
    
    //
    // LAPACK-DORGQR and LAPACK-SORGQR to form the matrix Q from QR
    template < class T >
    void    XORGQR(Matrix<T>& A, Matrix<T>& TAU);
    template < class T >
    void    xorgqr(int* m, int* n, int* k, T* A, int* ldA, T* TAU,
                   T* work, int* lwork, int* info);

    //
    // LAPACK-DSYSV and LAPACK-SSYSV for solving AX=B with A square, symmetric
    template < class T >
    void    XSYSV(Matrix<T>& A,  Matrix<T>& B, bool isUplo=true);
    template < class T >
    void    xsysv(char* uplo, int* n, int* nrhs, T* A, int* ldA, int* ipiv, T* B, 
                  int* ldB, T* work, int* lwork, int* info);


    //
    // LAPACK-DSYEVR and LAPACK-SSYEVR for selected eigenvalues and optionally 
    // eigenvectors of the real, symmetric matrix A
    template < class T >
    int    XSYEVR(Matrix<T>& A,  Matrix<T>& EIGENVALS, 
                  int whichEigenValue=0, T minEigenVal=0., T maxEigenVal=0., 
                  bool isUploA=true, T abstol=-1.);
    template < class T >
    int    XSYEVR(Matrix<T>& A,  Matrix<T>& EIGENVALS, Matrix<T>& EIGENVECTS, 
                  int whichEigenValue=0, T minEigenVal=0., T maxEigenVal=0., 
                  bool isUploA=true, T abstol=-1.);    
    template < class T >
    void    xsyevr(char* jobz, char* range, char* uplo, int* n, T* A, int* ldA, 
                   T* vL, T* vU, int* iL, int* iU, T* tol, int* m, T* W, T *Z, 
                   int* ldZ, int* isuppz, T* work, int* lwork, int* iwork, 
                   int* liwork, int* info);

  private:
    // only for MKL
    int kMKLAlignment;  
}; 

// common (between the C- and FBLAS wrappers) function template implementations
#include "XBLAS.tpp"  
// external 'C' function declarations (C linkage to the corresponding Fortran objects)
#include "FBLAS.h"
// template implementations
#include "FBLAS.tpp"




#endif  // FBLAS_HH