
// -------------------------------------------------------------------------- //
//                                                                            //
// There are two wrappers for using different BLAS implemetations:            // 
//  - CBLAS that is used USE_CBLAS is defined (i.e. compile with -DUSE_CBLAS) //
//  - FBLAS that is used USE_FBLAS is defined (i.e. compile with -DUSE_FBLAS) // 
//                                                                            //
// CBLAS is a wrapper for using different BLAS implementations. Some BLAS     //
// implementations (e.g. MKL, OpenBLAS) include the C-BLAS C-interface that   //
// makes possible to use either ROW- or COLUMN-MAJOR order. The CBLAS wrapper //
// provides this flexibility by calling the BLAS functions of the given       //
// implementation through its built-in C-BLAS interface. This makes possible  //
// to use either ROW- or COLUMN-MAJOR order with the restriction that not all //
// BLAS implementations can be used through this CBALS interface (only those  //
// that have their C-BLAS interface).                                         //
//
// Note, that an other wrapper for using different BLAS implementations, the  //
// FBLAS wrapper, is also available (by -DUSE_FBLAS instead of -DUSE_CBLAS).  // 
// When this FBLAS wrapper is selected instead of the CBLAS, the BLAS routines//
// are called DIRECTLY through their Fortarn interface. Since all different   //
// BLAS implemetations supports this Fortran style interface it's possible to //
// use any of the BLAS implementations through this wrapper. The consequence  //
// of using directly the Fortran interface is that the Fortran style ordering,//
// COLUMN-MAJOR ordering needs to be used and cannot be changed.              //
// -------------------------------------------------------------------------- //

#ifndef CBLAS_HH
#define CBLAS_HH

#include "definitions.hh"

#include "Matrix.hh"

#if USE_MKL_BLAS
  #include "mkl_service.h"
  #include "mkl_cblas.h"
  #include "mkl_lapacke.h"
#elif  USE_OPEN_BLAS
  #include "cblas.h"
  #include "lapacke.h"
#endif 


class CBLAS {
  public:
    CBLAS(int verbose=0);
    
    // allocate/free the Matrix structure fData data array using mkl_malloc, 
    // mkl_calloc and mkl_free. 
    template<class T, bool isColMajor>
    void Malloc(Matrix<T,isColMajor>& m);
    template<class T, bool isColMajor>
    void Calloc(Matrix<T,isColMajor>& m);
    template<class T, bool isColMajor>
    void Free(Matrix<T,isColMajor>& m);

    // e.g. MKL, OpenBLAS are MT
    void SetNumThreads(int nthreads, int verbose=0);

    //
    // BLAS Level 3
    //
    // invoke XDGEMM (X={s=float,g=double}) : C = alpha A^T B^T + beta C
    template <class T, bool isColMajor>
    void    XGEMM(Matrix<T,isColMajor>& A, Matrix<T,isColMajor>& B, Matrix<T,isColMajor>& C, 
                  T alpha=1., T beta=1., bool isTransA=false, bool isTransB=false);
    template <class T>
    void    xgemm(CBLAS_ORDER ord, CBLAS_TRANSPOSE trA, CBLAS_TRANSPOSE trB, 
                  int m, int n, int k, T alpha, T* A, int ldA, T* B, int ldB, 
                  T beta, T* C, int ldC);              
                 
    //             
    // LAPACK
    //
    
    //
    // SVD DECOMPOZITION
    //
    // Invoke XGESVD (SVD of the real m-by-n matrix A, optionaly left and/or right singular vectors)
    //   A = U SIGMA V^T
    //   - SIGMA is an m-by-n matrix which is zero except for its min(m,n) diagonal elements
    //   - U is an m-by-m orthogonal matrix. 
    //   - V^T (V transposed) is an n-by-n orthogonal matrix
    //  The diagonal elements of SIGMA are the singular values of A; they are real 
    //  and non-negative, and are returned in descending order. The first min(m,n) 
    //  columns of U and V are the left and right singular vectors of A.
    //  NOTE: The routine returns V^T, not V.
    //
    // The content of A will be destroyed or replaced:
    //  - JOBU ='O' : A is overwritten with the first min(m,n) columns of U 
    //              (the left singular vectors, stored columnwise)
    //  - JOBVT='O' : A is overwritten with the first min(m,n) rows of V**T 
    //              (the right singular vectors, stored rowwise)
    //
    //  JOBU: 
    //   - 'A':  all M columns of U are returned in array U:
    //   - 'S':  the first min(m,n) columns of U (the left singular
    //         vectors) are returned in the array U;
    //   - 'O':  the first min(m,n) columns of U (the left singular
    //         vectors) are overwritten on the array A;
    //   - 'N':  no columns of U (no left singular vectors) are
    //         computed.
    //
    // JOBVT:
    //   - 'A':  all N rows of V**T are returned in the array VT;
    //   - 'S':  the first min(m,n) rows of V**T (the right singular
    //           vectors) are returned in the array VT;
    //   - 'O':  the first min(m,n) rows of V**T (the right singular
    //           vectors) are overwritten on the array A;
    //   - 'N':  no rows of V**T (no right singular vectors) are
    //           computed.
    //
    // NOTE: size of U, SIGMA, VT should be set inside this method !!!!
    // S: is DOUBLE PRECISION array, dimension (min(M,N))
    //    The singular values of A, sorted so that S(i) >= S(i+1)
    //
    template<class T, bool isColMajor>
    void    XGESVD(Matrix<T,isColMajor>& A,  Matrix<T,isColMajor>& SIGMA, 
                   const char* JOBU, const char* JOBVT, Matrix<T,isColMajor>& U,
                   Matrix<T,isColMajor>& VT);
    // similar to the general version above but with fix JOBU='O' JOBVT='N' 
    // i.e. on exit, the cols of the M-by-N input matrix A will be overwritten 
    // by the first min(M,N) left sigular vetors; SIGMA will be min(M,N)-by-1
    // and will contain the corresponding singular values in S(i)>=S(i+1)..
    template<class T, bool isColMajor>
    void    XGESVD(Matrix<T,isColMajor>& A, Matrix<T,isColMajor>& SIGMA);
    //
    template<class T>
    int     xgesvd(int ord, char jobu, char jobvt, int m, int n, T* A, int ldA, 
                   T* SIGMA, T* U, int ldU, T* VT, int ldVT, T* sup);
 
    //
    // QR FACTORIZATION:
    //
    // Computes a QR factorization of a general real M-by-N matrix A:
    //  A = Q * R.
    //
    //  Note: 
    //  - the orthogonal matrix Q (M, min(M,N)) is not formed explicitely
    //  - the result is stored in A that is overwritten on exit: 
    //      = on entry, the M-by-N matrix 
    //      = on exit:
    //       :: elements on and above the diagonal of the array contain the 
    //          min(M,N)-by-N upper trapezoidal matrix R (R is upper triangular 
    //          if M >= N)
    //       :: elements below the diagonal, with the array TAU min(m,n), 
    //          represent the orthogonal matrix Q (M, min(M,N)) as a product of 
    //          min(M,N) elementary reflectors   
    //
    // The orthogonal matrix Q is stored as a product of elementary reflectors:
    //
    //   Q = H(1) H(2) . . . H(k), where k = min(m,n).
    //
    // Each H(i) has the form of  H(i) = I - tau * v * v', where tau is a real 
    // scalar, and v is a real vector with v(1:i-1) = 0 and v(i) = 1; v(i+1:m) 
    // is stored on exit in A(i+1:m,i), and tau in TAU(i).
    //
    // Special functions are available that can make use of the matrix Q in this
    // form e.g. to multiply (left, right) a real matrix with Q (or ist transpose)
    // one can use the LAPACK d/s-ormqr function (available here as XORMQR) or 
    // one can use the LAPACK d/s-orgqr function to form the matrix Q explicitely.
    //
    template<class T, bool isColMajor>
    void    XGEQRF(Matrix<T,isColMajor>& A,  Matrix<T,isColMajor>& TAU);
    template<class T>
    int     xgeqrf(int ord, int m, int n, T* A, int ldA, T* TAU);


    // 
    // USE THE ORTHOGONAL Q MATRIX FROM A PREVIOUSLY PERFORMED QR FACTORIZATION:
    // 
    // Multiply matrix C with a matrix Q obtained as the QR factorization of a 
    // matrix A (A=QR, see above):
    //  after the QR factorization of a matrix A (MxN), the resulted orthogonal 
    //  Q (M,min(M,N)) matrix is not formed explicitely but stored in the below
    //  diagonals of A together with the TAU min(M,N) vector. A matrix C (with 
    //  the appropriate dimensions) can multiplied later by using this method 
    //  providing the resluted matrix A and vector TAU.
    // 
    //
    // isQLeft  = true  => Left multiply by Q or Q^T (Right otherwise)
    // isTransQ = false => Mutiply with Q (with Q^T otherwise)
    template<class T, bool isColMajor>
    void    XORMQR(Matrix<T,isColMajor>& C,  Matrix<T,isColMajor>& A, 
                   Matrix<T,isColMajor>& TAU, bool isQLeft=true, bool isTransQ=false);
    template<class T>
    int     xormqr(int ord, char side, char trans, int m, int n, int k, T* A, 
                   int ldA, const T* TAU, T* C, int ldC);

    // 
    // FORMS THE ORTHOGONAL Q MATRIX FROM A PREVIOUSLY PERFORMED QR FACTORIZATION:
    //
    // Unfortunatly, ?ormqr (see above) do not give consistent results with 
    // intel MKL. It means, that when XORMQR is used to compute QA and QB such 
    // that the first n-columns of A and B are identical, the first n-columns of 
    // the results of QA and QB won't be exactly the same (as it should be) when 
    // intek MKL ?ormqr is used. (but everything is good with openBLAS). 
    // In order to get rid of this divergence, the ?ormgqr is used to form the 
    // Q matrix explicitely, then the multiplication can be performed separately.
    // 
    // If the QR factorization of A (M,N) was performed, then this methods forms 
    // the Q (M,N) orthogonal matrix (only its first N columns) by giving the 
    // matrix A as resulted after the QR factorisation and the corresponding 
    // Tau vetor. The result, i.e. matrix Q will be written into A on exit.
    template<class T, bool isColMajor>
    void    XORGQR(Matrix<T,isColMajor>& A, Matrix<T,isColMajor>& TAU);
    template<class T>
    int     xorgqr(int ord, int m, int n, int k, T* A, int ldA, const T* TAU);


    //
    // SOLVE SYSTEM OF LINEAR EQUATIONS WITH SYMMETRIC CEF. MATRIX A
    //
    // Solves for X the AX=B system with A being an N-by-N symmetric matrix and 
    // columns of B the individual right-hand sides. The solution will be in the 
    // columns of X. (The diagonal pivoting method is used. See ?posv for positve
    // definite matrices that will use the Cholesky decomposition.)
    //
    // The results are stored in B at the end and A is destroyed (overwritten). 
    template<class T, bool isColMajor>
    void    XSYSV(Matrix<T,isColMajor>& A,  Matrix<T,isColMajor>& B, bool isUplo=true);
    template<class T>
    int     xsysv(int ord, char uplo, int n, int nrhs, T* A, int ldA, int* ipiv, 
                  T* B, int ldB);
    

   // computes eigenvalues : all, indexed or values 
   // computes eigenvalues and eigenvectors : all, indexed or values

    //
    // COMPUTES SELECTED EIGENVALUES AND, OPTIONALLY, EIGENVECTORS OF A REAL SY
    // MATRIX. EIGENVALUES AND EIGENVECTORS CAN BE SELECTED BY GIVING EITHER THE 
    // RANGE OF VALUES OR RANGE OF INDICES.
    //
    // The lower or upper triangualr part of the input matix (depending which 
    // one was given) will be destroyed. 
    //  bool isEigenvaluesOnly, int whichEigen, bool isUplo
    // JOBZ ==> bool isEigenvaluesOnly = true
    //    = 'N':  Compute eigenvalues only;
    //    = 'V':  Compute eigenvalues and eigenvectors.
    // RANGE ==> int whichEigen = {0, 1, 2}
    //    = 'A': all eigenvalues will be found.
    //    = 'V': all eigenvalues in the half-open interval (VL,VU]
    //           will be found.
    //    = 'I': the IL-th through IU-th eigenvalues will be found.
    // UPLO ==> bool isUplo = flase
    //    = 'U':  Upper triangle of A is stored;
    //    = 'L':  Lower triangle of A is stored.
    // VL   ==> T minEigenVal
    //    Referenced only if RANGE='V' (=> 1), the lower bound of the eigenvalue 
    //    interval to be searched. VL <  VU.
    // VU   ==> T maxEigenVal
    //    Referenced only if RANGE='V' (=> 1), the upper bound of the eigenvalue 
    //    interval to be searched. VL <  VU.
    // IL   ==> int minEigenIndex
    //    Referenced only if RANGE='I' (=> 2), the index of the smallest 
    //    eigenvalue to be returned. 1 <= IL <= IU <= N 
    // IU   ==> int maxEigenIndex
    //    Referenced only if RANGE='I' (=> 2), the index of the largest 
    //    eigenvalue to be returned. 1 <= IL <= IU <= N 
    // ABSTOL ==> int absTolFlag = {0, 1, 2}
    //    The absolute error tolerance for the eigenvalues. 
    //    = > 0 An approximate eigenvalue is accepted as converged when it is 
    //          determined to lie in an interval [a,b] of width less than or equal to
    //
    //              ABSTOL + EPS *   max( |a|,|b| ) ,
    //
    //           where EPS is the machine precision.
    //    = = 0 EPS*|T|  will be used in place of ABSTOL, where |T| is the 1-norm 
    //          of the tridiagonal matrix obtained by reducing A to tridiagonal form.
    //    = < 0 If the highest precision is required: will be set to "Safe minimum"
    //
    // ==== OUTPUT =====
    // M  ==> int& M
    //    The number of eigenvalues found.  0 <= M <= N.
    //    = if RANGE=A (whichEigen = 0) => M=N
    //    = if RANGE=I (whichEigen = 2) => M=IU-IL+1
    //    = if RANGE=V (whichEigen = 1) => the exact value of M is not known in 
    //                                     advance (but for sure <= N)
    // ---eigenvalues---- 
    // W  ==> *W : (dim = N i.e. max) the first M elements contains the required 
    //         eigenvectors
    // --eigenvectors----
    // Z  ==> *Z : matrix that should contain M columns to store the required 
    //             orthonormal eigenvectors 
    //   = if JOBZ = N (isEigenvaluesOnly = true)  => it's not referenced
    //   = if JOBZ = V (isEigenvaluesOnly = false) => the first M columns of Z
    //               contain the orthonormal eigenvectors of the matrix A
    //               corresponding to the selected eigenvalues, with the i-th
    //               column of Z holding the eigenvector associated with W(i).
    //     Note: the user must ensure that at least M columns are supplied in 
    //           the array Z; if RANGE = 'V' (whichEigen = 1), the exact value 
    //           of M is not known in advance and an upper bound must be used.
    //           Supplying N columns is always safe.
    //
    // 1. eigenvalues only and whichEigenValue = {0,1,2} as all, min/max EigenVal 
    //    casted to int in case of whichEigenValue = 2, T abstol {>0, 0, <0}
    //    W a singe col(row) matrix with the size of N; and int& M vagy return with M 
    // 2. eigenvalues and eigenvectors: the same as above plus the matrix Z
    //
    template<class T, bool isColMajor>
    int    XSYEVR(Matrix<T,isColMajor>& A,  Matrix<T,isColMajor>& EIGENVALS, 
                  int whichEigenValue=0, T minEigenVal=0., T maxEigenVal=0., 
                  bool isUploA=true, T abstol=-1.);
    template<class T, bool isColMajor>
    int    XSYEVR(Matrix<T,isColMajor>& A,  Matrix<T,isColMajor>& EIGENVALS, 
                  Matrix<T,isColMajor>& EIGENVECTS, 
                  int whichEigenValue=0, T minEigenVal=0., T maxEigenVal=0., 
                  bool isUploA=true, T abstol=-1.);
    template<class T>
    int    xsyevr(int ord, char jobz, char range, char uplo, int n, T* A, int ldA, 
                  T vL, T vU, int iL, int iU, T tol, int* m, T* W, T *Z, int ldZ, 
                  int* isuppz);
  private:
    // only for MKL
    const int kMKLAlignment; 
    
};


#include "XBLAS.tpp"
#include "CBLAS.tpp"

#endif  // CBLAS_HH