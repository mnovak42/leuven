
//
// CBLAS function templates.
// This file is included into CBLAS.hh
//

// ========================================================================== //
// ==================== CBLAS-XGEMM:  ======================================= //
//
// computes: C = alpha A^T B^T + beta C;
// formats : both double and float data;
// order   : both col- and row-major matrix order
template <class T, bool isColMajor>
void CBLAS::XGEMM(Matrix<T,isColMajor>& A, Matrix<T,isColMajor>& B, Matrix<T,isColMajor>& C,
                  T alpha, T beta, bool isTransA, bool isTransB) {
  // all Matrix should have the same (either col or row major) order
//  assert ( A.IsColMajor()==B.IsColMajor() && "\n*** CBLAS::XGEMM: different Matrix Orders for A B***\n");
//  assert ( A.IsColMajor()==C.IsColMajor() && "\n*** CBLAS::XGEMM: different Matrix Orders for A C***\n");
  // Specifies row-major (C) or column-major (Fortran) data ordering.
  const CBLAS_ORDER     theOrder  = (A.IsColMajor()) ? CblasColMajor : CblasRowMajor;
  // Specifies whether to transpose matrix A.
  const CBLAS_TRANSPOSE theTransA = (isTransA)       ? CblasTrans    : CblasNoTrans;
  // Specifies whether to transpose matrix B.
  const CBLAS_TRANSPOSE theTransB = (isTransB)       ? CblasTrans    : CblasNoTrans;
  // Number of rows in matrices opt(A) and C.
  const size_t theM   = (isTransA) ? A.GetNumCols() : A.GetNumRows();
  // Number of cols in matrices opt(B) and C.
  const size_t theN   = (isTransB) ? B.GetNumRows() : B.GetNumCols();
  // Number of cols in matrix A; number of rows in matrix B.
  const size_t theK   = (isTransA) ? A.GetNumRows() : A.GetNumCols();
  // The size of the first dimention of matrix A[m][k]; m (col-major), k (row-major)
  const size_t theLDA = (A.IsColMajor()) ? A.GetNumRows() : A.GetNumCols();
  // The size of the first dimention of matrix B[k][n]; k (col-major), n (row-major)
  const size_t theLDB = (B.IsColMajor()) ? B.GetNumRows() : B.GetNumCols();
  // The size of the first dimention of matrix C[m][n]; m (col-major), n (row-major)
  const size_t theLDC = (C.IsColMajor()) ? C.GetNumRows() : C.GetNumCols();
  // invoke XGEMM (X={s=float,g=double}) : C = alpha A^T B^T + beta C
  xgemm(theOrder, theTransA, theTransB, theM, theN, theK, alpha, A.GetDataPtr(),
        theLDA, B.GetDataPtr(), theLDB, beta, C.GetDataPtr(), theLDC);
}


// ========================================================================== //
// ==================== LAPACKE-XGESVD: SVD ================================= //
//
// Two versions are available: one general and one with JOBU='O' and JOBVT='N'
//
// computes: A = U SIGMA V with A(M,N), U(M,M) V (N,N), SIGMA(M,N) diagonal
// formats : both double and float data;
// order   : both col- and row-major matrix order
//
//
template <class T, bool isColMajor>
void CBLAS::XGESVD(Matrix<T,isColMajor>& A,  Matrix<T,isColMajor>& SIGMA,
                   const char* JOBU, const char* JOBVT, Matrix<T,isColMajor>& U,
                   Matrix<T,isColMajor>& VT) {
  // ASSERT that all of them has the same row or col majority!!!: true by design
  // Specifies row-major (C) or column-major (Fortran) data ordering.
  const int theOrder = (A.IsColMajor()) ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR;
  // dimensions of input matrix A
  const size_t M    = A.GetNumRows();
  const size_t N    = A.GetNumCols();
  const size_t LDA  = A.IsColMajor() ? M : N;
  // dimension of U is M-by-M but the used part depends on JOBU:
  //  - 'A'  => U M-by-M i.e. all the M col-s required
  //  - 'S'  => U M-by-min(M,N) i.e. the min(M,N) col-s required
  //  - 'N' or O'  => not referenced (i.e. matrix U can be anything)
  const size_t LDU  = U.IsColMajor() ? U.GetNumRows() : U.GetNumCols();
  // dimensions of VT is N-by-N but the used part depends on JOBVT:
  //  - 'A'  => V^T N-by-N i.e. all the N row-s required
  //  - 'S'  => V^T min(M,N)-by-N i.e. the min(M,N) row-s required
  //  - 'N' or O'  => not referenced (i.e. matrix T can be anything)
  const size_t LDVT = VT.IsColMajor() ? VT.GetNumRows() : VT.GetNumCols();
  //
  // optimal work space query will be done inside
  int    info;
  // use consistent memory allocation
  Matrix<T> sb(std::min(M,N)-1);
  Malloc(sb);
  T* superb = sb.GetDataPtr();
  info = xgesvd(theOrder, *JOBU, *JOBVT, M, N, A.GetDataPtr(), LDA,
                SIGMA.GetDataPtr(), U.GetDataPtr(), LDU, VT.GetDataPtr(),
                LDVT, superb);
  if (info !=0 ) {
    printf ("\n*** CBLAS::XGESVD (LAPACKE_dgesvd): info !=0 ***\n");
    if (info < 0)
      printf (" ..... the %d-th argument had an illegal value\n", -info);
    else
      printf (" ..... DBDSQR did not converge with info = %d\n", info);
  }
  assert ( info==0 && "\n*** CBLAS::XGESVD (LAPACKE_dgesvd): info !=0 ***\n");

  // free allocated memory
  Free (sb);
}


template<class T, bool isColMajor>
void CBLAS::XGESVD(Matrix<T,isColMajor>& A, Matrix<T,isColMajor>& SIGMA) {
  // Specifies row-major (C) or column-major (Fortran) data ordering.
  const int theOrder = (A.IsColMajor()) ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR;
  // dimensions of input matrix A
  const size_t M    = A.GetNumRows();
  const size_t N    = A.GetNumCols();
  const size_t LDA  = A.IsColMajor() ? M : N;
  // U and VT are not refernced
  const size_t LDU  = 1;
  const size_t LDVT = 1;
  // the min(M,N) cols of A will contain the min(M,N) left singular vectors
  const char JOBU   = 'O';
  // no right singular vector computations
  const char JOBVT  = 'N';
  // U and VT are not referenced
  T* U         = nullptr;
  T* VT        = nullptr;
  //
  // optimal work space query will be done inside
  int     info;
  // use consistent memory allocation
  Matrix<T> sb(std::min(M,N)-1);
  Malloc(sb);
  T* superb = sb.GetDataPtr();
  info = xgesvd(theOrder, JOBU, JOBVT, M, N, A.GetDataPtr(), LDA,
                SIGMA.GetDataPtr(), U, LDU, VT, LDVT, superb);
  if (info !=0 ) {
    printf ("\n*** CBLAS::XGESVD (LAPACKE_dgesvd): info !=0 ***\n");
    if (info < 0)
      printf (" ..... the %d-th argument had an illegal value\n", -info);
    else
      printf (" ..... DBDSQR did not converge with info = %d\n", info);
  }
  assert ( info==0 && "\n*** CBLAS::XGESVD (LAPACKE_dgesvd): info !=0 ***\n");

  // free allocated memory
  Free (sb);
}


// ========================================================================== //
// ==================== LAPACKE-XGEQRF: QR ================================= //
//
// computes: A = QR (overwrites A with R; Q is in A and TAU)
// formats : both double and float data;
// order   : both col- and row-major matrix order
//
// size of TAU vector should be min(m,n)
template <class T, bool isColMajor>
void CBLAS::XGEQRF(Matrix<T,isColMajor>& A,  Matrix<T,isColMajor>& TAU) {
  // ASSERT if TAU is not a min(M,N) if matrix A has dimensions MxN
  assert (   std::min(A.GetNumRows(), A.GetNumCols()) + 1 == TAU.GetNumRows()+TAU.GetNumCols()
          && "\n*** CBLAS::XGEQRF: TAU must be a vector of min(M,N) ***\n");
  // Specifies row-major (C) or column-major (Fortran) data ordering.
  const int theOrder = (A.IsColMajor()) ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR;
  // dimensions of input matrix A
  const size_t M     = A.GetNumRows();
  const size_t N     = A.GetNumCols();
  const size_t LDA   = A.IsColMajor() ? M : N;
  //
  // optimal work space query will be done inside
  int info = xgeqrf(theOrder, M, N, A.GetDataPtr(), LDA, TAU.GetDataPtr());
  if (info !=0 ) {
    printf ("\n*** CBLAS::XGEQRF (LAPACKE_dgeqrf): info !=0 ***\n");
    if (info < 0)
      printf (" ..... the %d-th argument had an illegal value\n", -info);
  }
  assert ( info==0 && "\n*** CBLAS::XGESVD (LAPACKE_dgesvd): info !=0 ***\n");

}



// ========================================================================== //
// ==================== LAPACKE-XORMQR: MULTIPLY C BY Q FROM A QR =========== //
//
// computes: C = QC or Q^C or CQ or CQ^T depending on the input parameters
//           isQLeft, isTransQ
// formats : both double and float data;
// order   : both col- and row-major matrix order
//
//
template <class T, bool isColMajor>
void CBLAS::XORMQR(Matrix<T,isColMajor>& C,  Matrix<T,isColMajor>& A,
                   Matrix<T,isColMajor>& TAU, bool isQLeft, bool isTransQ) {
  // Specifies row-major (C) or column-major (Fortran) data ordering.
  const int theOrder = (isColMajor) ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR;
  // side of multiplication
  const char SIDE    = (isQLeft)  ? 'L' : 'R';
  // transpose Q before the multiplication
  const char TRANS   = (isTransQ) ? 'T' : 'N';
  // dimensions of input matrix C
  const size_t M     = C.GetNumRows();
  const size_t N     = C.GetNumCols();
  // number of elementary refractors stored in Q (and TAU)
  const size_t K     = std::min(A.GetNumRows(), A.GetNumCols());
  // leading dimension of matrix A and C
  const size_t LDA   = isColMajor ? A.GetNumRows() : A.GetNumCols();
  const size_t LDC   = isColMajor ? C.GetNumRows() : C.GetNumCols();
  //
  // optimal work space query will be done inside
  int info = xormqr(theOrder, SIDE, TRANS, M, N, K, A.GetDataPtr(), LDA,
                    TAU.GetDataPtr(), C.GetDataPtr(), LDC);
  if (info !=0 ) {
    printf ("\n*** CBLAS::XORMQR (LAPACKE_?ormqr): info !=0 ***\n");
    if (info < 0)
      printf (" ..... the %d-th argument had an illegal value\n", -info);
  }
  assert ( info==0 && "\n*** CBLAS::XORMQR (LAPACKE_?ormqr): info !=0 ***\n");

}


// ========================================================================== //
// ===== LAPACKE-XORGQR: FORMS THE MATRIX Q AFTER QR FACTORISATION =========== //
//
// computes: Q after A=QR giving the resulted A and tau (overwrites input A with Q)
// formats : both double and float data;
// order   : both col- and row-major matrix order
//
//
template <class T, bool isColMajor>
void CBLAS::XORGQR(Matrix<T,isColMajor>& A, Matrix<T,isColMajor>& TAU) {
  // Specifies row-major (C) or column-major (Fortran) data ordering.
  const int theOrder = (isColMajor) ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR;
  // dimensions of input matrix A
  const size_t M     = std::max(A.GetNumRows(), A.GetNumCols());
  const size_t N     = std::min(A.GetNumRows(), A.GetNumCols());
  const size_t K     = N;
  // leading dimension of matrix A
  const size_t LDA   = isColMajor ? A.GetNumRows() : A.GetNumCols();
  //
  // optimal work space query will be done inside
  int info = xorgqr(theOrder, M, N, K, A.GetDataPtr(), LDA, TAU.GetDataPtr());
  if (info !=0 ) {
    printf ("\n*** CBLAS::XORGQR (LAPACKE_?orgqr): info !=0 ***\n");
    if (info < 0)
      printf (" ..... the %d-th argument had an illegal value\n", -info);
  }
  assert ( info==0 && "\n*** CBLAS::XORGQR (LAPACKE_?orgqr): info !=0 ***\n");
}


// ========================================================================== //
// ==================== LAPACKE-XSYSV: SOLVE AX=B WITH A NxN SYMMETRIC ====== //
//
// computes: X such that AX = B
// formats : both double and float data;
// order   : both col- and row-major matrix order
//
//
template <class T, bool isColMajor>
void CBLAS::XSYSV(Matrix<T,isColMajor>& A,  Matrix<T,isColMajor>& B, bool isUplo) {
  // ASSERT if A and B has different number of rows
  assert ( A.GetNumRows() == A.GetNumCols() && "\n*** CBLAS::XSYS: A must be a square matrix ***\n");
  assert ( A.GetNumRows() == B.GetNumRows() && "\n*** CBLAS::XSYS: A and B must have same number of rows ***\n");
  // Specifies row-major (C) or column-major (Fortran) data ordering.
  const int theOrder = (isColMajor) ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR;
  // A is symmetric: either the upper or the lower  triangular part is given
  const char UPLO    = (isUplo)  ? 'U' : 'L';
  // dimensions of input matrix A (NxN square) and B (NxNRHS)
  const size_t N     = A.GetNumRows();
  const size_t NRHS  = B.GetNumCols();
  // leading dimension of matrix A (square so both are N)
  const size_t LDA   = N;
  const size_t LDB   = isColMajor ? B.GetNumRows() : B.GetNumCols();
  //
  // array to store infomation on the interchanges (not used later)
  int*  ipiv = (int*) malloc(sizeof(int)*N);
  //
  // optimal work space query will be done inside
  int info = xsysv(theOrder, UPLO, N, NRHS, A.GetDataPtr(), LDA, ipiv,
                   B.GetDataPtr(), LDB);
  if (info !=0 ) {
    printf ("\n*** CBLAS::XSYSV (LAPACKE_?sysv: info !=0 ***\n");
    if (info < 0)
      printf (" ..... the %d-th argument had an illegal value\n", -info);
    else
      printf (" ..... ?SYSV = %d factorization is completed, but D is singular \n", info);
  }
  assert ( info==0 && "\n*** CBLAS::XSYSV (LAPACKE_?sysv): info !=0 ***\n");

  // free allocated memory
  free(ipiv);
}



// ========================================================================== //
// ========== LAPACKE-XSYEVR: EIGENVALUES/VECTORS OF REAL SYM. MATRIX ======= //
//
// computes: all(whichEigenValue=0) or selected (based on min/max values or
//           indices whichEigenValue = 1 or 2) eigenvalues of the real symmetric
//           matrix A (NxN). Returns with the number of eigenvalues found M, that
//           wiil be stored in the EIGENVALS matrix on exit as the first M
//           elements in ascending order.
//           EIGENVALS matrix must be a single col/row matrix with N elements.
// formats : both double and float data;
// order   : both col- and row-major matrix order
template<class T, bool isColMajor>
int CBLAS::XSYEVR(Matrix<T,isColMajor>& A,  Matrix<T,isColMajor>& EIGENVALS,
                  int whichEigenValue, T minEigenVal, T maxEigenVal,
                  bool isUploA, T abstol) {
  // ASSERT if A is not a square matrix
  assert ( A.GetNumRows() == A.GetNumCols() && "\n*** CBLAS::XSYEVR: A must be a square matrix ***\n");
  // Specifies row-major (C) or column-major (Fortran) data ordering.
  const int theOrder = (isColMajor) ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR;
  // A is symmetric: either the upper or the lower triangular part is given
  const char UPLO    = (isUploA)  ? 'U' : 'L';
  // Dimensions of input matrix A (NxN square) and EIGENVAL (single col/row with N)
  const size_t N     = A.GetNumRows();
  const size_t NEIG  = (isColMajor) ? EIGENVALS.GetNumRows() : EIGENVALS.GetNumCols();
  assert ( N==NEIG && "\n*** CBLAS::XSYEVR: EIGENVALS should be a single col/row matrix with the same size as the input square matrix A. ***\n");
  (void)NEIG;
  // Leading dimension of matrix A (square so both are N)
  const size_t LDA   = N;
  // If the range of the required eignevalues are given as min/max index:
  //  - determine lower/upper indices of the required eigenvlaues
  const size_t iL    = (whichEigenValue==2) ? static_cast<size_t>(minEigenVal) : 0;
  const size_t iU    = (whichEigenValue==2) ? static_cast<size_t>(maxEigenVal) : 0;
  // If the range of the required eignevalues are given as min/max values:
  //  - determine lower/upper values of the required eigenvlaues
  const T      vL    = (whichEigenValue==1) ? minEigenVal : 0.;
  const T      vU    = (whichEigenValue==1) ? maxEigenVal : 0.;
  // All eigenvalues are required otherwise:
  const char RANGE   = (whichEigenValue==0  ? 'A' : (whichEigenValue==1 ? 'V' : 'I'));
  // Only eignevalues are required in this method (i.e. no eigenvectors)
  const char  JOBZ   = 'N';
  // Set absolute tolerance
  //abstol = ()
  // Number of eigenvalues found (on exit)
  int            M   = 0;
  // The followings are not referenced since eigenvectors are not required here
  T*             Z   = nullptr;
  size_t       LDZ   = 1;
  int*        ISUPPZ = nullptr;
  //
  // optimal work space query will be done inside
  int info = xsyevr(theOrder, JOBZ, RANGE, UPLO, N, A.GetDataPtr(), LDA, vL, vU,
                    iL, iU, abstol, &M, EIGENVALS.GetPtrToBlock(0), Z, LDZ, ISUPPZ);
  if (info !=0 ) {
    printf ("\n*** CBLAS::XSYEVR (LAPACKE_?syevr: info !=0 ***\n");
    if (info < 0)
      printf (" ..... the %d-th argument had an illegal value\n", -info);
    else
      printf (" ..... ?SYEVR = %d internal error. \n", info);
  }
  assert ( info==0 && "\n*** CBLAS::XSYEVR (LAPACKE_?syevr): info !=0 ***\n");
  //
  return M;
}
//
// computes: same as above, but computes the eigenvectors that corresponds to
//           the requested eigenvalues. These M, orthonormal eigenvectors are
//           stored in the M cols of the EIGENVECTS matrix: the i-th col contain
//           the eigenvector that corresponds to the eigenvalue stored at
//           EIGENVALS[i].
//           EIGENVECTS matrix must be an (NxM) matrix where M depends on the
//           number of the eigenvalues found (M=N if whichEigenValue=0 and
//           M=iU-iL+1 if whichEigenValue=1)
// formats : both double and float data;
// order   : both col- and row-major matrix order
template<class T, bool isColMajor>
int CBLAS::XSYEVR(Matrix<T,isColMajor>& A,  Matrix<T,isColMajor>& EIGENVALS,
                  Matrix<T,isColMajor>& EIGENVECTS, int whichEigenValue,
                  T minEigenVal, T maxEigenVal, bool isUploA, T abstol) {
  // ASSERT if A is not a square matrix
  assert ( A.GetNumRows() == A.GetNumCols() && "\n*** CBLAS::XSYEVR: A must be a square matrix ***\n");
  // Specifies row-major (C) or column-major (Fortran) data ordering.
  const int theOrder = (isColMajor) ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR;
  // A is symmetric: either the upper or the lower triangular part is given
  const char UPLO    = (isUploA)  ? 'U' : 'L';
  // Dimensions of input matrix A (NxN square) and EIGENVAL (single col/row with N)
  const size_t N     = A.GetNumRows();
  const size_t NEIG  = (isColMajor) ? EIGENVALS.GetNumRows() : EIGENVALS.GetNumCols();
  assert ( N==NEIG && "\n*** CBLAS::XSYEVR: EIGENVALS should be a single col/row matrix with the same size as the input square matrix A. ***\n");
  (void)NEIG;
  // Leading dimension of matrix A (square so both are N)
  const size_t LDA   = N;
  // If the range of the required eignevalues are given as min/max index:
  //  - determine lower/upper indices of the required eigenvlaues
  const size_t iL    = (whichEigenValue==2) ? static_cast<size_t>(minEigenVal) : 0;
  const size_t iU    = (whichEigenValue==2) ? static_cast<size_t>(maxEigenVal) : 0;
  // If the range of the required eignevalues are given as min/max values:
  //  - determine lower/upper values of the required eigenvlaues
  const T      vL    = (whichEigenValue==1) ? minEigenVal : 0.;
  const T      vU    = (whichEigenValue==1) ? maxEigenVal : 0.;
  // All eigenvalues are required otherwise:
  const char RANGE   = (whichEigenValue==0  ? 'A' : (whichEigenValue==1 ? 'V' : 'I'));
  // Make sure that the EIGENVECTS matrix has proper dimensions
  assert ( N==EIGENVECTS.GetNumRows() && "\n*** CBLAS::XSYEVR: EIGENVECTS matrix should have same number of row as the input square matrix A. ***\n");
 // assert ( whichEigenValue==0 && N==EIGENVECTS.GetNumCols() && "\n*** CBLAS::XSYEVR: EIGENVECTS matrix should have enough cols to store the N eigenvectors.(all eigens was required). ***\n");
 // assert ( whichEigenValue==2 && (iU-iL+1)>=EIGENVECTS.GetNumCols() && "\n*** CBLAS::XSYEVR: EIGENVECTS matrix should have enough cols to store the iU-iL+1 eigenvectors. ***\n");
  // Leading dimension of matrix EIGENVECTS
  const size_t LDZ   = (isColMajor) ? EIGENVECTS.GetNumRows() : EIGENVECTS.GetNumCols();
  // Both eignevalues and corresponding eigenvectors are required in this method
  const char  JOBZ   = 'V';
  // Set absolute tolerance
  //abstol = ()
  // Number of eigenvalues found (on exit)
  int            M   = 0;
  // We won't use this support infomation
  int*  ISUPPZ       = (int*)malloc(sizeof(int)*2*N);
  //
  // optimal work space query will be done inside
  int info = xsyevr(theOrder, JOBZ, RANGE, UPLO, N, A.GetDataPtr(), LDA, vL, vU,
                    iL, iU, abstol, &M, EIGENVALS.GetPtrToBlock(0),
                    EIGENVECTS.GetDataPtr(), LDZ, ISUPPZ);
  if (info !=0 ) {
    printf ("\n*** CBLAS::XSYEVR (LAPACKE_?syevr: info !=0 ***\n");
    if (info < 0)
      printf (" ..... the %d-th argument had an illegal value\n", -info);
    else
      printf (" ..... ?SYEVR = %d internal error. \n", info);
  }
  assert ( info==0 && "\n*** CBLAS::XSYEVR (LAPACKE_?syevr): info !=0 ***\n");
  //
  free(ISUPPZ);
  //
  return M;
}
