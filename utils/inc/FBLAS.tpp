
//
// FBLAS function templates.
// This file is included into FBLAS.hh
//

// ========================================================================== //
// ==================== FBLAS-XGEMM:  ======================================= //
// 
// computes: C = alpha A^T B^T + beta C; 
// formats : both double and float data; 
// order   : only the (Fortran style) col-major matrix order is supported
//
template<class T>
void FBLAS::XGEMM(Matrix<T>& A, Matrix<T>& B, Matrix<T>& C, T alpha, T beta, 
                  bool isTransA, bool isTransB) {
  // all Matrix should have the same Col-major order !!! true by design
  // Specifies whether to transpose matrix A. 
  char theTransA = (isTransA) ? 'T' : 'N';
  // Specifies whether to transpose matrix B.
  char theTransB = (isTransB) ? 'T' : 'N';
  // Number of rows in matrices A and C.  
  int theM       = (int)A.GetNumRows(); 
  // Number of cols in matrices B and C. 
  int theN       = (int)B.GetNumCols(); 
  // Number of cols in matrix A; number of rows in matrix B.
  int theK       = (int)A.GetNumCols(); 
  // The size of the first dimention of matrix A[m][k]; m (col-major)
  int theLDA     = theM;
  // The size of the first dimention of matrix B[k][n]; k (col-major)
  int theLDB     = theK; 
  // The size of the first dimention of matrix C[m][n]; m (col-major)
  int theLDC     = theM;
  // invoke xgem (X={s=float,g=double}) : C = alpha A^T B^T + beta C  
  xgemm(&theTransA, &theTransB, &theM, &theN, &theK, &alpha, A.GetDataPtr(), 
        &theLDA, B.GetDataPtr(), &theLDB, &beta, C.GetDataPtr(), &theLDC);
} 



// ========================================================================== //
// ==================== FBLAS-XGESVD: SVD ================================= //
//
// Two versions are available: one general and one with JOBU='O' and JOBVT='N'
//
// computes: A = U SIGMA V with A(M,N), U(M,M) V (N,N), SIGMA(M,N) diagonal
// formats : both double and float data; 
// order   : only the (Fortran style) col-major matrix order is supported
//
template < class T >
void FBLAS::XGESVD(Matrix<T>& A,  Matrix<T>& SIGMA, const char* JOBU, 
                   const char* JOBVT, Matrix<T>& U, Matrix<T>& VT) {
  // dimensions of input matrix A
  int M   = A.GetNumRows();
  int N   = A.GetNumCols();    
  int LDA = M;
  // dimension of U is M-by-M but the used part depends on JOBU:
  //  - 'A'  => U M-by-M i.e. all the M col-s required
  //  - 'S'  => U M-by-min(M,N) i.e. the min(M,N) col-s required
  //  - 'N' or O'  => not referenced (i.e. matrix U can be anything)
  int LDU  = U.GetNumRows(); 
  // dimensions of VT is N-by-N but the used part depends on JOBVT: 
  //  - 'A'  => V^T N-by-N i.e. all the N row-s required
  //  - 'S'  => V^T min(M,N)-by-N i.e. the min(M,N) row-s required
  //  - 'N' or O'  => not referenced (i.e. matrix T can be anything)
  int LDVT = VT.GetNumRows();
  //
  // Query and allocate the optimal workspace 
  int    info  =  0;
  int    lwork = -1;
  T      wkopt = 0.;
  // constness...
  char jobu    = *JOBU;
  char jobvt   = *JOBVT;
  xgesvd(&jobu, &jobvt, &M, &N, A.GetDataPtr(), &LDA, SIGMA.GetDataPtr(), 
         U.GetDataPtr(), &LDU, VT.GetDataPtr(), &LDVT, &wkopt, &lwork, &info);
  lwork = (int) wkopt;
  // consistent memory allocation
  Matrix<T> mw(lwork); 
  Malloc(mw);
  T* work = mw.GetDataPtr();
  //
  // Compute SVD
  xgesvd(&jobu, &jobvt, &M, &N, A.GetDataPtr(), &LDA, SIGMA.GetDataPtr(), 
         U.GetDataPtr(), &LDU, VT.GetDataPtr(), &LDVT, work, &lwork, &info);
  //
  // Check for convergence or error
  if (info !=0 ) {
    printf ("\n*** FBLAS::XGESVD (LAPACK__dgesvd): info !=0 ***\n");
    if (info < 0) 
      printf (" ..... the %d-th argument had an illegal value\n", -info);
    else 
      printf (" ..... DBDSQR did not converge with info = %d\n", info);
  }
  assert ( info==0 && "\n*** FBLAS::XGESVD (LAPACK_dgesvd): info !=0 ***\n");
  //
  // free allocated memeory
  Free (mw);
}

template < class T >
void FBLAS::XGESVD(Matrix<T>& A, Matrix<T>& SIGMA) {
  // dimensions of input matrix A
  int M      = A.GetNumRows();
  int N      = A.GetNumCols();    
  int LDA    = M;
  // U and VT are not refernced
  int LDU    = 1;
  int LDVT   = 1;
  // the min(M,N) cols of A will contain the min(M,N) left singular vectors 
  char jobu  = 'O';
  // no right singular vector computations
  char jobvt = 'N';
  // U and VT are not referenced
  T* U       = nullptr;
  T* VT      = nullptr;
  //
  // Query and allocate the optimal workspace 
  int    lwork = -1;
  int    info  =  0;
  T      wkopt = 0.;
  xgesvd(&jobu, &jobvt, &M, &N, A.GetDataPtr(), &LDA, SIGMA.GetDataPtr(), 
         U, &LDU, VT, &LDVT, &wkopt, &lwork, &info);
  lwork = (int) wkopt;
  // consistent memory allocation
  Matrix<T> mw(lwork); 
  Malloc(mw);
  T* work = mw.GetDataPtr();
  //
  // Compute SVD
  xgesvd(&jobu, &jobvt, &M, &N, A.GetDataPtr(), &LDA, SIGMA.GetDataPtr(), 
         U, &LDU, VT, &LDVT, work, &lwork, &info);
  //
  // Check for convergence or error
  if (info !=0 ) {
    printf ("\n*** FBLAS::XGESVD (LAPACK__dgesvd): info !=0 ***\n");
    if (info < 0) 
      printf (" ..... the %d-th argument had an illegal value\n", -info);
    else 
      printf (" ..... DBDSQR did not converge with info = %d\n", info);
  }
  assert ( info==0 && "\n*** FBLAS::XGESVD (LAPACK_dgesvd): info !=0 ***\n");
  //
  // free allocated memeory
  Free (mw);    
}



// ========================================================================== //
// ==================== LAPACK-XGEQRF: QR ================================= //
//
// computes: A = QR (overwrites A with R; Q is in A and TAU)
// formats : both double and float data; 
// order   : only the (Fortran style) col-major matrix order is supported
template < class T >
void FBLAS::XGEQRF(Matrix<T>& A, Matrix<T>& TAU) {
  // ASSERT if TAU is not a min(M,N) if matrix A has dimensions MxN
  assert (   std::min(A.GetNumRows(), A.GetNumCols()) + 1 == TAU.GetNumRows()+TAU.GetNumCols() 
          && "\n*** FBLAS::XGEQRF: TAU must be a vector of min(M,N) ***\n");
  // dimensions of input matrix A
  int M     = A.GetNumRows();
  int N     = A.GetNumCols();    
  int LDA   = M;
  //
  // Query and allocate the optimal workspace 
  int    lwork = -1;
  int    info  =  0;
  T      wkopt = 0.;
  xgeqrf(&M, &N, A.GetDataPtr(), &LDA, TAU.GetDataPtr(), &wkopt, &lwork, &info);
  lwork = (int) wkopt;
  // consistent memory allocation
  Matrix<T> mw(lwork); 
  Malloc(mw);
  T* work = mw.GetDataPtr();
  //
  // Compute QR factorization
  xgeqrf(&M, &N, A.GetDataPtr(), &LDA, TAU.GetDataPtr(), work, &lwork, &info);
  //
  // Check return code
  if (info !=0 ) {
    printf ("\n*** FBLAS::XGEQRF (LAPACK_?geqrf): info !=0 ***\n");
    if (info < 0) 
      printf (" ..... the %d-th argument had an illegal value\n", -info);
  }
  assert ( info==0 && "\n*** FBLAS::XGEQRF (LAPACK_?geqrf): info !=0 ***\n");
  //
  // free allocated memeory
  Free (mw);    
}



// ========================================================================== //
// ==================== LAPACK-XORMQR: MULTIPLY C BY Q FROM A QR ============ //
//
// computes: C = QC or Q^C or CQ or CQ^T depending on the input parameters 
//           isQLeft, isTransQ
// formats : both double and float data; 
// order   : only the (Fortran style) col-major matrix order is supported
//
//
template <class T>
void FBLAS::XORMQR(Matrix<T>& C,  Matrix<T>& A, Matrix<T>& TAU, 
                   bool isQLeft, bool isTransQ) {    
  // side of multiplication
  char SIDE    = (isQLeft)  ? 'L' : 'R';
  // transpose Q before the multiplication
  char TRANS   = (isTransQ) ? 'T' : 'N';
  // dimensions of input matrix C
  int  M     = C.GetNumRows();
  int  N     = C.GetNumCols();    
  // number of elementary refractors stored in Q (and TAU) 
  int  K     = std::min(A.GetNumRows(), A.GetNumCols());
  // leading dimension of matrix A and C
  int  LDA   = A.GetNumRows();
  int  LDC   = C.GetNumRows();
  //
  // Query and allocate the optimal workspace 
  int    lwork = -1;
  int    info  =  0;
  T      wkopt = 0.;
  xormqr(&SIDE, &TRANS, &M, &N, &K, A.GetDataPtr(), &LDA, TAU.GetDataPtr(), 
         C.GetDataPtr(), &LDC, &wkopt, &lwork, &info);
  lwork = (int) wkopt;
  //std::cout<< "****** lwork = " << lwork << " vs N = " << N << std::endl;
  // consistent memory allocation
  Matrix<T> mw(lwork);
  Malloc(mw);
  T* work = mw.GetDataPtr();
  //
  // Compute the matrix multiplication
  xormqr(&SIDE, &TRANS, &M, &N, &K, A.GetDataPtr(), &LDA, TAU.GetDataPtr(), 
         C.GetDataPtr(), &LDC, work, &lwork, &info);
  //
  // Check return code
  if (info !=0 ) {
    printf ("\n*** FBLAS::XORMQR (LAPACK_?ormqr): info !=0 ***\n");
    if (info < 0) 
      printf (" ..... the %d-th argument had an illegal value\n", -info);
  }
  assert ( info==0 && "\n*** FBLAS::XORMQR (LAPACK_?ormqr): info !=0 ***\n");
  //
  // free allocated memeory
  Free (mw);    
}


// ========================================================================== //
// ==== LAPACK-XORGQR: FORMS THE MATRIX Q AFTER QR FACTORISATION ============ //
//
// computes: Q after A=QR giving the resulted A and tau (overwrites input A with Q)
// formats : both double and float data; 
// order   : only the (Fortran style) col-major matrix order is supported
//
//
template <class T>
void FBLAS::XORGQR(Matrix<T>& A, Matrix<T>& TAU) {    
  // dimensions of input matrix A (that will be Q)
  int  M     = std::max(A.GetNumRows(), A.GetNumCols());
  int  N     = std::min(A.GetNumRows(), A.GetNumCols());
  // number of elementary refractors stored in Q (and TAU) 
  int  K     = N;
  // leading dimension of matrix A
  int  LDA   = A.GetNumRows();
  //
  // Query and allocate the optimal workspace 
  int    lwork = -1;
  int    info  =  0;
  T      wkopt = 0.;
  xorgqr(&M, &N, &K, A.GetDataPtr(), &LDA, TAU.GetDataPtr(), &wkopt, &lwork, &info);
  lwork = (int) wkopt;
  // consistent memory allocation
  Matrix<T> mw(lwork);
  Malloc(mw);
  T* work = mw.GetDataPtr();
  //
  // Compute the matrix multiplication
  xorgqr(&M, &N, &K, A.GetDataPtr(), &LDA, TAU.GetDataPtr(), work, &lwork, &info);
  //
  // Check return code
  if (info !=0 ) {
    printf ("\n*** FBLAS::XORGQR (LAPACK_?orgqr): info !=0 ***\n");
    if (info < 0) 
      printf (" ..... the %d-th argument had an illegal value\n", -info);
  }
  assert ( info==0 && "\n*** FBLAS::XORGQR (LAPACK_?orgqr): info !=0 ***\n");
  //
  // free allocated memeory
  Free (mw);    
}


// ========================================================================== //
// ==================== LAPACK-XSYSV: SOLVE AX=B WITH A NxN SYMMETRIC ======= //
//
// computes: X such that AX = B 
// formats : both double and float data; 
// order   : only the (Fortran style) col-major matrix order is supported
//
//
template <class T>
void FBLAS::XSYSV(Matrix<T>& A,  Matrix<T>& B, bool isUplo) {    
  // ASSERT if A and B has different number of rows
  assert ( A.GetNumRows() == A.GetNumCols() && "\n*** FBLAS::XSYS: A must be a squared matrix ***\n");
  assert ( A.GetNumRows() == B.GetNumRows() && "\n*** FBLAS::XSYS: A and B must have same number of rows ***\n");
  // A is symmetric: either the upper or the lower  triangular part is given
  char UPLO = (isUplo)  ? 'U' : 'L';
  // dimensions of input matrix A (NxN squared) and B (NxNRHS)
  int N     = A.GetNumRows();
  int NRHS  = B.GetNumCols();
  // leading dimension of matrix A (squared so both are N)
  int LDA   = N;
  int LDB   = B.GetNumRows();
  //
  // array to store infomation on the interchanges (not used later)
  int*  ipiv = (int*) malloc(sizeof(int)*N);
  //
  // Query and allocate the optimal workspace 
  int    lwork = -1;
  int    info  =  0;
  T      wkopt =  0.;
  xsysv(&UPLO, &N, &NRHS, A.GetDataPtr(), &LDA, ipiv, B.GetDataPtr(), &LDB, 
        &wkopt, &lwork, &info);
  lwork = (int) wkopt;
  // consistent memory allocation
  Matrix<T> mw(lwork);
  Malloc(mw);
  T* work = mw.GetDataPtr();
  //
  // Compute the matrix multiplication
  xsysv(&UPLO, &N, &NRHS, A.GetDataPtr(), &LDA, ipiv, B.GetDataPtr(), &LDB, 
        work, &lwork, &info);
  if (info !=0 ) {
    printf ("\n*** FBLAS::XSYSV (LAPACK_?sysv: info !=0 ***\n");
    if (info < 0) 
      printf (" ..... the %d-th argument had an illegal value\n", -info);
    else 
      printf (" ..... ?SYSV = %d factorization is completed, but D is singular \n", info);
  }
  assert ( info==0 && "\n*** FBLAS::XSYSV (LAPACK_?sysv): info !=0 ***\n");            
  //
  // free allocated memory
  free (ipiv);  
  Free (mw);    
}  



// ========================================================================== //
// ========== LAPACK-XSYEVR: EIGENVALUES/VECTORS OF REAL SYM. MATRIX ======= //
//
// computes: all(whichEigenValue=0) or selected (based on min/max values or 
//           indices whichEigenValue = 1 or 2) eigenvalues of the real symmetric 
//           matrix A (NxN). Returns with the number of eigenvalues found M, that
//           will be stored in the EIGENVALS matrix on exit as the first M 
//           elements in ascending order.
//           EIGENVALS matrix must be a single col matrix with N elements. 
// formats : both double and float data; 
// order   : only the (Fortran style) col-major matrix order is supported
template < class T >
int FBLAS::XSYEVR(Matrix<T>& A,  Matrix<T>& EIGENVALS, int whichEigenValue, 
                  T minEigenVal, T maxEigenVal, bool isUploA, T abstol) {
  // A is symmetric: either the upper or the lower  triangular part is given
  char UPLO = (isUploA)  ? 'U' : 'L';
  // dimensions of input matrix A (NxN squared)
  int     N = A.GetNumRows();
  // Make sure that the EIGENVALS vector has the proper size
  assert ( N==EIGENVALS.GetNumRows() && "\n*** FBLAS::XSYEVR: EIGENVALS should be a single col matrix with the same size as the input square matrix A. ***\n");                  
  // Leading dimension of matrix A (squared so both are N)
  int   LDA = N;
  // If the range of the required eignevalues are given as min/max index:
  //  - determine lower/upper indices of the required eigenvlaues
  int    iL = (whichEigenValue==2) ? static_cast<int>(minEigenVal) : 0;
  int    iU = (whichEigenValue==2) ? static_cast<int>(maxEigenVal) : 0;
  // If the range of the required eignevalues are given as min/max values:
  //  - determine lower/upper values of the required eigenvlaues
  T      vL = (whichEigenValue==1) ? minEigenVal : 0.;
  T      vU = (whichEigenValue==1) ? maxEigenVal : 0.;
  // All eigenvalues are required otherwise:
  char RANGE = (whichEigenValue==0  ? 'A' : (whichEigenValue==1 ? 'V' : 'I'));
  // Only eigenvalues are required in this method (i.e. no eigenvectors)
  char JOBZ = 'N';
  // Set absolute tolerance
  //abstol = ()
  // Number of eigenvalues found (on exit)
  int     M = 0;
  // The followings are not referenced since eigenvectors are not required here
  T*      Z = nullptr;
  int   LDZ = 0;
  int* ISUPPZ = nullptr;
  //
  // Query and allocate the optimal workspace 
  int info   =  0;
  int lwork  = -1;
  int liwork = -1;
  int iwkopt =  0;  
  T   wkopt  =  0.;
  xsyevr(&JOBZ, &RANGE, &UPLO, &N, A.GetDataPtr(), &LDA, &vL, &vU, &iL, &iU, 
         &abstol, &M, EIGENVALS.GetDataPtr(), Z, &LDZ, ISUPPZ, &wkopt, &lwork,
         &iwkopt, &liwork, &info); 
  lwork  = (int) wkopt;
  liwork = (int) iwkopt;
  // consistent memory allocation
  Matrix<T> mw(lwork);
  Malloc(mw);
  T* work = mw.GetDataPtr();
  //
  Matrix<int> miw(liwork);
  Malloc(miw);
  int* iwork = miw.GetDataPtr(); 
  //
  // Compute the requested eigenvalues of the matrix A
  xsyevr(&JOBZ, &RANGE, &UPLO, &N, A.GetDataPtr(), &LDA, &vL, &vU, &iL, &iU, 
         &abstol, &M, EIGENVALS.GetDataPtr(), Z, &LDZ, ISUPPZ, work, &lwork,
         iwork, &liwork, &info);   
   if (info !=0 ) {
     printf ("\n*** FBLAS::XSYEVR (LAPACK_?syevr: info !=0 ***\n");
     if (info < 0) 
       printf (" ..... the %d-th argument had an illegal value\n", -info);
     else 
       printf (" ..... ?SYEVR = %d internal error. \n", info);
   }
   assert ( info==0 && "\n*** FBLAS::XSYEVR (LAPACK_?syevr): info !=0 ***\n");            
   //
   // free allocated memory
   Free (mw);    
   Free (miw);    
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
// order   : only the (Fortran style) col-major matrix order is supported
template < class T >
int FBLAS::XSYEVR(Matrix<T>& A,  Matrix<T>& EIGENVALS, Matrix<T>& EIGENVECTS, 
                  int whichEigenValue, T minEigenVal, T maxEigenVal, 
                  bool isUploA, T abstol) {
  // A is symmetric: either the upper or the lower  triangular part is given
  char UPLO = (isUploA)  ? 'U' : 'L';
  // dimensions of input matrix A (NxN squared)
  int     N = A.GetNumRows();
  // Make sure that the EIGENVALS vector has the proper size
  assert ( N==EIGENVALS.GetNumRows() && "\n*** FBLAS::XSYEVR: EIGENVALS should be a single col matrix with the same size as the input square matrix A. ***\n");                  
  // Leading dimension of matrix A (squared so both are N)
  int   LDA = N;
  // If the range of the required eignevalues are given as min/max index:
  //  - determine lower/upper indices of the required eigenvlaues
  int    iL = (whichEigenValue==2) ? static_cast<int>(minEigenVal) : 0;
  int    iU = (whichEigenValue==2) ? static_cast<int>(maxEigenVal) : 0;
  // If the range of the required eignevalues are given as min/max values:
  //  - determine lower/upper values of the required eigenvlaues
  T      vL = (whichEigenValue==1) ? minEigenVal : 0.;
  T      vU = (whichEigenValue==1) ? maxEigenVal : 0.;
  // All eigenvalues are required otherwise:
  char RANGE = (whichEigenValue==0  ? 'A' : (whichEigenValue==1 ? 'V' : 'I'));
  // Make sure that the EIGENVECTS matrix has proper dimensions
  assert ( N==EIGENVECTS.GetNumRows() && "\n*** CBLAS::XSYEVR: EIGENVECTS matrix should have same number of row as the input square matrix A. ***\n");                  
//  assert ( !(whichEigenValue==0 && N==EIGENVECTS.GetNumCols()) && "\n*** CBLAS::XSYEVR: EIGENVECTS matrix should have enough cols to store the N eigenvectors.(all eigens was required). ***\n");                  
//  assert ( !(whichEigenValue==2 && (iU-iL+1)>=EIGENVECTS.GetNumCols()) && "\n*** CBLAS::XSYEVR: EIGENVECTS matrix should have enough cols to store the iU-iL+1 eigenvectors. ***\n");                  
  // Leading dimension of matrix EIGENVECTS
  int LDZ   = EIGENVECTS.GetNumRows();
  // Both eignevalues and corresponding eigenvectors are required in this method
  char JOBZ = 'V';
  // Set absolute tolerance
  //abstol = ()
  // Number of eigenvalues found (on exit)
  int     M = 0;
  // We won't use this support infomation
  int*  ISUPPZ = (int*)malloc(sizeof(int)*2*N);
  //
  // Query and allocate the optimal workspace 
  int info   =  0;
  int lwork  = -1;
  int liwork = -1;
  int iwkopt =  0;  
  T   wkopt  =  0.;
  xsyevr(&JOBZ, &RANGE, &UPLO, &N, A.GetDataPtr(), &LDA, &vL, &vU, &iL, &iU, 
         &abstol, &M, EIGENVALS.GetDataPtr(), EIGENVECTS.GetDataPtr(), &LDZ, 
         ISUPPZ, &wkopt, &lwork, &iwkopt, &liwork, &info);   
  lwork  = (int) wkopt;
  liwork = (int) iwkopt;
  // consistent memory allocation
  Matrix<T> mw(lwork);
  Malloc(mw);
  T* work = mw.GetDataPtr();
  //
  Matrix<int> miw(liwork);
  Malloc(miw);
  int* iwork = miw.GetDataPtr(); 
  //
  // Compute the requested eigenvalues and eigenvectors of the matrix A
  xsyevr(&JOBZ, &RANGE, &UPLO, &N, A.GetDataPtr(), &LDA, &vL, &vU, &iL, &iU, 
         &abstol, &M, EIGENVALS.GetDataPtr(), EIGENVECTS.GetDataPtr(), &LDZ, 
         ISUPPZ, work, &lwork, iwork, &liwork, &info);   
   if (info !=0 ) {
     printf ("\n*** FBLAS::XSYEVR (LAPACK_?syevr: info !=0 ***\n");
     if (info < 0) 
       printf (" ..... the %d-th argument had an illegal value\n", -info);
     else 
       printf (" ..... ?SYEVR = %d internal error. \n", info);
   }
   assert ( info==0 && "\n*** FBLAS::XSYEVR (LAPACK_?syevr): info !=0 ***\n");            
   //
   // free allocated memory
   Free (mw);    
   Free (miw);    
   //
   return M;
}
