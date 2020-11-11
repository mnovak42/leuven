
#include "sys/time.h"
#include "time.h"

// for std::memcpy
#include <cstring>
#include <thread>

//
// Note, the following needs to be done before calling the training:
//   - parameters of the kernel function must be set (SetKernelParameters)
//   - the training data matrix pointer must be set  (SetInputTrainingDataMatrix)
//   - the incomplete Cholesky matrix must be set (SetIncCholeskyMatrix)
//   - number of clusters to be found must be set  (SetNumberOfClustersToFind)
//   - a cluster encoding, assigment and quality measure must be set (SetEncodingAndQualityMeasureType)
template <class TKernel, typename T,  typename TInputD>
void
KscWkpcaIChol<TKernel, T, TInputD>::Train(size_t numBLASThreads, bool isQMOnTraining, size_t qmFlag, int verbose) {
  // create the BLAS for memory managment and to call CPU BLAS, LAKACK methods
  BLAS  theBlas;
  theBlas.SetNumThreads(numBLASThreads, verbose);
  const int kThreads = numBLASThreads;
  // set dimensions: number, dimension of the training data and the rank of the 
  //                 incomplete Cholesky approximation i.e. number of cols in G
  const size_t theNumTrData  = fInputTrainingDataM->GetNumRows(); // (N_tr); row major!
  const size_t theDimTrData  = fInputTrainingDataM->GetNumCols(); // each row is one data
  const size_t theRankOfAprx = fIncCholeskyM->GetNumCols(); // (R); rows should be = theNumTrData
  //
  // ------------------------------------------------------------------------ // 
  // 1. Compute the K-1 leading approximated eigenvectors of the D^{-1}M_D\Omega 
  //    matrix and (only in case of BLF, AMS) the corresponding K-1 approximated 
  //    bias terms (the Cholesky matrix will be destroyed):
  //
  //  - the approximated bias terms: computed only in case of BFL and AMS:
  // clean the memory if its needed and allocate memory if its needed
  if (fTheAprxBiasTermsM) {
    theBlas.Free(*fTheAprxBiasTermsM);
    delete fTheAprxBiasTermsM;
    fTheAprxBiasTermsM = nullptr;
  }   
  if (fEncodingAndQM->GetQualityMeasureType()!=KscQMType::kBAS) {
    fTheAprxBiasTermsM = new Matrix<T>(fNumberOfClusters-1, 1);
    theBlas.Malloc(*fTheAprxBiasTermsM);
  } 
  // - the approximated eigenvectors
  Matrix<T> theAprxEigenvectM(theNumTrData, fNumberOfClusters-1);
  theBlas.Calloc(theAprxEigenvectM); 
  // Note: the approximated bias terms will be compute if the fTheAprxBiasTermsM!=nulltr
  if (verbose>1) {
    std::cout<< "          ====> Starts computing eigenvectors... " << std::endl;    
  }
  ComputeApproximatedEigenvectors(theAprxEigenvectM, numBLASThreads, verbose);
  //
  // ------------------------------------------------------------------------ //
  // 2. Create the reduced set and compute the reduced set coefficient matrix
  //    2.1. Create the reduced set data matrix member
  // NOTE: the first R (theRankOfAprx) points in the fInputTrainingDataM are 
  //       assumed to be the reduced set points i.e. the permutations, corresponding 
  //       to the incomplete Cholesky, should have already been done on this matrix.
  //    2.2. Calculate the reduced set coefficients:
  //      2.2 a. Form the within reduced set and reduced set training set kernel matrix
  //      2.2 b. Solve the system of linear equations to get the reduce coefficients
  //             (store the transpose i.e. the K-1 dimensional coefs as cols)
  // 2.1 & 2.2 a.
  if (fTheReducedSetDataM) {
    theBlas.Free(*fTheReducedSetDataM);
    delete fTheReducedSetDataM;
  }
  fTheReducedSetDataM = new Matrix<TInputD, false>(theRankOfAprx, theDimTrData);
  theBlas.Malloc(*fTheReducedSetDataM);
  //
  if (verbose>1) {
    std::cout<< "          ====> Starts forming the Reduced-Reduced and Reduced-Test kernelmatrix... " << std::endl;      
  }  
  Matrix<T> theReducedSetKernelM(theRankOfAprx, theRankOfAprx);
  Matrix<T> theReducedTrainingSetKernelM(theRankOfAprx, theNumTrData);
  theBlas.Malloc(theReducedSetKernelM);
  theBlas.Malloc(theReducedTrainingSetKernelM);
  ComputeKernelMatrix(theReducedTrainingSetKernelM, *fInputTrainingDataM, *fInputTrainingDataM, kThreads);
  // copy the reduced set kernel matrix part (first R cols that has R rows each)
  std::memcpy(theReducedSetKernelM.GetDataPtr(), theReducedTrainingSetKernelM.GetDataPtr(), sizeof(T)*theRankOfAprx*theRankOfAprx);
  // copy reduced set data
  std::memcpy(fTheReducedSetDataM->GetDataPtr(), fInputTrainingDataM->GetDataPtr(), sizeof(TInputD)*fInputTrainingDataM->GetNumCols()*theRankOfAprx);  
  // 2.2 b. create reduced set coef. matrix and allocate memory
  if (fTheReducedSetCoefM) {
    theBlas.Free(*fTheReducedSetCoefM);
    delete fTheReducedSetCoefM;
  }
  fTheReducedSetCoefM = new Matrix<T>(fNumberOfClusters-1, theRankOfAprx);
  theBlas.Calloc(*fTheReducedSetCoefM);
  // its transpose is computed then it will be copied
  Matrix<T> theReducedSetCoefM (theRankOfAprx, fNumberOfClusters-1);
  theBlas.Calloc(theReducedSetCoefM);  
  if (verbose>1) {
    std::cout<< "          ====> Starts computing the reduced set coefs... " << std::endl;        
  }
  // compute B = \Omega_{RxN_tr} \Beta_{N_trxK-1} + B
  theBlas.XGEMM(theReducedTrainingSetKernelM, theAprxEigenvectM, theReducedSetCoefM);
  // free theAprxEigenvectM
  theBlas.Free(theAprxEigenvectM);  
  // solve \Omega_{RxR} X_{RxK-1} = B_{RxK-1}: X will be in B at the end
  theBlas.XSYSV(theReducedSetKernelM, theReducedSetCoefM);
  // take the trasnpose -> memory continous K-1 dimensional data
  for (size_t ir=0; ir<theRankOfAprx; ++ir) {
    for (size_t ic=0; ic<fNumberOfClusters-1; ++ic) {
      fTheReducedSetCoefM->SetElem(ic, ir, theReducedSetCoefM.GetElem(ir, ic));
    }
  }
  // clean the memory allocated for theReducedSetKernelM and to the intermediate 
  // theReducedSetCoefM
  theBlas.Free(theReducedSetKernelM);  
  theBlas.Free(theReducedSetCoefM);  
  // theReducedTrainingSetKernelM will be used later...
  //
  // ------------------------------------------------------------------------ //
  // 3. Generate cluster membership encoding
  //  3.1. In case of BFL and AMS: compute the approximated score matrix as 
  //       theReducedSetCoefM_{#number_of_clusters-1,R}\Omega{RN_tr} plus the 
  //       (K-1) approximated bias terms to each columns. The way this is 
  //       computed is first wihout the bias term:
  //        [theReducedSetCoefM_{R,#number_of_clusters-1}] \Omega{RN_tr} then 
  //        each col. of the result is a K-1 dimensional score valiable minus 
  //        the K-1 diemnsional bias term that will be added.
  //  3.1. In case of BAS, the approximated score matrix is not needed because 
  //       the code book is formed based on the reduced set coeffitients.
  //  3.2. The code book i.e. the cluster membership encoding is generated.
  if (verbose>1) {
    std::cout<< "          ====> Starts generating encoding... " << std::endl;          
  }
  Matrix<T>* theAprxScoreVariableM = nullptr;
  if (fEncodingAndQM->GetQualityMeasureType()!=KscQMType::kBAS) {
    // BLF OR AMS case: 
    // 3.1. compute the approximated score variables minus the bias term 
    theAprxScoreVariableM = new Matrix<T>(fNumberOfClusters-1, theNumTrData);
    theBlas.Calloc(*theAprxScoreVariableM);
    theBlas.XGEMM(*fTheReducedSetCoefM, theReducedTrainingSetKernelM, *theAprxScoreVariableM);
    // complete 3.1. i.e. add the approximated bias terms 
    for (size_t idat=0; idat<theNumTrData; ++idat) {
      for (size_t is=0; is<fNumberOfClusters-1; ++is) {
        theAprxScoreVariableM->SetElem(is, idat, theAprxScoreVariableM->GetElem(is, idat)+fTheAprxBiasTermsM->GetElem(is, 0));
      }
    }
    // 3.2.
    fEncodingAndQM->GenerateCodeBook(*theAprxScoreVariableM, fNumberOfClusters);
  } else {
    // BAS
    // 3.2. 
    fEncodingAndQM->GenerateCodeBook(*fTheReducedSetCoefM, fNumberOfClusters);
  }  
  // 
  // ------------------------------------------------------------------------ //
  // 4. The training is done at this point. The last part, clustering the training
  //    data and computing the corresponding quality measure, is optional.
  //  4.1. Compute the approximated score variable matrix:
  //        In case of BLF and AMS: it's already done because it's necessary to 
  //                                for the encoding.
  //        in case of BAS: it's done here but the score variable minus bias term
  //                        is computed 
  //  4.2. Compute KSC model quality measure on the this training set data.

  //
  if (isQMOnTraining) {
    // allocate the matrix to store the clustering results (number of cols depends 
    // on the input arg ...)    
    size_t ncl = qmFlag<2 ? qmFlag+1 : fNumberOfClusters+1;
    if (fTheClusterMembershipM) {
      theBlas.Free(*fTheClusterMembershipM);
      delete fTheClusterMembershipM;
    }
    fTheClusterMembershipM = new Matrix<T, false>(theNumTrData, ncl);
    theBlas.Malloc(*fTheClusterMembershipM);
    // the quality measure value
    T theQMValue = 0.;
    if (fEncodingAndQM->GetQualityMeasureType()==KscQMType::kBLF) {
      // == BLF ===============================================================
      fEncodingAndQM->ClusterDataSet(*theAprxScoreVariableM, *fTheClusterMembershipM, qmFlag);
      // 3.3. a. BLF: in case of BLF and K=2: compute the second variable vector 
      //      as the col sums of the theReducedTrainingSetKernelM RxN_tr that 
      //      gives the N_tr second variable by adding the single bias term.
      Matrix<T>* theSecondVarForBLFM = nullptr;
      if (fNumberOfClusters==2) {
        theSecondVarForBLFM = new Matrix<T>(theNumTrData, 1);
        theBlas.Malloc(*theSecondVarForBLFM);
        for (size_t id=0; id<theNumTrData; ++id) {
          T sum =0.;
          for (size_t ir=0; ir<theRankOfAprx; ++ir) {
            sum += theReducedTrainingSetKernelM.GetElem(ir, id);
          }
          theSecondVarForBLFM->SetElem(id, 0, sum + fTheAprxBiasTermsM->GetElem(0,0));
        }
      }
      theQMValue = fEncodingAndQM->ComputeQualityMeasure(*fTheClusterMembershipM, theAprxScoreVariableM, theSecondVarForBLFM);
      if (theSecondVarForBLFM) {
        theBlas.Free(*theSecondVarForBLFM);
        delete theSecondVarForBLFM;
      }
    } else if (fEncodingAndQM->GetQualityMeasureType()==KscQMType::kAMS) {
      // == AMS ===============================================================
      fEncodingAndQM->ClusterDataSet(*theAprxScoreVariableM, *fTheClusterMembershipM, qmFlag);
      theQMValue = fEncodingAndQM->ComputeQualityMeasure(*fTheClusterMembershipM);
    } else {
      // == BAS ===============================================================
      // 3.1. compute the approximated score variables minus the bias term 
      theAprxScoreVariableM = new Matrix<T>(fNumberOfClusters-1, theNumTrData);
      theBlas.Calloc(*theAprxScoreVariableM);
      theBlas.XGEMM(*fTheReducedSetCoefM, theReducedTrainingSetKernelM, *theAprxScoreVariableM);
      //
      fEncodingAndQM->ClusterDataSet(*theAprxScoreVariableM, *fTheClusterMembershipM, qmFlag);
      theQMValue = fEncodingAndQM->ComputeQualityMeasure(*fTheClusterMembershipM);
      // clean allocated memory
    }
    _unused(theQMValue);  
    // ====== Print out result ===== 
//    std::cerr<< "  == The KSC model evaluation criterion: = " << fEncodingAndQM->GetName() << std::endl;
//    std::cerr<< "  == QM-value = " << fEncodingAndQM->GetTheQualityMeasureValue() << std::endl;
//    std::cerr<< "  == QM-etaBalance = " << fEncodingAndQM->GetCoefEtaBalance() << std::endl;
    
//    fTheClusterMembershipM->WriteToFile("CRes.dat");
//    fInputTrainingDataM->WriteToFile("Data.dat");       
    //
    // clean remaining allocated memory (keep only theAprxScoreVariableM if not 
    // for tuning i.e. isQMOnTraining = true)
    if (theAprxScoreVariableM) {
      theBlas.Free(*theAprxScoreVariableM);
      delete theAprxScoreVariableM;
    }
  } else {
    if (fTheAprxScoreVariableM) {
      theBlas.Free(*fTheAprxScoreVariableM);
      delete fTheAprxScoreVariableM;
    }
    fTheAprxScoreVariableM = theAprxScoreVariableM;
  }
  theBlas.Free(theReducedTrainingSetKernelM); 
  //
  //
}


template <class TKernel, typename T,  typename TInputD>
void
KscWkpcaIChol<TKernel, T, TInputD>::ComputeApproximatedEigenvectors(Matrix<T>& theAprxEigenvectM, int numBLASThreads, int verbose) {
  // create the BLAS for memory managment and to call CPU BLAS, LAKACK methods
  BLAS  theBlas;
  theBlas.SetNumThreads(numBLASThreads, verbose);

  // set dimensions: number, dimension of the training data and the rank of the 
  //                 incomplete Cholesky approximation i.e. number of cols in G
  const size_t theNumTrData  = fInputTrainingDataM->GetNumRows(); // (N_tr); row major!
//  const size_t theDimTrData  = fInputTrainingDataM->GetNumCols(); // each row is one data
  const size_t theRankOfAprx = fIncCholeskyM->GetNumCols(); // (R); rows should be = theNumTrData
  
  //
  // 1. Compute the approximated diagonal degree matrix The ICD matrix G (NxR will be destroyed)!
  // D_ii = sum_{j=1}^{N_tr} \Omega_{ij} => \tilde{D}_ii = sum_{j=1}^{N_tr} \tilde{\Omega}_{ij}  = 
  // sum_{j=1}^{N_tr} {GG^T}_{ij}  = G [G^T 1_{N}] i.e. first the sum of each of the R cols
  // then each of the N rows of G is multiplied by this vector to give the (N_tr) diagD vector.
  if (verbose>2) {
    std::cout<< "             ...... degree business starts... " << std::endl;      
  }
  std::vector<T> theAprxDegreeVect(theNumTrData, 0.);
  std::vector<T> dumv(theRankOfAprx, 0.);
  // make sure we move along memory -> vectorized inner loop
  for (size_t ic=0; ic<theRankOfAprx; ++ic) {
    T dum0 = 0.;
    for (size_t ir=ic; ir<theNumTrData; ++ir) { // along memory <- col major : vectorized
      dum0 += fIncCholeskyM->GetElem(ir, ic);
    }
    dumv[ic] = dum0;
  }
  // make sure we move along memory here as well -> vectorized inner loop
  for (size_t ic=0; ic<theRankOfAprx; ++ic) {
    const T dum0 = dumv[ic];
    for (size_t ir=ic; ir<theNumTrData; ++ir) { // along memory <- col major : vectorized
      theAprxDegreeVect[ir] += dum0*fIncCholeskyM->GetElem(ir, ic);
    }
  }  
  //
  // 2. Generate the \tilde{D}^{-1/2}M_{\tilde{D}}G matrix: in place of G -> so 
  //    G will be destroyed here:
  //   a. the 1/d_i weighet mean of each of the cols of G needs to be ecomputed
  //   b. then 1/sqrt(d_i) [G_ij - that] 
  if (verbose>2) {
    std::cout<< "             ...... generating D^-1/2 M_D G matrix starts... " << std::endl;        
  }
  std::vector<T> theInvAprxDegreeVect(theNumTrData);
  std::vector<T> theSqrtInvAprxDegreeVect(theNumTrData);
  T theSumInvDegree = 0.;
  for (size_t i=0; i<theNumTrData; ++i) {
    const T dum = std::max(theAprxDegreeVect[i], 1.0E-16);
    theAprxDegreeVect[i]        = dum;
    const T idum                = 1./dum;
    theInvAprxDegreeVect[i]     = idum;
    theSqrtInvAprxDegreeVect[i] = std::sqrt(idum);
    theSumInvDegree               += idum;
  }
  theSumInvDegree = 1./theSumInvDegree; // 1./[sum_i^N_tr 1/d_i]
  for (size_t ic=0; ic<theRankOfAprx; ++ic) {
    T weightedColMeanOfG = 0.;
    for (size_t ir=ic; ir<theNumTrData; ++ir) { // G is lower triangular
      weightedColMeanOfG += fIncCholeskyM->GetElem(ir, ic)*theInvAprxDegreeVect[ir];
    }
    weightedColMeanOfG *= theSumInvDegree;
    for (size_t ir=0; ir<theNumTrData; ++ir) {
      fIncCholeskyM->SetElem(ir, ic, (fIncCholeskyM->GetElem(ir, ic) - weightedColMeanOfG)*theSqrtInvAprxDegreeVect[ir]);
    }
  }
  //
  // 3. QR factorization of the \tilde{D}^{-1/2}M_{\tilde{D}}G matrix prepared at 2.:
  //   a. the input matrix will be overwritten by the upper triangular matrix R;
  //      matrix Q is not formed but information stored in the input matrix as well 
  //      as in the additional theTauVect col vector (these will be used later)
  //   b. retrive the (RxR) upper triangular matrix of \tilde{D}^{-1/2}M_{\tilde{D}}G = QR 
  // a.: create the tau vector and allocate the memory
  Matrix<T> theSigmaVect(theRankOfAprx, 1);
  theBlas.Malloc(theSigmaVect);

//
// === ON DEVICE computation:
//   Perform the QR decomposition of the \tilde{D}^{-1/2}M_{\tilde{D}}G matrix,
//   the SVD decompositon of the resulted R matrix, computation of the eigenvectors 
//   of the symmetric problem (by left multiplying the left-singular vectors with Q).
//   (Only is building with the USE_CUBLAS CMake option, i.e. with CUDA support, 
//    and requested by setting the fUseGPU member.)
  
#if USE_CUBLAS
  if ( fUseGPU ) {
      BLAS_gpu  theBlas_gpu;
      if (verbose>2) {
        std::cout<< "             ...... QR starts... on GPU" << std::endl;
      }
      Matrix<T> theDMG_d(theNumTrData, theRankOfAprx);
      Matrix<T> theTauVect_d(theRankOfAprx, 1);
      theBlas_gpu.Malloc(theDMG_d);      // the \tilde{D}^{-1/2}M_{\tilde{D}}G matrix to be QR-ed
      theBlas_gpu.Malloc(theTauVect_d);  // tau-vector to store coefs of the refs.
      // copy the \tilde{D}^{-1/2}M_{\tilde{D}}G matrix
      theBlas_gpu.CopyToGPU(*fIncCholeskyM, theDMG_d);
      // call QR factorization
      theBlas_gpu.XGEQRF(theDMG_d, theTauVect_d);
      // b.: create matrix R, allocate memory and obtain as the upper-triangular of the QR decomposed theDMG_d
      Matrix<T> theRM_d(theRankOfAprx, theRankOfAprx);
      theBlas_gpu.Calloc(theRM_d);
      theBlas_gpu.GetUpperTriangular(theDMG_d, theRM_d);// theNumTrData, theRankOfAprx
      //
      // 4. Perform SVD on the R matrix: on completion, R matrix will contain the 
      //    left singular vectors (all i.e. as many as cols in R) and theSigmaVect 
      //    will contain the corresponding singular values.
      // create the sigma vector and allocate memory
      if (verbose>2) {
        std::cout<< "             ...... SVD starts... on GPU" << std::endl;        
      }
      Matrix<T> theSigmaVect_d(theRankOfAprx, 1);
      theBlas_gpu.Malloc(theSigmaVect_d);
      // call the SVD
      theBlas_gpu.XGESVD(theRM_d, theSigmaVect_d);
      //
      // 5. Take the fNumberOfClusters-1 leading left singular vectors into the 
      //    theAprxEigenvectM and multiply this matrix by Q (from left) to get the 
      //    fNumberOfClusters-1 leading eigenvectors of the symmetrix problem. 
      //    (This is why we make a larger, N_tr x fNumberOfClusters-1 matrix
      //    instead of the required R x fNumberOfClusters-1).
      // NOTE: with intel MKL, the result of ?ormqr depends on the number of columns 
      //       in the matrix: when computed QA and QB such that the first n=cols of 
      //       A and B are identical, the first n-cols of the resulted QA and QB are
      //       not identical (small numerical differences). To avoid this, the 
      //       matrix Q (theNumTrData, theRankOfAprx) is formed explicitely and QU 
      //       is computed.
      // form Q (in the Ichol matrix)
      if (verbose>2) {
        std::cout<< "             ...... computing QU starts ... GPU" << std::endl;            
      }
      theBlas_gpu.XORGQR(theDMG_d, theTauVect_d); 
      // take the K-1 left singular vectors into U
      Matrix<T>  theUM_d(theRankOfAprx, fNumberOfClusters-1);
      theBlas_gpu.Malloc(theUM_d);
      theBlas_gpu.CopyOnGPU(theRM_d.GetDataPtr(), theUM_d.GetDataPtr(), sizeof(T)*theRankOfAprx*(fNumberOfClusters-1));
      // compute QU into theAprxEigenvectM (that's the matrix with the \beta eigenvects) 
      Matrix<T>  theAprxEigenvectM_d(theAprxEigenvectM.GetNumRows(), theAprxEigenvectM.GetNumCols());
      theBlas_gpu.Calloc(theAprxEigenvectM_d);
      theBlas_gpu.XGEMM(theDMG_d, theUM_d, theAprxEigenvectM_d);
      // copy results from device to host
      theBlas_gpu.CopyFromGPU(theAprxEigenvectM_d, theAprxEigenvectM);
      theBlas_gpu.CopyFromGPU(theSigmaVect_d, theSigmaVect);
      // free all device side memory
      theBlas_gpu.Free(theDMG_d);
      theBlas_gpu.Free(theTauVect_d);
      theBlas_gpu.Free(theRM_d);
      theBlas_gpu.Free(theSigmaVect_d);
      theBlas_gpu.Free(theUM_d);
      theBlas_gpu.Free(theAprxEigenvectM_d);
      // also free the input Cholesky matrix (has been destroyed) and reset pointer
      theBlas.Free(*fIncCholeskyM);
      delete fIncCholeskyM;
      fIncCholeskyM = nullptr;
  } else {    
#endif  // USE_CUBLAS
  // not USE_CUBLAS: keep computaing the QR, SVD and QU on the CPU 
  if (verbose>2) {
    std::cout<< "             ...... QR starts... " << std::endl;
  }
  Matrix<T> theTauVect(theRankOfAprx, 1);    
  theBlas.Malloc(theTauVect);
  // call QR factorization
  theBlas.XGEQRF(*fIncCholeskyM, theTauVect);
  // b.: create matrix R and allocate memory
  Matrix<T> theRM(theRankOfAprx, theRankOfAprx);
  theBlas.Calloc(theRM);
  for (size_t ic=0; ic<theRankOfAprx; ++ic) {
    for (size_t ir=0; ir<ic+1; ++ir) {
      theRM.SetElem(ir, ic, fIncCholeskyM->GetElem(ir, ic));
    }
  }
  //
  // 4. Perform SVD on the R matrix: on completion, R matrix will contain the 
  //    left singular vectors (all i.e. as many as cols in R) and theSigmaVect 
  //    will contain the corresponding singular values.
  // create the sigma vector and allocate memory
  if (verbose>2) {
    std::cout<< "             ...... SVD starts... " << std::endl;        
  }
//  Matrix<T> theSigmaVect(theRankOfAprx, 1);
//  theBlas.Malloc(theSigmaVect);
  // call the SVD
  theBlas.XGESVD(theRM, theSigmaVect);
  //
  // 5. Take the fNumberOfClusters-1 leading left singular vectors into the 
  //    theAprxEigenvectM and multiply this matrix by Q (from left) to get the 
  //    fNumberOfClusters-1 leading eigenvectors of the symmetrix problem. 
  //    (This is why we make a larger, N_tr x fNumberOfClusters-1 matrix
  //    instead of the required R x fNumberOfClusters-1).
  // NOTE: with intel MKL, the result of ?ormqr depends on the number of columns 
  //       in the matrix: when computed QA and QB such that the first n=cols of 
  //       A and B are identical, the first n-cols of the resulted QA and QB are
  //       not identical (small numerical differences). To avoid this, the 
  //       matrix Q (theNumTrData, theRankOfAprx) is formed explicitely and QU 
  //       is computed.
  // form Q (in the Ichol matrix)
  if (verbose>2) {
    std::cout<< "             ...... computing QU starts ... " << std::endl;            
  }
  theBlas.XORGQR(*fIncCholeskyM, theTauVect); 
  // take the K-1 left singular vectors into U
  Matrix<T>  theUM(theRankOfAprx, fNumberOfClusters-1);
  theBlas.Malloc(theUM);
  std::memcpy(theUM.GetDataPtr(), theRM.GetDataPtr(), sizeof(T)*theRankOfAprx*(fNumberOfClusters-1));
  // compute QU into theAprxEigenvectM (that's the matrix with the \beta eigenvects) 
  theBlas.XGEMM(*fIncCholeskyM, theUM, theAprxEigenvectM);
  //
  // theTauVect, theRM and fIncCholeskyM can be freed here  
  theBlas.Free(theTauVect);
  theBlas.Free(theRM);
  theBlas.Free(theUM);
  theBlas.Free(*fIncCholeskyM);
  delete fIncCholeskyM;
  fIncCholeskyM = nullptr;
#if USE_CUBLAS
  }  
#endif  // USE_CUBLAS
  //
  // 6. Compute the approximated eigenvectors of the original, non-symetric problem 
  //    by left multiplying theAprxEigenvectM (\beta-s) with \tilde{D}^-1/2 
  if (verbose>2) {
    std::cout<< "             ...... computing the original eigenvectors starts ... " << std::endl;            
  }
  for (size_t ic=0; ic<fNumberOfClusters-1; ++ic) {
    for (size_t ir=0; ir<theNumTrData; ++ir) {
      theAprxEigenvectM.SetElem(ir, ic, theAprxEigenvectM.GetElem(ir, ic)*theSqrtInvAprxDegreeVect[ir]);
    }
  }
  //
  // The fNumberOfClusters-1 apporximated eigenvectors of the D^{-1}M_D Omega
  // matrix are now in the theAprxEigenvectM matrix !!!
  //
  // 7. Calculation of the approximated bias terms according to 
  //   \tilde{b}^{(l)} = \farc{1}{n_{tr}}[\lambda^{(l)}-1]\mathbb{1}_{N_{tr}}\tilde{D}\tilde{\mathbf{\beta}}^{(l)} l=1,..,#clusters-1
  // NOTE: it's needed only in case of BLF and AMS quality measures
//  Matrix<T> theAprxBiasTermsM(fNumberOfClusters-1, 1);
//  theBlas.Malloc(theAprxBiasTermsM);
  if (verbose>2) {
    std::cout<< "             ...... computing bias terms starts ... " << std::endl;          
  }
  if (fTheAprxBiasTermsM) {
    const T invNumTrData = 1./theNumTrData;
    for (size_t ic=0; ic<fNumberOfClusters-1; ++ic) {
      T dum0 = 0;
      for (size_t ir=0; ir<theNumTrData; ++ir) {
        dum0 += theAprxEigenvectM.GetElem(ir, ic)*theAprxDegreeVect[ir];
      }
      const T dum1 = theSigmaVect.GetElem(ic, 0);
      fTheAprxBiasTermsM->SetElem ( ic, 0, dum0*(dum1*dum1-1.)*invNumTrData );
    }
  }
  // clean the memory allocated for theSigmaVect 
  theBlas.Free(theSigmaVect);  
}



template < class TKernel, typename T,  typename TInputD>
template < typename TKernelParameterType >
void
KscWkpcaIChol<TKernel, T, TInputD>::Tune(std::vector<TKernelParameterType>& theKernelParametersVect, size_t minNumberOfClusters, size_t maxNumberOfClusters,  Matrix<TInputD, false>& theValidInputDataM, size_t numBLASThreads, int verbose) {
  // create the BLAS for memory managment and to call CPU BLAS, LAPACK methods
  BLAS  theBlas;
  theBlas.SetNumThreads(numBLASThreads, verbose);
  // set dimensions: number, dimension of the training data and the rank of the 
  //                 incomplete Cholesky approximation i.e. number of cols in G
//  const size_t theNumTrData   = fInputTrainingDataM->GetNumRows(); // (N_tr); row major!
//  const size_t theDimTrData   = fInputTrainingDataM->GetNumCols(); // each row is one data
  const size_t theRankOfAprx  = fIncCholeskyM->GetNumCols(); // (R); rows should be = theNumTrData
  const size_t theNumValidData = theValidInputDataM.GetNumRows();
  //
  size_t optimalKernelParameterIndx = 0;
  size_t optimalNumberOfClusters    = 0;
  T      optimalQMValue             = 0.;
  // the kernel parameter axis of the 2D grid is given as input but the number 
  // of clusters one needs to be generated (note: its relatively cheap to generate
  // model for a given cluster number after the max-clusyert number is done)
  const size_t theSizeNumberOfClustersVect = maxNumberOfClusters - minNumberOfClusters + 1;
  std::vector<size_t> theNumberOfClustersVect(theSizeNumberOfClustersVect);
  for (size_t i=0; i<theSizeNumberOfClustersVect; ++i) {
    theNumberOfClustersVect[i] = minNumberOfClusters+i;
  }
  //
  // Allocate the tuning result matrix to store the quality measure values over 
  // the 2D 'kernel parameter(index)' - 'number of clusters' grid.
  if (fTheResTuningM) {
    theBlas.Free(*fTheResTuningM);
    delete fTheResTuningM;
  }
  fTheResTuningM = new Matrix<T, false>(theKernelParametersVect.size(), theSizeNumberOfClustersVect); 
  theBlas.Calloc(*fTheResTuningM);
  //
  // Make a copy of the Incomplete Cholesky Matrix: it will be destroyed and
  // the corresponding memory will be freed when calculating the eigenvectors. 
  // So 1. save the current pointer to the orignal Ichol matrix, make a copy of 
  // the matrix and set the fIncCholeskyM pointer to that (the copy will be 
  // destroyed) and reset the fIncCholeskyM pointer to the original, untouched 
  // matrix at the end.
  Matrix<T>* fIncCholeskyM_ptr = fIncCholeskyM;
  //
  // == the approximated score variable matrix with max required capacity
  Matrix<T> theAprxTestScoreVariableM(maxNumberOfClusters-1, theNumValidData);
  theBlas.Malloc(theAprxTestScoreVariableM); // will be set to zero 
  // == the reduced-valid set kernel matrix 
  Matrix<T> theReducedValidSetKernelM(theRankOfAprx, theNumValidData);
  theBlas.Malloc(theReducedValidSetKernelM);
  // == clustering results 
  Matrix<T, false> theClusterMembershipM(theNumValidData, 2);
  theBlas.Malloc(theClusterMembershipM);
  //

const int kThreads = numBLASThreads;  
KscEncodingAndQM<T>* theObjVect[kThreads];
Matrix<T, false>*    theClusterMembershipMVect[kThreads]; 
std::vector<int> intIndx(kThreads);
for (int i=0; i<kThreads; ++i)  {
  if (fEncodingAndQM->GetQualityMeasureType()        == KscQMType::kAMS) {
     theObjVect[i] = new KscEncodingAndQM_AMS<T>();
  } else if (fEncodingAndQM->GetQualityMeasureType() == KscQMType::kBAS) {
     theObjVect[i] = new KscEncodingAndQM_BAS<T>();  
  } else {
     theObjVect[i] = new KscEncodingAndQM_BLF<T>();  
  }
  theObjVect[i]->SetCoefEtaBalance(fEncodingAndQM->GetCoefEtaBalance());
  theObjVect[i]->SetOutlierThreshold(fEncodingAndQM->GetOutlierThreshold());
  theClusterMembershipMVect[i] = new Matrix<T, false>(theNumValidData, 2);
  theBlas.Malloc(*(theClusterMembershipMVect[i]));
}
std::vector<std::thread> theThreads(kThreads);  
 
  // loop over the given kernel parametres  
  for (size_t ikp=0; ikp<theKernelParametersVect.size(); ++ikp) {
    if (verbose > 1 ) {
      std::cout<< "  === KscWkpcaIChol::Tune : tuning for the " << ikp << "-th kernel parameters out of the " << theKernelParametersVect.size()-1<< std::endl;
    }
    // 1. Make a copy of the Incomplete Cholesky Matrix: it will be destroyed and
    //    the corresponding memory will be freed when calculating the eigenvectors. 
    if (verbose > 2 ) {
      std::cout<< "      --- Starts copy ichol matrix... " << std::endl;  
    }
    fIncCholeskyM = new Matrix<T>(fIncCholeskyM_ptr->GetNumRows(), fIncCholeskyM_ptr->GetNumCols());
    theBlas.Malloc(*fIncCholeskyM);
    memcpy(fIncCholeskyM->GetDataPtr(), fIncCholeskyM_ptr->GetDataPtr(), sizeof(T)*fIncCholeskyM_ptr->GetSize());
    // 2. Use the current kernel parameters, train a model and evaluate its 
    //    performance on the test-data set at each possible number-of-clusters 
    //    on the [minNumberOfClusters, maxNumberOfClusters].
    //  2.1. - Set the kernel parameters to the current value and the number of 
    //         required clusters to maxNumberOfClusters
    fKernel->SetParameters(theKernelParametersVect[ikp]);
    fNumberOfClusters = maxNumberOfClusters;
    //  2.2. - Train a model with using this maxNumberOfClusters using the training 
    //         data: this will set the model (reduced set data and coefficients 
    //         matrix, generates the membership encoding and the approximated 
    //         bias terms (if needed) ) for maxNumberOfClusters.
    // false => do not cluster and compute QM for the training data
    // NOTE: Training on the Training data set
    if (verbose > 2 ) {
      std::cout<< "      --- Starts training... " << std::endl;    
    }
    Train(numBLASThreads, false);
    // === compute the reduced-test set kernel matrix with the current kernle parameters
    if (verbose > 2 ) {
      std::cout<< "      --- Starts forming the reduced-valid set kenel matrix... " << std::endl;  
    }
    ComputeKernelMatrix(theReducedValidSetKernelM, *fTheReducedSetDataM, theValidInputDataM, kThreads);
/*
    for (size_t ic=0; ic<theNumValidData; ++ic) {
      const TInputD* inpData1 = theValidInputDataM.GetPtrToBlock(ic);
      for (size_t ir=0; ir<theRankOfAprx; ++ir) {
        const TInputD* inpData2 = fTheReducedSetDataM->GetPtrToBlock(ir);
        const T valKernel = fKernel->Evaluate(inpData1, inpData2, theDimTrData);
        theReducedValidSetKernelM.SetElem(ir, ic, valKernel);
      }
    } 
*/
       
    //  2.3. - compute the approximated score variables on the TEST set for 
    //         the maxNumberOfClusters-1 case: theAprxScoreVariableM with (maxK-1)x(N_test)
    if (verbose > 2 ) {
      std::cout<< "      --- Score matrix computation... " << std::endl;        
    }
    std::memset(theAprxTestScoreVariableM.GetDataPtr(), 0, sizeof(T)*theAprxTestScoreVariableM.GetSize());
    theBlas.XGEMM(*fTheReducedSetCoefM, theReducedValidSetKernelM, theAprxTestScoreVariableM);
    // add the approximated bias terms (only if BLF or AMS) 
    if (fEncodingAndQM->GetQualityMeasureType()!=KscQMType::kBAS) {
      for (size_t idat=0; idat<theNumValidData; ++idat) {
        for (size_t is=0; is<fNumberOfClusters-1; ++is) {
          theAprxTestScoreVariableM.SetElem(is, idat, theAprxTestScoreVariableM.GetElem(is, idat)+fTheAprxBiasTermsM->GetElem(is, 0));
        }
      }
    }
    // 3. Loop over all possible number of clusters and for eack K [minK, maxK]:
    //  3.1. - form the code book for the current cluster number, perform the 
    //         clustering of the test data set and compute the quality measure
    if (verbose > 2 ) {
      std::cout<< "      --- Starts clustering for each possible cluster number K in [K_min, K_max] ... " << std::endl;    
    }
int ic = theSizeNumberOfClustersVect-1; 
int nn = (theSizeNumberOfClustersVect/(double)kThreads);
for (int ii=0; ii<nn; ++ii)  {  
  Matrix<T>* encodeMatrix = fEncodingAndQM->GetQualityMeasureType() == KscQMType::kBAS ?  fTheReducedSetCoefM : fTheAprxScoreVariableM;
  
  // NOTE: in case of BLF the K=2 case will be skipped and left to the tail processing 
  for (int t=0; t<kThreads; ++t) {
    if (fEncodingAndQM->GetQualityMeasureType() != KscQMType::kBLF || theNumberOfClustersVect[ic] != 2) {
      theThreads[t] = std::thread(&KscEncodingAndQM<T>::DoAll, theObjVect[t], std::move(encodeMatrix), theNumberOfClustersVect[ic], std::move(&theAprxTestScoreVariableM), std::move(theClusterMembershipMVect[t]));      
      //theObjVect[t]->DoAll(encodeMatrix, fNumberOfClusters, &theAprxTestScoreVariableM, theClusterMembershipMVect[ic%kThreads]);
    } 
    intIndx[t] = ic;
    --ic;
  }
  
  for (int t=0; t<kThreads; ++t) {
    if (fEncodingAndQM->GetQualityMeasureType() == KscQMType::kBLF && theNumberOfClustersVect[intIndx[t]] == 2) {
      ++ic;
      continue;
    }
    theThreads[t].join();
  }  
  for (int t=0; t<kThreads; ++t) {
    if (fEncodingAndQM->GetQualityMeasureType() == KscQMType::kBLF && theNumberOfClustersVect[intIndx[t]] == 2) {
      continue;
    }
    T theQMValue = theObjVect[t]->GetTheQualityMeasureValue();
    size_t iic = intIndx[t];
    if (theQMValue>optimalQMValue) {
      optimalQMValue             = theQMValue;
      optimalKernelParameterIndx = ikp;
      optimalNumberOfClusters    = theNumberOfClustersVect[iic];
    }
    fTheResTuningM->SetElem(ikp, iic, theQMValue);
  }
}
// delete all memory allocated for threads!!!
//for (int i=0; i<kThreads; ++i)  {
//  delete theObjVect[i];
//  theBlas.Free(*(theClusterMembershipMVect[i]));
//  delete theClusterMembershipMVect[i];
//}


for (;ic>-1; ic--) {

//for (int ic=theSizeNumberOfClustersVect-1; ic>-1; ic--) {
  fNumberOfClusters = theNumberOfClustersVect[ic];

//    for (int ic=theSizeNumberOfClustersVect-1; ic>-1; ic--) {
//      fNumberOfClusters = theNumberOfClustersVect[ic];

      // generate code book for the given number of clusters on the Training Set 
      // i.e. on theAp
      // NOTE on the TRAINING SET
      if (fEncodingAndQM->GetQualityMeasureType()!=KscQMType::kBAS) {
        fEncodingAndQM->GenerateCodeBook(*fTheAprxScoreVariableM, fNumberOfClusters);
      } else {
        fEncodingAndQM->GenerateCodeBook(*fTheReducedSetCoefM, fNumberOfClusters);
      }  
      // perform the clustering and the quality measure computation
      T theQMValue = 0.;
      if (fEncodingAndQM->GetQualityMeasureType()==KscQMType::kBLF) {
        // == BLF ===============================================================
        fEncodingAndQM->ClusterDataSet(theAprxTestScoreVariableM, theClusterMembershipM, 0);
        // in case of BLF and K=2: compute the second variable vector as the col 
        // sums of the theReducedValidSetKernelM RxN_tr that gives the N_test 
        // second variable by adding the corresponding single bias term.
        Matrix<T>* theSecondVarForBLFM = nullptr;
        if (fNumberOfClusters==2) {
          theSecondVarForBLFM = new Matrix<T>(theNumValidData, 1);
          theBlas.Malloc(*theSecondVarForBLFM);
          for (size_t id=0; id<theNumValidData; ++id) {
            T sum =0.;
            for (size_t ir=0; ir<theRankOfAprx; ++ir) {
              sum += theReducedValidSetKernelM.GetElem(ir, id);
            }
            theSecondVarForBLFM->SetElem(id, 0, sum + fTheAprxBiasTermsM->GetElem(0,0));
          }
        }
        theQMValue = fEncodingAndQM->ComputeQualityMeasure(theClusterMembershipM, &theAprxTestScoreVariableM, theSecondVarForBLFM);
        if (theSecondVarForBLFM) {
          theBlas.Free(*theSecondVarForBLFM);
          delete theSecondVarForBLFM;
        }
      } else if (fEncodingAndQM->GetQualityMeasureType()==KscQMType::kAMS) {
        // == AMS ===============================================================
        fEncodingAndQM->ClusterDataSet(theAprxTestScoreVariableM, theClusterMembershipM, 1);
        theQMValue = fEncodingAndQM->ComputeQualityMeasure(theClusterMembershipM);
      } else {
        // == BAS ===============================================================
        fEncodingAndQM->ClusterDataSet(theAprxTestScoreVariableM, theClusterMembershipM, 1);
        theQMValue = fEncodingAndQM->ComputeQualityMeasure(theClusterMembershipM);
      }        
      // check the resulted quality measure
      if (theQMValue>optimalQMValue) {
        optimalQMValue             = theQMValue;
        optimalKernelParameterIndx = ikp;
        optimalNumberOfClusters    = fNumberOfClusters;
      }
      fTheResTuningM->SetElem(ikp, ic, theQMValue);    
    }


    
  }
  //
  // delete all memory allocated for threads!!!
  for (int i=0; i<kThreads; ++i)  {
    delete theObjVect[i];
    theBlas.Free(*(theClusterMembershipMVect[i]));
    delete theClusterMembershipMVect[i];
  }
  //
  // CLEAN ALLOCATED MEMORY 
  theBlas.Free(theAprxTestScoreVariableM);
  theBlas.Free(theReducedValidSetKernelM);
  theBlas.Free(theClusterMembershipM);
  //
  //
// this is the result of the all tuning !!!  
//  if (fTheResTuningM) {
//    theBlas.Free(*fTheResTuningM);
//    delete fTheResTuningM;
//    fTheResTuningM = nullptr;
//  }
  if (fTheReducedSetCoefM) {
    theBlas.Free(*fTheReducedSetCoefM);
    delete fTheReducedSetCoefM;
    fTheReducedSetCoefM = nullptr;
  }

  // set some results of the tuning
  fTheOptimalClusterNumber =  optimalNumberOfClusters;
  fTheOptimalKernelParIndx =  optimalKernelParameterIndx;
  fTheOptimalQMValue       =  optimalQMValue;
  // reset the pointer to the original, untouched Incomplete Cholesky matrix
  fIncCholeskyM = fIncCholeskyM_ptr;
}


template <class TKernel, typename T,  typename TInputD>
void
KscWkpcaIChol<TKernel, T, TInputD>::Test(Matrix<TInputD, false>& theTestInputDataM, size_t numBLASThreads, size_t qmFlag, int verbose) {
  // create the BLAS for memory managment and to call CPU BLAS, LAPACK methods
  BLAS  theBlas;
  theBlas.SetNumThreads(numBLASThreads, verbose);
  // set dimensions: reduced set size and test set size
  const size_t theDimTrData  = fInputTrainingDataM->GetNumCols();
  const size_t theRankOfAprx  = fTheReducedSetDataM->GetNumRows();
  const size_t theNumTestData = theTestInputDataM.GetNumRows();  
  //
  assert (theDimTrData==theTestInputDataM.GetNumCols() && " Training and Test data must have the same type (i.e. dimensions)");
  // allocate memory for the reduced-set - test-set kernel matrix and fill it
  Matrix<T> theReducedTestSetKernelM(theRankOfAprx, theNumTestData);
  theBlas.Malloc(theReducedTestSetKernelM);
  for (size_t ic=0; ic<theNumTestData; ++ic) {
    const TInputD* inpData1 = theTestInputDataM.GetPtrToBlock(ic);
    for (size_t ir=0; ir<theRankOfAprx; ++ir) {
      const TInputD* inpData2 = fTheReducedSetDataM->GetPtrToBlock(ir);
      const T valKernel = fKernel->Evaluate(inpData1, inpData2, theDimTrData);
      theReducedTestSetKernelM.SetElem(ir, ic, valKernel);
    }
  }
  // compute the approximated score variables of the TEST set 
  //  - theAprxScoreVariableM with (K-1)x(N_test)
  Matrix<T> theAprxTestScoreVariableM(fNumberOfClusters-1, theNumTestData);
  theBlas.Calloc(theAprxTestScoreVariableM); 
  theBlas.XGEMM(*fTheReducedSetCoefM, theReducedTestSetKernelM, theAprxTestScoreVariableM);
  // add the approximated bias terms (only if BLF or AMS)
  if (fEncodingAndQM->GetQualityMeasureType()!=KscQMType::kBAS) {
    for (size_t idat=0; idat<theNumTestData; ++idat) {
      for (size_t is=0; is<fNumberOfClusters-1; ++is) {
        theAprxTestScoreVariableM.SetElem(is, idat, theAprxTestScoreVariableM.GetElem(is, idat)+fTheAprxBiasTermsM->GetElem(is, 0));
      }
    }
  }
  //
  // === Perform the clustering and the quality measure computation
  // 
  // clean the membership matrix and allocate memory
  size_t ncl = qmFlag<2 ? qmFlag+1 : fNumberOfClusters+1;
  if (fTheClusterMembershipM) {
    theBlas.Free(*fTheClusterMembershipM);
    delete fTheClusterMembershipM;
  }
  fTheClusterMembershipM = new Matrix<T, false>(theNumTestData, ncl);
  theBlas.Malloc(*fTheClusterMembershipM);
  // the quality measure value
  T theQMValue = 0.;
  if (fEncodingAndQM->GetQualityMeasureType()==KscQMType::kBLF) {
    // == BLF ===============================================================
    fEncodingAndQM->ClusterDataSet(theAprxTestScoreVariableM, *fTheClusterMembershipM, qmFlag);
    // in case of BLF and K=2: compute the second variable vector as the col 
    // sums of the theReducedValidSetKernelM RxN_tr that gives the N_test 
    // second variable by adding the corresponding single bias term.
    Matrix<T>* theSecondVarForBLFM = nullptr;
    if (fNumberOfClusters==2) {
      theSecondVarForBLFM = new Matrix<T>(theNumTestData, 1);
      theBlas.Malloc(*theSecondVarForBLFM);
      for (size_t id=0; id<theNumTestData; ++id) {
        T sum =0.;
        for (size_t ir=0; ir<theRankOfAprx; ++ir) {
          sum += theReducedTestSetKernelM.GetElem(ir, id);
        }
        theSecondVarForBLFM->SetElem(id, 0, sum + fTheAprxBiasTermsM->GetElem(0,0));
      }
    }
    theQMValue = fEncodingAndQM->ComputeQualityMeasure(*fTheClusterMembershipM, &theAprxTestScoreVariableM, theSecondVarForBLFM);
    if (theSecondVarForBLFM) {
      theBlas.Free(*theSecondVarForBLFM);
      delete theSecondVarForBLFM;
    }
  } else if (fEncodingAndQM->GetQualityMeasureType()==KscQMType::kAMS) {
    // == AMS ===============================================================
    fEncodingAndQM->ClusterDataSet(theAprxTestScoreVariableM, *fTheClusterMembershipM, qmFlag);
    theQMValue = fEncodingAndQM->ComputeQualityMeasure(*fTheClusterMembershipM);
  } else {
    // == BAS ===============================================================
    fEncodingAndQM->ClusterDataSet(theAprxTestScoreVariableM, *fTheClusterMembershipM, qmFlag);
    theQMValue = fEncodingAndQM->ComputeQualityMeasure(*fTheClusterMembershipM);
  }  
  _unused(theQMValue);
  // clean memory allocated 
  theBlas.Free(theAprxTestScoreVariableM);
  theBlas.Free(theReducedTestSetKernelM);
} 


template <class TKernel, typename T,  typename TInputD>
void
KscWkpcaIChol<TKernel, T, TInputD>::ComputeKernelMatrix(Matrix<T>& theKernelM, Matrix<TInputD, false>& theRowDataM, Matrix<TInputD, false>& theColDataM, size_t numThreads) {
  // theKernelM.GetNumRows() input data will be used from theRowDataM to generate the rows
  // theKernelM.GetNumCols() input data will be used from theColDataM to generate the cols of the K(row, col)
  // const size_t numRows = theKernelM.GetNumRows();
  const size_t numCols = theKernelM.GetNumCols();
  // split along the col 
  //   ==> each thread will fill along rows -> memory continous 
  //   ==> each thread will take one continous data from theRowDataM
  const size_t sizeBlocks = (numThreads > 1) ? numCols/(double)numThreads : 0 ;
  size_t ic = 0;
  if (sizeBlocks>0) {
    std::vector<std::thread> theThreads;
    theThreads.reserve(numThreads);
    for (size_t it=0; it<numThreads; ++it) {
      theThreads.push_back( std::thread(&KscWkpcaIChol<TKernel, T, TInputD>::ComputeKernelMatrixPerTherad, this, std::ref(theKernelM), std::ref(theRowDataM), std::ref(theColDataM),  ic, ic+sizeBlocks) );
      ic += sizeBlocks;
    }
    for (size_t it=0; it<numThreads; ++it) {
      theThreads[it].join();
    }
  }
  // remaining
  ComputeKernelMatrixPerTherad(theKernelM, theRowDataM, theColDataM,  ic, numCols);
}

template <class TKernel, typename T,  typename TInputD>
void
KscWkpcaIChol<TKernel, T, TInputD>::ComputeKernelMatrixPerTherad(Matrix<T>& theKernelM, Matrix<TInputD, false>& theRowDataM, Matrix<TInputD, false>& theColDataM, size_t fromCol, size_t tillCol) {
  const size_t numRows = theKernelM.GetNumRows();
  const size_t theDim  = theRowDataM.GetNumCols();
  for (size_t ic=fromCol; ic<tillCol; ++ic) {
    const TInputD* inpData1 = theColDataM.GetPtrToBlock(ic);
    for (size_t ir=0; ir<numRows; ++ir) {
      const TInputD* inpData2 = theRowDataM.GetPtrToBlock(ir);
      const T valKernel = fKernel->Evaluate(inpData1, inpData2, theDim);
      theKernelM.SetElem(ir, ic, valKernel);
    }
  }
}
