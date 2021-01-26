// for std::memcpy
#include <cstring>
#include <thread>

//
// Note, the following needs to be done before calling the training:
//   - parameters of the kernel function must be set (SetKernelParameters)
//   - the training data matrix pointer must be set  (SetInputTrainingDataMatrix)
//   - number of clusters to be found must be set  (SetNumberOfClustersToFind)
//   - a cluster encoding, assigment and quality measure must be set (SetEncodingAndQualityMeasureType)
template <class TKernel, typename T,  typename TInputD>
void
KscWkpca<TKernel, T, TInputD>::Train(size_t numBLASThreads, bool isQMOnTraining, size_t qmFlag, int verbose) {
  // create the BLAS for memory managment and to call CPU BLAS, LAKACK methods
  BLAS  theBlas;
  theBlas.SetNumThreads(numBLASThreads, verbose);
  const int kThreads = numBLASThreads;
  // set number of the training data
  const size_t theNumTrData  = fTheInpTrDataM->GetNumRows(); // (N_tr); row major!
//  const size_t theDimTrData  = fTheInpTrDataM->GetNumCols(); // each row is one data
  //
  // ------------------------------------------------------------------------ //
  // 1. Compute the K-1 leading eigenvectors of the D^{-1}M_D\Omega
  //    matrix and the corresponding K-1 bias terms.
  //
  if (verbose>1) {
    std::cout<< "          ====> Starts forming the training set kernel matrix... " << std::endl;
  }
  Matrix<T> theTrainingSetKernelM(theNumTrData, theNumTrData);
  theBlas.Malloc(theTrainingSetKernelM);
  ComputeKernelMatrix(theTrainingSetKernelM, *fTheInpTrDataM, *fTheInpTrDataM, kThreads);
  // - the matrix that will store the K-1 leading eigenvectors of the D^-1 M_D Omega_tr
  if (fTheEigenVectorM) {
    theBlas.Free(*fTheEigenVectorM);
    delete fTheEigenVectorM;
  }
  fTheEigenVectorM = new Matrix<T>(theNumTrData, fNumberOfClusters-1);
  theBlas.Malloc(*fTheEigenVectorM);
  if (verbose>1) {
    std::cout<< "          ====> Starts computing the leading eigenvectors of the D^-1 M_D Omega matrix... " << std::endl;
  }
  // Note: the col./row sums of the training set kernel matrix is needed when
  //       the BLF QM is used and QM value need to be computed for K=2. So store it.
  std::vector<T> theDegreeVect(theNumTrData,0.);
  ComputeEigenvectors(theTrainingSetKernelM, theDegreeVect, numBLASThreads, verbose);
  theBlas.Free(theTrainingSetKernelM);
  //
  // ------------------------------------------------------------------------ //
  // 3. Generate cluster membership encoding
  //    - the code book i.e. the cluster membership encoding is generated.
  if (verbose>1) {
    std::cout<< "          ====> Starts generating encoding... " << std::endl;
  }
  fEncodingAndQM->GenerateCodeBook(*fTheTrScoreVariableM, fNumberOfClusters);
  //
  // ------------------------------------------------------------------------ //
  // 4. The training is done at this point. The last part, clustering the training
  //    data and computing the corresponding quality measure, is optional.
  //  4.1. Compute KSC model quality measure on the this training set data.
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
      fEncodingAndQM->ClusterDataSet(*fTheTrScoreVariableM, *fTheClusterMembershipM, qmFlag);
      // 3.3. a. BLF: in case of BLF and K=2: compute the second variable vector
      //      as the col sums of the theTrainingSetKernelM N_trxN_tr that
      //      gives the N_tr second variable by adding the single bias term.
      Matrix<T>* theSecondVarForBLFM = nullptr;
      if (fNumberOfClusters==2) {
        theSecondVarForBLFM = new Matrix<T>(theNumTrData, 1);
        theBlas.Malloc(*theSecondVarForBLFM);
        const T theBias = fTheBiasTermsM->GetElem(0,0);
        for (size_t id=0; id<theNumTrData; ++id) {
          theSecondVarForBLFM->SetElem(id, 0, theDegreeVect[id] + theBias);
        }
      }
      theQMValue = fEncodingAndQM->ComputeQualityMeasure(*fTheClusterMembershipM, fTheTrScoreVariableM, theSecondVarForBLFM);
      if (theSecondVarForBLFM) {
        theBlas.Free(*theSecondVarForBLFM);
        delete theSecondVarForBLFM;
      }
    } else { // kAMS
      // == AMS ===============================================================
      fEncodingAndQM->ClusterDataSet(*fTheTrScoreVariableM, *fTheClusterMembershipM, qmFlag);
      theQMValue = fEncodingAndQM->ComputeQualityMeasure(*fTheClusterMembershipM);
      std::cout << " theQMValue = " << theQMValue << std::endl;
    }
    _unused(theQMValue);
  }
}


template <class TKernel, typename T,  typename TInputD>
void
KscWkpca<TKernel, T, TInputD>::ComputeEigenvectors(Matrix<T>& theTrainingSetKernelM, std::vector<T>& theDegreeVect, int numBLASThreads, int verbose) {
  // create the BLAS for memory managment and to call CPU BLAS, LAKACK methods
  BLAS  theBlas;
  theBlas.SetNumThreads(numBLASThreads, verbose);
  // set dimensions: number, dimension of the training data a
  const size_t theNumTrData  = fTheInpTrDataM->GetNumRows(); // (N_tr); row major!
  //
  // 1. Compute the diagonal degree matrix: D_ii = sum_{j=1}^{N_tr} \Omega_{ij}
  if (verbose>2) {
    std::cout<< "             ...... degree business starts... " << std::endl;
  }
  std::vector<T> theInvDegreeVect(theNumTrData, 0.);
  std::vector<T> theInvDegreeVectSqrt(theNumTrData, 0.);
  std::vector<T> dumv(theNumTrData, 0.); // weighted mean of the rows of [M_D \Omega]
  // make sure we move along memory -> vectorized inner loop
  T theSumInvDegree = 0.;
  for (size_t ic=0; ic<theNumTrData; ++ic) {
    T dum0 = 0.;
    for (size_t ir=0; ir<theNumTrData; ++ir) { // along memory <- col major : vectorized
      dum0 += theTrainingSetKernelM.GetElem(ir, ic);
    }
    dum0 = std::max(dum0, 1.0E-16);
    theDegreeVect[ic]        = dum0;
    theInvDegreeVect[ic]     = 1./dum0;
    theInvDegreeVectSqrt[ic] = std::sqrt(theInvDegreeVect[ic]);
    theSumInvDegree         += theInvDegreeVect[ic];
  }
  theSumInvDegree = 1./theSumInvDegree;
  //
  // 2. Compute the D^{-1/2} [M_d Omega M_D^T] D^{-1/2} matix:
  // - comput weighted sum of cols and remove them
  for (size_t ic=0; ic<theNumTrData; ++ic) {
    T dum0 = 0.;
    for (size_t ir=0; ir<theNumTrData; ++ir) { // along memory <- col major : vectorized
      dum0 += theTrainingSetKernelM.GetElem(ir, ic)*theInvDegreeVect[ir];
    }
    dum0 *= theSumInvDegree; // normalisation ==> 1/d_i weighted mean of the ic-th col
    for (size_t ir=0; ir<theNumTrData; ++ir) { // along memory <- col major : vectorized
      const T wval    = theTrainingSetKernelM.GetElem(ir, ic) - dum0;
      theTrainingSetKernelM.SetElem(ir, ic, wval);
      // already compute 1/d_j weighted row sums of the result
      dumv[ir] += wval*theInvDegreeVect[ic];
    }
  }
  // remove now the weighted row means and generate M_d Omega M_D^T
  // also apply the D^{-1/2} [M_d Omega M_D^T] D^{-1/2}
  for (size_t ic=0; ic<theNumTrData; ++ic) {
    const T invDegreeSqrt = theInvDegreeVectSqrt[ic];
    for (size_t ir=0; ir<theNumTrData; ++ir) {
      const T wval = theTrainingSetKernelM.GetElem(ir, ic) - dumv[ir]*theSumInvDegree;
      theTrainingSetKernelM.SetElem(ir, ic, wval*invDegreeSqrt*theInvDegreeVectSqrt[ir]);
    }
  }
  //
  // 3. Compute the K-1 leading eignevectors of the D^{-1/2} [M_d Omega M_D^T] D^{-1/2} matrix
  if (verbose>2) {
    std::cout<< "             ...... computing the K-1 leading eigenvectors starts... " << std::endl;
  }
  Matrix<T> theEigenValueVect0(fNumberOfClusters-1, 1);
  theBlas.Malloc(theEigenValueVect0);
  Matrix<T> theEigenVectorM0(theNumTrData, fNumberOfClusters-1);
  theBlas.Malloc(theEigenVectorM0);
  //
  // compute the K-1 leading eignevectors of the training data kernel matrix
  // note: the eigen values/vectors are in ascending order so the index goes from
  //       N_tr-(K-1)+1 : N_Tr (also note that indices are 1,2,...,N_tr i.e. strat form 1.
#if USE_CUBLAS
  if ( fUseGPU ) {
      BLAS_gpu  theBlas_gpu;
      if (verbose>2) {
        std::cout<< "             ...... on GPU" << std::endl;
      }
      // construct and allocate memory for on device
      Matrix<T> theEigenValueVect_d(fNumberOfClusters-1, 1);
      Matrix<T> theTrainingSetKernelM_d(theNumTrData, theNumTrData);
      Matrix<T> theEigenVectorM_d(theNumTrData, fNumberOfClusters-1);
      theBlas_gpu.Malloc(theEigenValueVect_d);
      theBlas_gpu.Malloc(theTrainingSetKernelM_d);
      theBlas_gpu.Malloc(theEigenVectorM_d);
      // copy the training data kernel matrix to the device
      theBlas_gpu.CopyToGPU(theTrainingSetKernelM, theTrainingSetKernelM_d);
      // compute the K-1 leading eigenevctors
      theBlas_gpu.XSYEVDX(theTrainingSetKernelM_d, theEigenValueVect_d, theEigenVectorM_d, 2, static_cast<T>(theNumTrData-fNumberOfClusters+2), static_cast<T>(theNumTrData));
      // copy the reults from the device to the host
      theBlas_gpu.CopyFromGPU(theEigenValueVect_d, theEigenValueVect0);
      theBlas_gpu.CopyFromGPU(theEigenVectorM_d, theEigenVectorM0);
      // clear all device side memory
      theBlas_gpu.Free(theEigenValueVect_d);
      theBlas_gpu.Free(theTrainingSetKernelM_d);
      theBlas_gpu.Free(theEigenVectorM_d);
  } else {
#endif  // USE_CUBLAS

      theBlas.XSYEVR(theTrainingSetKernelM, theEigenValueVect0, theEigenVectorM0, 2, static_cast<T>(theNumTrData-fNumberOfClusters+2), static_cast<T>(theNumTrData));

#if USE_CUBLAS
  }
#endif  // USE_CUBLAS
  // reverse order of the eigenvectors and eigen-values in order to have them in
  // descending order
  Matrix<T> theEigenValueVect(fNumberOfClusters-1, 1);
  theBlas.Malloc(theEigenValueVect);
  const size_t aSize = sizeof(T)*theNumTrData; // theNumTrData = theEigenValueVect0.GetNumRows()
  size_t j = 0;
  for (int i=fNumberOfClusters-2; i>-1; --i) {
    theEigenValueVect.SetElem(j,0, theEigenValueVect0.GetElem(i,0));
    // copy entire col
    std::memcpy(fTheEigenVectorM->GetPtrToBlock(j), theEigenVectorM0.GetPtrToBlock(i), aSize);
    ++j;
  }
  // free auxiliary memory
  theBlas.Free(theEigenValueVect0);
  theBlas.Free(theEigenVectorM0);
  //
  // 4. Compute the eigenvectors of the original, non-symetric problem
  //    by left multiplying the K-1 leading eigenvetors (fTheEigenVectorM) of the
  //    symmetric problem with \tilde{D}^-1/2
  if (verbose>2) {
    std::cout<< "             ...... computing the original eigenvectors starts ... " << std::endl;
  }
  for (size_t ic=0; ic<fNumberOfClusters-1; ++ic) {
    for (size_t ir=0; ir<theNumTrData; ++ir) {
      fTheEigenVectorM->SetElem(ir, ic, fTheEigenVectorM->GetElem(ir, ic)*theInvDegreeVectSqrt[ir]);
    }
  }
  //
  // The fNumberOfClusters-1 leading eigenvectors of the D^{-1}M_D Omega
  // matrix are now in the fTheEigenVectorM matrix !!!
  //
  // 5. Calculation of the K-1 bias terms according to Eq.(2.26) in my thesis
  if (verbose>2) {
    std::cout<< "             ...... computing bias terms starts ... " << std::endl;
  }
  // clean the memory and (re-)allocate
  if (fTheBiasTermsM) {
    theBlas.Free(*fTheBiasTermsM);
    delete fTheBiasTermsM;
    fTheBiasTermsM = nullptr;
  }
  fTheBiasTermsM = new Matrix<T>(fNumberOfClusters-1, 1);
  theBlas.Malloc(*fTheBiasTermsM);
  const T invNumTrData = 1./theNumTrData;
  for (size_t ic=0; ic<fNumberOfClusters-1; ++ic) {
    T dum0 = 0;
    for (size_t ir=0; ir<theNumTrData; ++ir) {
      dum0 += fTheEigenVectorM->GetElem(ir, ic)*theDegreeVect[ir];
    }
    const T dum1 = theEigenValueVect.GetElem(ic, 0);
    fTheBiasTermsM->SetElem ( ic, 0, dum0*(dum1-1.)*invNumTrData );
  }
  //
  // 6. Compute the score variables as \lambda^(k)D\beta^(k) according to
  //    Eq.(2.19)
  //    Note: Z will be K-1 x N_tr

  if (fTheTrScoreVariableM) {
    theBlas.Free(*fTheTrScoreVariableM);
    delete fTheTrScoreVariableM;
  }
  fTheTrScoreVariableM = new Matrix<T>(fNumberOfClusters-1, theNumTrData);
  theBlas.Calloc(*fTheTrScoreVariableM);
  for (size_t ik=0; ik<fNumberOfClusters-1; ++ik) {
    const T lambda = theEigenValueVect.GetElem(ik,0);
    for (size_t id=0; id<theNumTrData; ++id) {
      fTheTrScoreVariableM->SetElem(ik, id, lambda*fTheEigenVectorM->GetElem(id, ik)*theDegreeVect[id]);
    }
  }
  //
  // clean the memory allocated for theSigmaVect
  theBlas.Free(theEigenValueVect);
}





template < class TKernel, typename T,  typename TInputD>
template < typename TKernelParameterType >
void
KscWkpca<TKernel, T, TInputD>::Tune(std::vector<TKernelParameterType>& theKernelParametersVect, size_t minNumberOfClusters, size_t maxNumberOfClusters,  Matrix<TInputD, false>& theValidInputDataM, size_t numBLASThreads, int verbose) {
  // create the BLAS for memory managment and to call CPU BLAS, LAPACK methods
  BLAS  theBlas;
  theBlas.SetNumThreads(numBLASThreads, verbose);
  // number of training and validation data points
  const size_t theNumTrData    = fTheInpTrDataM->GetNumRows(); // (N_tr); row major!
  const size_t theNumValidData = theValidInputDataM.GetNumRows();   // (N_v); row major!
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
  // == the test set score variable matrix with max required capacity: Z in (K-1) x N_v
  Matrix<T> theTestScoreVariableM(maxNumberOfClusters-1, theNumValidData);
  theBlas.Malloc(theTestScoreVariableM); // will be set to zero


  // == the training-valid set kernel matrix: Omega in N_tr x N_v
  Matrix<T> theTrainingValidSetKernelM(theNumTrData, theNumValidData);
  theBlas.Malloc(theTrainingValidSetKernelM);
  // == clustering results
  Matrix<T, false> theClusterMembershipM(theNumValidData, 2);
  theBlas.Malloc(theClusterMembershipM);
  //

const int kThreads = numBLASThreads;
KscEncodingAndQM<T>* theObjVect[kThreads];
Matrix<T, false>*    theClusterMembershipMVect[kThreads];
std::vector<int> intIndx(kThreads);
for (int i=0; i<kThreads; ++i)  {
  if (fEncodingAndQM->GetQualityMeasureType() == KscQMType::kBLF) {
     theObjVect[i] = new KscEncodingAndQM_BLF<T>();
  } else {
     theObjVect[i] = new KscEncodingAndQM_AMS<T>();
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
      std::cout<< "  === KscWkpca::Tune : tuning for the " << ikp << "-th kernel parameters out of the " << theKernelParametersVect.size()-1<< std::endl;
    }
    // 2. Use the current kernel parameters, train a model and evaluate its
    //    performance on the test-data set at each possible number-of-clusters
    //    on the [minNumberOfClusters, maxNumberOfClusters].
    //  2.1. - Set the kernel parameters to the current value and the number of
    //         required clusters to maxNumberOfClusters
    fKernel->SetParameters(theKernelParametersVect[ikp]);
    fNumberOfClusters = maxNumberOfClusters;
    //  2.2. - Train a model with using this maxNumberOfClusters using the training
    //         data: this will set the model for maxNumberOfClusters.
    // false => do not cluster and compute QM for the training data
    // NOTE: Training on the training data set
    if (verbose > 2 ) {
      std::cout<< "      --- Starts training... " << std::endl;
    }
    Train(numBLASThreads, false);
    //
    // compute the training - validation set kernel matrix with the current kernel
    // parameter
    if (verbose > 2 ) {
      std::cout<< "      --- Starts forming the training-validation set kenel matrix... " << std::endl;
    }
    ComputeKernelMatrix(theTrainingValidSetKernelM, *fTheInpTrDataM, theValidInputDataM, kThreads);
    //  2.3. - compute the score variables on the TEST set for the maxNumberOfClusters-1
    //         case: theTestScoreVariableM with (maxK-1)x(N_test)
    if (verbose > 2 ) {
      std::cout<< "      --- Score matrix computation... " << std::endl;
    }
//    std::memset(theTestScoreVariableM.GetDataPtr(), 0, sizeof(T)*theTestScoreVariableM.GetSize());
    // Z in (K-1)xN_v  =   B^T Omega  where B in (K-1)xN_tr and Omega in N_tr x N_v
    theBlas.XGEMM(*fTheEigenVectorM, theTrainingValidSetKernelM, theTestScoreVariableM, 1., 0., true, false);
    // add the bias terms
    for (size_t idat=0; idat<theNumValidData; ++idat) {
      for (size_t is=0; is<fNumberOfClusters-1; ++is) {
        theTestScoreVariableM.SetElem(is, idat, theTestScoreVariableM.GetElem(is, idat)+fTheBiasTermsM->GetElem(is, 0));
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
  Matrix<T>* encodeMatrix = fTheTrScoreVariableM;

  // NOTE: in case of BLF the K=2 case will be skipped and left to the tail processing
  for (int t=0; t<kThreads; ++t) {
    if (fEncodingAndQM->GetQualityMeasureType() != KscQMType::kBLF || theNumberOfClustersVect[ic] != 2) {
      theThreads[t] = std::thread(&KscEncodingAndQM<T>::DoAll, theObjVect[t], std::move(encodeMatrix), theNumberOfClustersVect[ic], std::move(&theTestScoreVariableM), std::move(theClusterMembershipMVect[t]));
      //theObjVect[t]->DoAll(encodeMatrix, fNumberOfClusters, &theTestScoreVariableM, theClusterMembershipMVect[ic%kThreads]);
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
      fEncodingAndQM->GenerateCodeBook(*fTheTrScoreVariableM, fNumberOfClusters);
      // perform the clustering and the quality measure computation
      T theQMValue = 0.;
      if (fEncodingAndQM->GetQualityMeasureType()==KscQMType::kBLF) {
        // == BLF ===============================================================
        fEncodingAndQM->ClusterDataSet(theTestScoreVariableM, theClusterMembershipM, 0);
        // in case of BLF and K=2: compute the second variable vector as the col
        // sums of the theTrainingValidSetKernelM (K-1)xN_v that gives the N_v
        // second variable by adding the corresponding single bias term.
        Matrix<T>* theSecondVarForBLFM = nullptr;
        if (fNumberOfClusters==2) {
          theSecondVarForBLFM = new Matrix<T>(theNumValidData, 1);
          theBlas.Malloc(*theSecondVarForBLFM);
          for (size_t id=0; id<theNumValidData; ++id) {
            T sum =0.;
            for (size_t ir=0; ir<theNumTrData; ++ir) {
              sum += theTrainingValidSetKernelM.GetElem(ir, id);
            }
            theSecondVarForBLFM->SetElem(id, 0, sum + fTheBiasTermsM->GetElem(0,0));
          }
        }
        theQMValue = fEncodingAndQM->ComputeQualityMeasure(theClusterMembershipM, &theTestScoreVariableM, theSecondVarForBLFM);
        if (theSecondVarForBLFM) {
          theBlas.Free(*theSecondVarForBLFM);
          delete theSecondVarForBLFM;
        }
      } else {
        // == AMS ===============================================================
        fEncodingAndQM->ClusterDataSet(theTestScoreVariableM, theClusterMembershipM, 1);
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
  theBlas.Free(theTestScoreVariableM);
  theBlas.Free(theTrainingValidSetKernelM);
  theBlas.Free(theClusterMembershipM);
  //
  //
// this is the result of the all tuning !!!
//  if (fTheResTuningM) {
//    theBlas.Free(*fTheResTuningM);
//    delete fTheResTuningM;
//    fTheResTuningM = nullptr;
//  }
  // set some results of the tuning
  fTheOptimalClusterNumber =  optimalNumberOfClusters;
  fTheOptimalKernelParIndx =  optimalKernelParameterIndx;
  fTheOptimalQMValue       =  optimalQMValue;
}





template <class TKernel, typename T,  typename TInputD>
void
KscWkpca<TKernel, T, TInputD>::Test(Matrix<TInputD, false>& theTestInputDataM, size_t numBLASThreads, size_t qmFlag, int verbose) {
  // create the BLAS for memory managment and to call CPU BLAS, LAPACK methods
  BLAS  theBlas;
  theBlas.SetNumThreads(numBLASThreads, verbose);
  // set training, test data number and their dimensions
  const size_t theDimTrData   = fTheInpTrDataM->GetNumCols();
  const size_t theNumTrData   = fTheInpTrDataM->GetNumRows();
  const size_t theNumTestData = theTestInputDataM.GetNumRows();
  //
  assert (theDimTrData==theTestInputDataM.GetNumCols() && " Training and Test data must have the same type (i.e. dimensions)");
  // allocate memory for the training - test-set kernel matrix and fill it
  Matrix<T> theTrainingTestSetKernelM(theNumTrData, theNumTestData);
  theBlas.Malloc(theTrainingTestSetKernelM);
  for (size_t ic=0; ic<theNumTestData; ++ic) {
    const TInputD* inpData1 = theTestInputDataM.GetPtrToBlock(ic);
    for (size_t ir=0; ir<theNumTrData; ++ir) {
      const TInputD* inpData2 = fTheInpTrDataM->GetPtrToBlock(ir);
      const T valKernel = fKernel->Evaluate(inpData1, inpData2, theDimTrData);
      theTrainingTestSetKernelM.SetElem(ir, ic, valKernel);
    }
  }
  // compute the score variables of the TEST set
  //  - theTestScoreVariableM with (K-1)x(N_test)
  Matrix<T> theTestScoreVariableM(fNumberOfClusters-1, theNumTestData);
  theBlas.Calloc(theTestScoreVariableM);
  // Z in (K-1)xN_v  =   B^T Omega  where B in (K-1)xN_tr and Omega in N_tr x N_v
  theBlas.XGEMM(*fTheEigenVectorM, theTrainingTestSetKernelM, theTestScoreVariableM, 1., 0., true, false);
  // add the bias terms
  for (size_t idat=0; idat<theNumTestData; ++idat) {
    for (size_t is=0; is<fNumberOfClusters-1; ++is) {
      theTestScoreVariableM.SetElem(is, idat, theTestScoreVariableM.GetElem(is, idat)+fTheBiasTermsM->GetElem(is, 0));
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
    fEncodingAndQM->ClusterDataSet(theTestScoreVariableM, *fTheClusterMembershipM, qmFlag);
    // in case of BLF and K=2: compute the second variable vector as the col
    // sums of the theTrainingTestSetKernelM N_trxN_v that gives the N_v
    // second variable by adding the corresponding single bias term.
    Matrix<T>* theSecondVarForBLFM = nullptr;
    if (fNumberOfClusters==2) {
      theSecondVarForBLFM = new Matrix<T>(theNumTestData, 1);
      theBlas.Malloc(*theSecondVarForBLFM);
      for (size_t id=0; id<theNumTestData; ++id) {
        T sum =0.;
        for (size_t ir=0; ir<theNumTrData; ++ir) {
          sum += theTrainingTestSetKernelM.GetElem(ir, id);
        }
        theSecondVarForBLFM->SetElem(id, 0, sum + fTheBiasTermsM->GetElem(0,0));
      }
    }
    theQMValue = fEncodingAndQM->ComputeQualityMeasure(*fTheClusterMembershipM, &theTestScoreVariableM, theSecondVarForBLFM);
    if (theSecondVarForBLFM) {
      theBlas.Free(*theSecondVarForBLFM);
      delete theSecondVarForBLFM;
    }
  } else {
    // == AMS ===============================================================
    fEncodingAndQM->ClusterDataSet(theTestScoreVariableM, *fTheClusterMembershipM, qmFlag);
    theQMValue = fEncodingAndQM->ComputeQualityMeasure(*fTheClusterMembershipM);
  }
  _unused(theQMValue);
  // clean memory allocated
  theBlas.Free(theTestScoreVariableM);
  theBlas.Free(theTrainingTestSetKernelM);
}




template <class TKernel, typename T,  typename TInputD>
void
KscWkpca<TKernel, T, TInputD>::ComputeKernelMatrix(Matrix<T>& theKernelM, Matrix<TInputD, false>& theRowDataM, Matrix<TInputD, false>& theColDataM, size_t numThreads) {
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
      theThreads.push_back( std::thread(&KscWkpca<TKernel, T, TInputD>::ComputeKernelMatrixPerTherad, this, std::ref(theKernelM), std::ref(theRowDataM), std::ref(theColDataM),  ic, ic+sizeBlocks) );
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
KscWkpca<TKernel, T, TInputD>::ComputeKernelMatrixPerTherad(Matrix<T>& theKernelM, Matrix<TInputD, false>& theRowDataM, Matrix<TInputD, false>& theColDataM, size_t fromCol, size_t tillCol) {
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
