#ifndef KSCWKPCA_HH
#define KSCWKPCA_HH

/**
 * Kernel Spectral Clustering formulated as weighted Kernel PCA.
 */

#include "Kernels.hh"
#include "Matrix.hh"


#include "KscEncodingAndQM_BLF.hh"
#include "KscEncodingAndQM_BAS.hh"
#include "KscEncodingAndQM_AMS.hh"


template <class TKernel, typename T,  typename TInputD>
class KscWkpca {

public:

  /**Constructor */
  KscWkpca() {
    fUseGPU                = false;
    // create the kernel functon object
    fKernel                = new TKernel();
    // input data needs to be set by the user
    fTheInpTrDataM         = nullptr; // not owned by the clas
    fEncodingAndQM         = nullptr; // owned by the class (AMS)
    // data generated during the training (Train) and all ownd by the class
    fTheBiasTermsM         = nullptr;
    fTheEigenVectorM       = nullptr;
    fTheClusterMembershipM = nullptr;
    //
    fTheTrScoreVariableM = nullptr;
    //
    fTheResTuningM         = nullptr;
  }

  /**Destructor. */
 ~KscWkpca() {
    BLAS  theBlas;
    if (fKernel)                delete fKernel;
    if (fEncodingAndQM)         delete fEncodingAndQM;
    //
    if (fTheBiasTermsM)         { theBlas.Free(*fTheBiasTermsM);         delete fTheBiasTermsM;}
    if (fTheEigenVectorM)       { theBlas.Free(*fTheEigenVectorM);       delete fTheEigenVectorM;}
    if (fTheClusterMembershipM) { theBlas.Free(*fTheClusterMembershipM); delete fTheClusterMembershipM;}
    if (fTheTrScoreVariableM)   { theBlas.Free(*fTheTrScoreVariableM);   delete fTheTrScoreVariableM;}
    if (fTheResTuningM)         { theBlas.Free(*fTheResTuningM);         delete fTheResTuningM;}
  }


  /**
   * Public method to set the parameters of the kernel function object.
   *
   * The type of the kernel function object is selected at instantiation of an
   * KscWkpca object since the class is templated on this type. This public
   * method can be used to set the parameters of the kernel function object.
   * Note, that the method will invoke the corresponding \f$ \texttt{SetKernelParameters}\f$
   * method of the kernel function object through the base class KernelBase::SetParameters
   * interface method by passing all provided parameters as arguments.
   *
   * @param[in] args input argument list.
   */
  template < typename... Args >
  void SetKernelParameters(Args... args) {
    fKernel->SetParameters(args...);
  }

  /**
   * Public method to set the pointer to the input data matrix.
   *
   * @param[in] inDataM pointer to the input data matrix that stores the input
   *   data vector in row-major (memory continous) order.
   * @note The class does NOT own the data i.e. the corresponding memory needs
   *   to be freed by the caller.
   */
  void  SetInputTrainingDataMatrix(Matrix<TInputD, false>* inTrDataM) { fTheInpTrDataM = inTrDataM; }


  /**
   * Set number of clusters to find in the training.
   * @param[in] nc number of required clusters
   */
  void   SetNumberOfClustersToFind(size_t nc)  { fNumberOfClusters = nc;   }

  /**Get number of clusters to find in the training.
   * @return Number of required clusters.
   */
  size_t GetNumberOfClustersToFind() const     { return fNumberOfClusters; }


//LOOK HERE: needs to be changed
  /**
    * Request to use GPU in the computations of the approximated eigenvectors related
    * to the training.
    *
    * When building with the \f$\texttt{-DUSE_CBLAS}\f$, \f$\texttt{CMake}\f$ configuration
    * option, i.e. with GPU (CUDA) support, the most compute-intensive part of the
    * training might be accelerated by using the GPU. This is the computation of
    * K-1 leading eigenvectors of the \f$D^{-1}M_D\Omega_{tr}\f$ matrix, especially
    * the eigen-decompsition using the LAPACK ?SYEVR that might be accelerated
    * by replacing it with the cuSolver SYEVDX. This computation is performed on
    * the GPU when this flag is turned to be true. This can especially be usefull
    * in case of larger training set sizes (\f$ N_{tr} > 2-3000 \f$).
    *
    * @param val The value to request GPU based computation. If \f$\texttt{true}\f$,
    *  computation of the K-1 leading eigenvector of the symmetric problem is
    *  done on the GPU instead of the CPU. Th computations si done by using the
    *  CPU otherwise (default).
    */
    void  SetUseGPU(bool val) { fUseGPU = val; }


  /**
   * Public method to set the cluster membership encoding/decoding (assigment)
   *        scheme and the corresponding and clustering quality measure.
   *
   * See more on the available cluster membership encodings, assigments and
   * the corresponding model evaluation criteria in KscEncodingAndQM base class.
   * The following encoding, assigment and model quality measures are supported:
   * 1. **sign based encoding** and **Hamming distance**: Balanced Line Fit (BLF) KscEncodingAndQM_BLF
   * 2. **direction based encoding-1** and **cosine distance**: Average Membership Strength (AMS) KscEncodingAndQM_AMS
   *
   * @param[in] qmType type of the cluster membership encoding and the corresponding
   *            quality measure
   *            - KscQMType::kBLF  KscEncodingAndQM_BLF
   *            - KscQMType::kAMS  KscEncodingAndQM_AMS
   */
  void   SetEncodingAndQualityMeasureType(KscQMType qmType) {
            if (fEncodingAndQM) delete fEncodingAndQM;
            switch (qmType) {
              case KscQMType::kBLF: fEncodingAndQM = new KscEncodingAndQM_BLF<T>(); break;
              case KscQMType::kAMS: fEncodingAndQM = new KscEncodingAndQM_AMS<T>(); break;
              default: fEncodingAndQM = new KscEncodingAndQM_AMS<T>();
            };
          }

  /**
   * Public method to get the encoding and quality measure object pointer.
   *
   * @return Pointer to the encoding and quality measure object.
   */
  const KscEncodingAndQM<T>* GetEncodingAndQualityMeasure() const { return fEncodingAndQM; }

  /**
   * Public method to set the weigth to be given to the balance term in the model
   * selection criterion.
   *
   * The model selection (quality measure) contains a term that accounts how
   * balanced is the result of the clustering. This parameter gives the weight
   * of this term over the other (collinearity) term.
   *
   * @param[in] etaBalance weight of the balance term. Must be in [0,1] where
   *    1 means that all the important is given to the balance term while 0
   *    removes this term from the qualty measure.
   */
  void   SetQualityMeasureEtaBalance(T etaBalance) {
      fEncodingAndQM->SetCoefEtaBalance(etaBalance);
  }


  /**
   * Public method to set the outlier threshold to be used in the model selection
   * criterion computation.
   *
   * When contibutions form the different cluster to the model selection (quality
   * measure) criterion is computed, clusters with cardinality below a certain
   * threshold are considered to contain outliers and the corresponding clusters
   * won't contibute to the model selection criterion value. This treshold can
   * be set by using this method.
   *
   * @param[in] val The minimum required cardinality or outlier threshold value.
   */
  void   SetQualityMeasureOutlierThreshold(size_t val) {
      fEncodingAndQM->SetOutlierThreshold(val);
  }


  /**
   * Public method to train the KSC model.
   *
   * The method will train a KSC model on the training data set which means that
   * all the required parameter values and quantities will be determined and
   * stored in the obejct. The object can be used to cluster any unseen input
   * data after the training using its Test() method. However, certain parameter
   * values, object pointers need to be set before invoking the training (see
   * below).
   *
   * **Training steps**:
   *
   * - computes the K-1 leading, eigenvectors of the \f$ D^{-1}M_D\Omega_{tr}\f$
   *   matrix (K is the number of required clusters and \$\Omega_{tr}\f$ is the
   *   training data kernel matrix)
   * - generates the cluster membership encoding
   * - (optionally) clusters the training data and computes the corresponding
   *   model selection criterion
   *
   * **Needs to be done before training**:
   *
   * - parameters of the kernel function object needs to be set by using the
   *   SetKernelParameters<>() method
   * - the training data matrix pointer must be set by using the SetInputTrainingDataMatrix()
   *   method
   * - the number of required clusters number of clusters must be set by the
   *   SetNumberOfClustersToFind()
   * - a cluster membership encoding scheme, that also defines the model selection
   *   criterion, must be set by SetEncodingAndQualityMeasureType() method
   *
   * @param numBLASThreads number of threads to be used in the BLAS and LAPACK
   *   functions (if the implementation used supports multi threading).
   * @param isQMOnTraining flag to indicate if the optional clustering of the
   *   training data set, with the model selection criterion calculation, should
   *   also be done (true by default). The corresponding result can be obtained
   *   by using the GetTheClusterMembershipMatrix().
   * @param qmFlag a value that determines what clustering information needs to
   *   be generated i.e. which one out of the 3 KscEncodingAndQM<>::ClusterDataPoint()
   *   interface should be used (generate only cluster assigment, also strength
   *   of this assigment or strength to for each cluster). Used only when clustering
   *   of the training data set is required i.e. when \f$\texttt{isQMOnTraining=true}\f$!
   * @param verbose verbosity level that controls the verbosity of the output information.
   *
   */
  void  Train(size_t numBLASThreads, bool isQMOnTraining=true, size_t qmFlag=1, int verbose=0);



  /**
   * Tuning of the hyper parameters of the KSC model.
   *
   * The KSC model depends on the number of required clusters and the given kernel
   * function parameters. This method trains a KSC model on the given training
   * data set and evaluates the model selection criterion on the given validation
   * data set over a 2D grid of cluster-number x kernel-parameters. The 2D grid
   * is determined by the input parameters and the resulted KSC model evaluation
   * criterion matrix can be obtained by using the GetTheTuningResultMatrix()
   * method. The 2D grid point, that gives the highest value of model evaluation
   * criterion, is also available through the GetTheOptimalClusterNumber() and
   * GetTheOptimalKernelParIndex() methods. However, a more careful investigation
   * of the resulted model evaluation surface is suggested to select the optimal
   * KSC model parameters.
   *
   * Since the KSC model is trained on the training data set at each point of 2D
   * parameter grid, **the same things needs to be done as described at the Train()
   * method before invoking the Tune() method***.
   *
   * @param theKernelParametersVect vector that contains the kernel parameters
   *  that determines the rows of the 2D parameter grid. The SetKernelParameters<>()
   *  method will be invoked for each of these paraneters during the tuning.
   * @param minNumClusters minim of the cluster number parameter that determines
   *  the minimum value of the columns of the 2D parameter grid.
   * @param maxNumClusters maximum of the cluster number parameter that determines
   *  the maximum value of the columns of the 2D parameter grid.
   * @param theValidInputDataM reference to the validation data set on which the
   *  model evaluation criterion will be evaluated at each point of the 2D parameter
   *  grid after training the model on the training data set. Note, that the training
   *  data set needs to be set by the SetInputTrainingDataMatrix() method before
   * invoking this Tune() method.
   * @param numBLASThreads number of threads to be used in the BLAS and LAPACK
   *   functions (if the implementation used supports multi threading).
   * @param verbose verbosity level that controls the verbosity of the output information.
   *
   */
  template < typename TKernelParameterType >
  void  Tune(std::vector<TKernelParameterType>& theKernelParametersVect, size_t minNumClusters, size_t maxNumClusters,  Matrix<TInputD, false>& theValidInputDataM, size_t numBLASThreads, int verbose=0);



  void  Test(Matrix<TInputD, false>& theTestInputDataM, size_t numBLASThreads, size_t qmFlag=1, int verbose=0);


  /**
   * Public method to obtain the result of the clustering.
   *
   * @return Pointer to the matrix that stores the result of the clustering. Ecah
   *   row correspond to the result obtained for the input data with the corresponding
   *   row index in either the permuted input training data matrix (after Train())
   *   or int the test data matrix (after Test()).
   */
  const Matrix<T, false>*  GetTheClusterMembershipMatrix() const { return fTheClusterMembershipM; }

  /**
   * Obtain pointer to the model evaluation criterion matrix over the 2D KSC parameter
   * grid generated during the tuning (Tune()).
   * @return Pointer to the matrix that contains the KSC model evaluation criterion values
   *  over the 2D paraeter grid. Each row of the matrix contains model quality measure
   *  values that belongs to one kernel parameter and each column contains the values
   *  for a given cluster number.
   */
  const Matrix<T, false>*  GetTheTuningResultMatrix() const { return fTheResTuningM; }
  size_t    GetTheOptimalClusterNumber()              const { return fTheOptimalClusterNumber; }
  size_t    GetTheOptimalKernelParIndex()             const { return fTheOptimalKernelParIndx; }
  T         GetTheQMValueAtTheOptimalPoint()          const { return fTheOptimalQMValue; }


private:

  /** Auxilary method to compute the K-1 leading eigenvetors of the \f$ D^{-1}M_D\Omega_{tr} \f$ matrix.*/
  void ComputeEigenvectors(Matrix<T>& theTrainingSetKernelM, std::vector<T>& theDegreeVect, int numBLASThreads, int verbose=0);

  void ComputeKernelMatrix(Matrix<T>& theKernelM, Matrix<TInputD, false>& theRowDataM, Matrix<TInputD, false>& theColDataM, size_t numThreads);
  void ComputeKernelMatrixPerTherad(Matrix<T>& theKernelM, Matrix<TInputD, false>& theRowDataM, Matrix<TInputD, false>& theColDataM, size_t fromCol, size_t tillCol);

private:
/**
 * @name Data members
 */
// @{
  /** Use GPU (if available and built with the -DUSE_CUBLAS CMake option) in the
    * approximated training set eigenvector computation.
    */
  bool                        fUseGPU;
  /**Number of clusters to find. */
  size_t                      fNumberOfClusters;
  /**Pointer to the kernel object that implements the kernel function. */
  TKernel*                    fKernel;
  /**Pointer to the input data to be used for training (doesn't owned by the class). */
  Matrix<TInputD, false>*     fTheInpTrDataM;
  //
  KscEncodingAndQM<T>*        fEncodingAndQM;
  //
  // Set during the training (and Test i.e. fTheClusterMembershipM)
  //
  /**Only if BLF or AMS encoding and QM; Approximated bias terms (K-1); computed like Eq.(26) with substit of after Eq(56) */
  Matrix<T>*                  fTheBiasTermsM;
  Matrix<T>*                  fTheEigenVectorM;       // K-1 leading of the D^-1 M_D Omega_tr matrix
  Matrix<T, false>*           fTheClusterMembershipM;
  // kept in Train() only if isQMOnTraining = false i.e. for Tuning
  Matrix<T>*                  fTheTrScoreVariableM;
  //
  // Set during the tuning (each row corresponds to a given kernel parameter
  // and each col. corresponds to a given cluster number)
  Matrix<T, false>*           fTheResTuningM;
  size_t                      fTheOptimalClusterNumber;
  size_t                      fTheOptimalKernelParIndx;
  T                           fTheOptimalQMValue;

  //

// @} // end Data members  group


};

#include "KscWkpca.tpp"

#endif
