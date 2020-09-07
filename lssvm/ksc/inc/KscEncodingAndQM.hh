

#ifndef KSCENCODINGANDQM_HH
#define KSCENCODINGANDQM_HH

#include <vector>
#include <string>
#include <algorithm>

#include "Matrix.hh"



/**
 * @enum KscQMType
 * @brief Cluster membership encoding and clustering quality measure types.
 */
enum KscQMType { 
  kBLF,  /** sign based encoding and Balanced Line Fit quality measure */
  kAMS,  /**< direction based (score data + cosine distance) encoding and Average Membership Strength quality measure */
  kBAS   /**< direction based (reduced set coeffitient data and Euclidean distance) encoding and Balanced Angular Similarity quality measure*/
};



/**
 * @class KscEncodingAndQM
 * @brief Base class for different KSC cluster membership encoding and quality
 *        measure implementations.
 *
 * This base class provides interfaces for implementing different **cluster**  
 * **membership encoding**, **cluster assigment** depending on the encoding and 
 * **model evaluation criterion** for model selection.
 * - **cluster membership encoding**:
 *    each cluster is represented by a special vector that depends on the selected 
 *    cluster encoding scheme. These cluster representations are generated based 
 *    on a training data set by calling the KscEncodingAndQM<T>::GenerateCodeBook interface method. Each 
 *    derived class can implement its own algorithm to gerate these vectors
 *    depending on the corresponding ecoding scheme. 
 *    
 *    Since all implemented cluster membership ecoding schemes make use of the 
 *    **sign** (of the training score data) **based encoding** when generating their own
 *    encodings, this **is implemented** here **in this base class** KscEncodingAndQM<T>::GenerateCodeBook method. 
 * - **cluster assigment**:
 *    when assigning a data to a cluster, each scheme (independently from the 
 *    type of the encoding) needs to compute the distance of this data point 
 *    measured from the clustres i.e. fromthe vectors representing each clustres. 
 *    Both the type of these cluster prototype vectors and the distance computation 
 *    depends on the selected encoding scheme. Therefore, the KscEncodingAndQM<T>::ComputeDistance 
 *    interface method is provided for the implementaton of computing the distance 
 *    between the score variable space representation of an input data point and 
 *    a cluster. This computation is used then in the KscEncodingAndQM<T>::ClusterDataPoint interface 
 *    methods to assign the input data point to a cluster and (in some cases like 
 *    BAS, AMS) compute further infomation regarding the strength of this 
 *    membership. A higher level method, the KscEncodingAndQM<T>::ClusterDataSet, that depends exclusively 
 *    on the KscEncodingAndQM<T>::ClusterDataPoint interfaces, is implemented in this base class to 
 *    cluster a set of data.
 *    
 *    Since the base class implements the sign based cluster membership encoding 
 *    scheme, the corresponding **Hamming distance** computation **is implemented** in 
 *    the KscEncodingAndQM<T>::ComputeDistance interface method here in the base class. 
 *    Having the Hamming distance computation implemented, **the corresponding cluster** 
 *    **assigment (based on the smallest Hamming distance) is also implemented** in 
 *    the KscEncodingAndQM<T>::ClusterDataPoint interfaces methods. 
 * - **model evaluation criterion**: 
 *    the special structure of the score variable space representation of the 
 *    clusters in the ideal case makes possible to measure the quality of a 
 *    model or a completed clustering. Different model quality measures are 
 *    available but some of them depends on the cluster membership encoding and 
 *    the cluster assigment. Therefore, the quality measure computation is linked 
 *    with the cluster membership encoding. The base class provides the 
 *    KscEncodingAndQM<T>::ComputeQualityMeasure interface method for implementing the model quality 
 *    evaluation algorithm. Each derived class implements their own algorithm.
 *
 * The following cluster membership encoding, assigment schemes are available 
 * with the corresponding model selection criterion: 
 * - KscEncodingAndQM_BLF: sign based encoding; assigment based on minimum Hamming 
 *    distance i.e. binary membership indicator; Balanced Line Fit (BLF) model selection 
 *    criterion that measures the within cluster collinearity (can be used for model selection \f$K \geq 2\f$ 
 * - KscEncodingAndQM_AMS: direction based encoding; assigment based on highest 
 *    membership indicator value that is based on cosine distance (measured from the 
 *    cluster ptototype directions); soft membership indicator; Average Membership Strength (AMS) 
 *    model selection criterion that measures the average within cluster collinearity 
 *    (can be used for model selection \f$K \geq 2\f$ 
 * - KscEncodingAndQM_BAS: similart to the above but special for sparse KSC model 
 *    obtained by using the reduced set method; direction based encoding; 
 *    assigment based on the smallest Euclidean distance (measured from the 
 *    cluster ptototype directions); Balanced Angular Similarity (BAS) model 
 *    selection that penalizes KSC models yielding more data near the decision boundaries;
 *    (can be used for model selection \f$K > 2\f$.
 *
 * @author M. Novak
 * @date   February 2020
 */



// T is the data type used for the computation (double or float) 
template <typename T>
class KscEncodingAndQM {

public:
  
  /**
   * @brief The only available constructor.
   *
   * @param[in] qmt the cluster membership encoding and quality measure type 
   *            - KscQMType::kBLF  sign based encoding and BLF quality measure (see more at KscEncodingAndQM_BLF)
   *            - KscQMType::kAMS  direction based (score data + cosine distance) encoding and AMS quality measure (see more at KscEncodingAndQM_AMS)
   *            - KscQMType::kBAS  direction based (reduced set coeffitient data and Euclidean distance) encoding and quality measure (see more at KscEncodingAndQM_BAS)
   */
  KscEncodingAndQM(KscQMType qmt, const std::string& name) 
  : fKscQMType(qmt), fKscQMTypeName(name), fNumClusters(2), fOutlierThreshold(0), fEtaBalance(0.) { }
  
  /**@brief Destructor*/
  virtual ~KscEncodingAndQM() {}
  
  /**
   * @brief Public method to obtain the cluster membership encoding and quality 
   *        measure type. 
   * @return The cluster membership encoding and quality measure type.
   */
  KscQMType GetQualityMeasureType() const { return fKscQMType; }
  
  /**
   * @brief Public method to obtain the name of the cluster membership encoding 
   *        and quality measure type. */
  const std::string& GetName() const { return fKscQMTypeName; }

  /**@brief Public method to set the balance term coefficient used in the quality 
    *       measure computation to determine the weight of the balance over the 
    *       collinearity terms (should be in [0,1]).
    */
  void      SetCoefEtaBalance(T eta) { fEtaBalance = eta; }  
  
  /**@brief Public method to obtain the balance term coefficient used in the quality measure.*/
  T         GetCoefEtaBalance()         const {return fEtaBalance; }
  
  /**@brief Public method to set the outlier threshold used in the quality 
    *       measure computation: clusters below this cardinality value are 
    *       considered to contain outliers and do not cotribute to the quality 
    *       measure value.
    *
    * @param[in] val The outlier threshold value; 
    */  
  void      SetOutlierThreshold(size_t val) { fOutlierThreshold = val; }
  
  /**@brief Public method to obtain outlier threshold value used in the quality measure.*/
  size_t    GetOutlierThreshold() const { return fOutlierThreshold; } 
  
  
  /**@brief Public method to obtain the KSC model quality measure value computed 
   *        and set when invoking the ComputeQualityMeasure interface method.*/
  T         GetTheQualityMeasureValue() const  {return fTheQualityMeasureValue; }
      
  /**
   *@brief Interface method to generate the code book (the cluster encoding). 
   *
   * In case of KSC, each cluster is represented by a special vector that depends 
   * on the selected cluster encoding scheme. These cluster representations are 
   * generated in this interface method based on a training data set.
   * Since the type of the cluster prototype vector depends on the encoding 
   * scheme, each derived class implements it own algorithm.
   *
   * Since all implemented cluster ecoding schemes make use of the **sign** 
   * (of the training score data or the corresoonding reduced set coefficient data) 
   * **based encoding** when generating their own encodings, this **is implemented** 
   * here **in this base class** method through the KscEncodingAndQM<T>::GenerateSignBasedCodeBook
   * method. Therefore, each derived class can invokde this base class implementation 
   * to generate the sign based encoding. The method also sets the 
   * KscEncodingAndQM<T>::fNumClusters member variable to the number of desired clusters.
   *   
   * @param[in] encodeMatrix reference to **either the training set score data** 
   *    matrix **or to the corresponding reduced set coefficient** matrix.
   * @param[in] numClusters number of required clusters
   * @param[in] isScoreVarBased flag to indicate if the encoding is score 
   *    variable based (BLF,AMS) or not (BAS) i.e. the encodeMatrix is a
   *    reference to the training set score data or to the corresponding 
   *    reduced set coeffitients.
   */
  virtual void  GenerateCodeBook(const Matrix<T>& encodeMatrix, size_t numClusters) {
    GenerateSignBasedCodeBook(encodeMatrix, numClusters);
  }
  
  /**
   * @brief Public method to obtain a reference to the sign based code book 
   *        generated by calling the GenerateCodeBook method.
   * @return A reference to the sign bbased code book: a vector of fNumClusters, 
   *         fNumClusters-1 dimensional boolean vectors encoding the clusters 
   *         (true = +, flase = -).
   */
  const std::vector<std::vector<bool> >& GetTheSignBasedCodeBook() { return fTheSignBasedCodeBook; }

  /**
   * @brief Public method to get the number of required clusters set when calling 
   *       the the GenerateCodeBook method.
   */
  size_t   GetNumClusters() { return fNumClusters; }




  /**
   *@brief One of the 3 different interface methods to cluster a data point given 
   * its representation in the score variable space. 
   *
   * Any data point can be clusterred given by its representation in the score 
   * varibale space after generating the cluster membership encoding by calling 
   * the GenerateCodeBook interface method. 
   * The 3 different methods provided to perform this cluster membership assigment
   * differes regarding the information filled in. Only the index of the cluster, 
   * to which this data is assigned, is returned in this case. The other two method 
   * can be useful in case of **soft cluster membership encodings** such as the 
   * AMS or BAS since they provide the infomation on how certain the given 
   * assigment is: either the assigment to the selected cluster or to all the 
   * possible clusters.
   *
   * Since this base class implements the sign based code book generation, this 
   * cluster assigment method is implemented: assigning the data to the cluster 
   * that gives the minimum Hamming distance when that is computed between the 
   * binarised score data and the corresponding sign based code word. Note, that
   * it also indicates that the Hamming distance computation is implemented in 
   * the ComputeDistance interface method is this base class.
   * Therefore, all ingerdients that the BLF encoding and assigment requires is 
   * implemented in theis base class. 
   *
   * @param[in] aScoreData  pointer to a memory where a fNumClusters-1 dimensional 
   *     score data is stored (in memory continous way) that is to be clusterred.
   * @param[in] dim  dimension of the score data (must be fNumClusters-1).
   * @return Index of the cluster the input data is assigned to. 
   */
  virtual size_t  ClusterDataPoint(const T* aScoreData, size_t dim);

  /**
   * @brief Interface method to cluster a data point given its representation 
   *        in the score variable space. 
   *
   * Same as above with the difference that a soft cluster membership indicator 
   * i.e. a value representing how ceratin is the given assigment, will be given 
   * in addition to the index of the cluster to which the data is assigned.
   * 
   * Note, this additional information on the strength of the cluster membership 
   * will be set to 1 in case of hard membership indicators such as the one 
   * when using the pure sign based encoding and the Hamming distance i.e. in 
   * case of BLF.
   * @param[in] aScoreData  pointer to a memory where a fNumClusters-1 dimensional 
   *     score data is stored (in memory continous way) that is to be clusterred.
   * @param[in] dim  dimension of the score data (must be fNumClusters-1).
   * @param[in] aMembership reference to fill the cluster membership strength. 
   * @return Index of the cluster the input data is assigned to. 
   */
  virtual size_t  ClusterDataPoint(const T* aScoreData, size_t dim, T& aMemberships);
  
  /**
   * @brief Interface method to cluster a data point given its representation 
   *        in the score variable space. 
   *
   * Same as above with the difference that a soft cluster membership indicator 
   * i.e. a value representing how ceratin is the given assigment, will be given 
   * not only for the assigned cluster but all possible assigments. This infomation 
   * is available only in case of AMS. The strength of the cluster membership 
   * indicator will be set only for the cluster to which the data is assigned to 
   * in all other cases (to 1 in case of BLF as hard assigment and to the soft 
   * value in case of BAS).
   *
   * @param[in] aScoreData  pointer to a memory where a fNumClusters-1 dimensional 
   *     score data is stored (in memory continous way) that is to be clusterred.
   * @param[in] dim  dimension of the score data (must be fNumClusters-1).
   * @param[in] aMembership a pointer to a continuos memory where the 
   *     cluster membership strength to be filled.
   * @return Index of the cluster the input data is assigned to. 
   */
  virtual size_t  ClusterDataPoint(const T* aScoreData, size_t dim, T* aMemberships);
  
  
    
  /**
   * @brief Method to cluster a set of data points given their representation in 
   *        the score varibale space.
   * 
   * @param[in] aScoreMatrix refrence to the matrix that contains the representation 
   *   of the data points to be clusterred in the score variable space i.e. the 
   *   score data matrix. 
   * @param[in,out] aCMMatrix reference to a matrix where the assigned cluster 
   *   index as well as possible membership infomation will be filled. The fisrt 
   *   column of the matrix will contain the assigned cluster indices: the \f$i\f$-th 
   *   row will contan the index of teh cluster to which the data given in the 
   *   \f$i\f$-th column of the sore data matrix is assigned to. Further column(s)
   *   along this row might contain information on the strength of the assigment:
   *   - \f$\texttt{flag}\f$=1 (BAS,AMS): the columns with index = 1 will contain
   *       the strength of the assignment to the selected cluster (which index 
   *       is given in col=0).
   *   - \f$\texttt{flag}\f$=2 (AMS): the columns with index \f$= [1,\dots,K]\f$ 
   *       will contain the strength of assigning the given data to the cluster 
   *       with index=k.
   * @param[in] flag a flag to indicate if:
   *   - \f$\texttt{flag}\f$=0: simple cluster membership assigment is required.
   *   - \f$\texttt{flag}\f$=1: the strength of the assigment is aslo required.
   *   - \f$\texttt{flag}\f$=2: the strength to assigning to all clusters are requied.
   *
   * This method is implemented in this base class and used by all the derived 
   * classes since the only functionality required must be implemented in the  
   * ClusterDataPoint interface methods. 
   *
   * @note score data are assumed to be stored in the input \f$\texttt{aScoreMatrix}\f$
   * as columns i.e. the aScoreMatrix is assumed to be \f$\in \mathbb{R}^{(K-1)xN}\f$
   * where \f$ K\f$ is the number of clusters and \f$ N\f$ is the number of data 
   * point stored in the matrix. Moreover, the \f$\texttt{aCMMatrix}\f$ is assumed 
   * to be a row major matrix with \f$ N\f$ rows and 1,2 or K+1 number of columns 
   * depending on the values of the \f$\texttt{flag}=0,1\f$ or \f$2\f$.
   */
  void ClusterDataSet(Matrix<T>& aScoreMatrix, Matrix<T, false>& aCMMatrix, size_t flag=0);


  /**
   *@brief Intrface method to compute the quality measure for model selection.
   *
   * Depending on th selected cluster membership encoding (BLF, AMS, BAS) one 
   * can compute the corresponding quality measure i.e. a measure on how far the 
   * model is from the ideal situation. Since the type of the quality measure
   * depends on the selected encoding, each derived class needs to implement its 
   * own version of this interface method. 
   *
   * @param[in] aCMMatrix reference to the matrix that stores the assigned 
   *   cluster indices and in some cases (AMS, BAS) further membership infomation. 
   * @param[in] aScoreMatrix the score data matrix used in to obtain the cluster 
   *   assigment. Used only in case of the BLF quality measure computation.
   * @param[in] aScoreMatrix the score data matrix used in to obtain the cluster 
   *   assigment. **Used only in case of the BLF** quality measure computation.
   * @param[in] theSecondVarForBLFM the second variable matrix (vector) for BLF 
   *   (**used only in case of BLF when the number of desired clusters = 2**)
   * @param Returns with the computed quality easure that indicates how far the 
   *   model is from the ideal case. This can be used for model selection when 
   *   combined with a grid search.
   */  
  virtual T ComputeQualityMeasure(Matrix<T, false>& aCMMatrix, const Matrix<T>* aScoreMatrix=nullptr, const Matrix<T>* theSecondVarForBLFM=nullptr) {
    (void)aCMMatrix;
    (void)aScoreMatrix;
    (void)theSecondVarForBLFM;
    return 0.;
  }


  void DoAll(Matrix<T>* encodeMatrix, size_t numClusters, Matrix<T>* aScoreMatrix, Matrix<T, false>* aCMMatrix) {
    GenerateCodeBook(*encodeMatrix, numClusters);
    ClusterDataSet(*aScoreMatrix, *aCMMatrix, 1);
    if (GetQualityMeasureType()==KscQMType::kBLF) {
      ComputeQualityMeasure(*aCMMatrix, aScoreMatrix);
    } else {
      ComputeQualityMeasure(*aCMMatrix);      
    }    
  }


  // This base class implements Hamming distance computation between the sign 
  // coding of the input score data point and the sign based code word that 
  // belongs to the given pCluster
  /**
   * @brief Interface method to compute the distance of a data point from a cluster.
   *  
   * The data point is given with its score variable space representation. The 
   * method needs to implement the computation of the (encoding dependent) distance 
   * of the input data measured from the cluster prototype vector generated during
   * the GenerateCodeBook method. Since the distance computation depends on the 
   * cluster membership encoding scheme, each derived class implements its own 
   * distance computation.
   *
   * Since this base class implements the sign based cluster membership encoding 
   * in the GenerateCodeBook method, the corresponding distance, the Hamming  
   * distance computation is implemented in this base class methods.
   *
   * @param[in] aScoreData pointer to a memeory space that stores the score variable 
   *    space representation of a data point in a memory continuos way.
   * @param[in] pCluster index of the cluster from which the distance needs to be computed.
   * @param[in] dim dimension of the input score data (must be fNumClusters-1).
   * @return The distance of the input data measured from the cluster specified 
   *         with its index. The Hamming distance computation is implemented in 
   *         this base class method.
   */
  virtual T ComputeDistance(const T* aScoreData, size_t pCluster, size_t dim) {
    //assert (dim==fNumClusters-1 && "\n Different vector size in the Hamming distance calculation! \n");  
    _unused(dim);
    size_t distHamming = 0;
    for (size_t ic=0; ic<fNumClusters-1; ++ic) {
      distHamming += ((aScoreData[ic] > 0.)!=fTheSignBasedCodeBook[pCluster][ic]);
    }
    return distHamming;
  }

  // not implemented (vector of boolean is not necessarily continous)
  virtual T ComputeDistance(const T* av, const T* bv, size_t dim) {
    _unused(av);
    _unused(bv);
    _unused(dim);
    return 0.;
  }


protected:
  // aData is assumed to store the data(vectors) to be clustered as cols in a 
  // memory continous (i.e. col-major) way. The ComputeDistance interface method 
  // is used to compute the distance between the data vectors and the prototypes 
  void KMeans(std::vector< std::vector<T> >& thePrototypeVect, const Matrix<T>& aDataM, bool doNormalise);

private:
  
  /**
   * @brief Private method to generate the sing based cluster encoding.
   *
   * Generates the sign based code book: the top K, K-1 sign coding determined 
   * from the K-1 dimensional entires of the input matrix.
   * It also sets the fNumClusters member variable to the number of clusters.
   * The encode matrix is **either the score variable matrix** \f$ \in \mathbb{R}^{(K-1)\times N_{tr}} \f$ 
   * **or the reduced set coefficient matrix** \f$ \in \mathbb{R}^{R\times (K-1)} \f$ 
   * that is indicated by the \f$\texttt{isScoreVarBased}\f$ flag.
   *
   * After binarising the \f$ K-1\f$ dimensional input (**either score or reduced**
   * **set coeffitients**) data by taking the sign of the components, the \f$ K\f$ 
   * most frequent \f$ \textbf{cw}^{(k)} \in \{-1,+1\}^{K-1}, k=1,\dots,K \f$ sign
   * based code words are determined. These \f$ \textbf{cw}^{(k)}, k=1,\dots,K \f$ code words 
   * are stored then in the KscEncodingAndQM<T>::fTheSignBasedCodeBook 
   * member as \f$ K, K-1\f$ dimensional boolean vectors (\f$ \texttt{true}=+; \texttt{false}=-\f$)
   * and a reference to this member can be obtained by using the KscEncodingAndQM<T>::GetTheSignBasedCodeBook
   * public method.
   */
  void  GenerateSignBasedCodeBook(const Matrix<T>& encodeMatrix, size_t numClusters);
  

protected:

  /**@brief The value of the computed KSC model quality measure (will be set in
   *        the ComputeQualityMeasure method.
   */
  T                                fTheQualityMeasureValue; 


private:
  /**@brief Cluster membership encoding and quality measure type*/
  KscQMType                        fKscQMType;
  /**@brief Name of the cluster membership encoding and quality measure type*/
  const std::string                fKscQMTypeName;
  /**@brief Number of required cluster*/
  size_t                           fNumClusters;
  /**@brief The collection of the fNumClusters the sign based code words.*/
  std::vector< std::vector<bool> > fTheSignBasedCodeBook;
  /**@brief A minimum required cardinality below which clusters are considered to 
    *       to contain outliers and do not contribute to the quality measure.*/
  size_t                           fOutlierThreshold;
  /**@brief The weight given to the balance term over the collinearity in the quality measure.*/
  T                                fEtaBalance;
  
};

// --------------------------------------------------------------------------- //
//  Form the book of the sign based cluster encoding words (each reperesenting 
//  one cluster) by determining the K, most frequent K-1 dimensional sing 
//  codings given either the score variable matrix ((K-1) x N_{tr}) or the 
//  reduced set coeffitients (R x (K-1)). The former is used in case of BLF and 
//  AMS while the later is used in case of BAF and must be indicated by the 
//  isScoreVarBased flag.
template <typename T>
void  KscEncodingAndQM<T>::GenerateSignBasedCodeBook(const Matrix<T>& encodeMatrix, size_t numClusters) {
  const size_t theBookPreSize = 2*numClusters;
  const size_t theNumTrData   = encodeMatrix.GetNumCols();
  const size_t theNumClusters = numClusters;                  
  const size_t theNumClustersMone = numClusters-1;
  // set the number of clusters member variable
  fNumClusters = theNumClusters; 
  
//  std::cerr<< " =====KK = " << numClusters <<" " << fNumClusters <<"   R = " << encodeMatrix.GetNumRows() << "  C = " << encodeMatrix.GetNumCols()<< std::endl;
  
  //
  // prepare containers
  size_t   curNumOfCodeWords = 0;
  fTheSignBasedCodeBook.clear();
  fTheSignBasedCodeBook.resize(theBookPreSize, std::vector<bool> (theNumClustersMone) );
  std::vector<size_t> theCodeWordFrequencies(theBookPreSize, 0);
  //
  // 1. generate the top K sign based code words from the 
  //   - isScoreVarBased = true  (BLF, AMS): N_tr score points
  //   - isScoreVarBased = false (BAS)     : R reduced set coefficients
  // insert the first code word  
  for (size_t is=0; is<theNumClustersMone; ++is) { 
    fTheSignBasedCodeBook[0][is] = (encodeMatrix.GetElem(is, 0) > 0.);
  }
  theCodeWordFrequencies[0] = 1;
  curNumOfCodeWords         = 1;
  // check all remaining code words
  std::vector<bool> temp(theNumClustersMone); 
  for (size_t idat=1; idat<theNumTrData; ++idat) {
    for (size_t is=0; is<theNumClustersMone; ++is) { 
      temp[is] = (encodeMatrix.GetElem(is, idat) > 0.);
    }
    // loop over the current code words
    for (size_t ib=0; ib<curNumOfCodeWords; ++ib) {
      size_t is = 0;
      for (; is<theNumClustersMone && temp[is]==fTheSignBasedCodeBook[ib][is]; ++is) {/**/}
      if (is==theNumClustersMone) {         // not a new code word
        ++theCodeWordFrequencies[ib];
        break;
      } else if (ib==curNumOfCodeWords-1) {    // new code word -> needs to be added
        // check if the book needs to be resized
        if (fTheSignBasedCodeBook.size()-1==curNumOfCodeWords) {
          const size_t ss = fTheSignBasedCodeBook.size()+theBookPreSize;
          fTheSignBasedCodeBook.resize(ss, std::vector<bool> (theNumClusters-1,false) ); 
          theCodeWordFrequencies.resize(ss, 0);
        }
        // add the new code word and set its frequency
        for (is=0; is<theNumClustersMone; ++is) { 
          fTheSignBasedCodeBook[curNumOfCodeWords][is] = temp[is];
        }
        theCodeWordFrequencies[curNumOfCodeWords] = 1;
        ++curNumOfCodeWords; 
        break;
      }
    }
  }
  // order the code words up to numClusters and erase all the tails
  size_t sumCardinalities = 0;
  for (size_t i=0; i<theNumClusters; ++i) {
    for (size_t j=i; j<curNumOfCodeWords; ++j) {
      if (theCodeWordFrequencies[i] < theCodeWordFrequencies[j]) {
        const size_t idum = theCodeWordFrequencies[i];
        theCodeWordFrequencies[i] = theCodeWordFrequencies[j];
        theCodeWordFrequencies[j] = idum;
        for (size_t is=0; is<theNumClustersMone; ++is) {
          const bool bdum = fTheSignBasedCodeBook[i][is];
          fTheSignBasedCodeBook[i][is] = fTheSignBasedCodeBook[j][is];
          fTheSignBasedCodeBook[j][is] = bdum;
        }
      }
    }
    sumCardinalities += theCodeWordFrequencies[i];
  }
  // erase the tail of the code book
  fTheSignBasedCodeBook.erase(fTheSignBasedCodeBook.begin()+theNumClusters, fTheSignBasedCodeBook.end());
  theCodeWordFrequencies.erase(theCodeWordFrequencies.begin()+theNumClusters, theCodeWordFrequencies.end());  
 //
 // std::cerr << "  Sum cardinalities = " << sumCardinalities << std::endl;
 // for (size_t i=0; i<theNumClusters; ++i) {
 //   std::cerr<< " sbv [ " << i << " ] = { ";
 //   for (size_t j=0; j<theNumClusters-1; ++j) {
 //     std::cerr<< fTheSignBasedCodeBook[i][j] << " ";
 //   }
 //   std::cerr<< "} " << std::endl;
 // }
}

// --------------------------------------------------------------------------- //
// cluster assigment implementation based on the sign based encoding and minimum 
// Hamming distance computation
template <typename T>
size_t  KscEncodingAndQM<T>::ClusterDataPoint(const T* aScoreData, size_t /*dim*/) {
  //assert (dim==fNumClusters-1 && "\n Different vector size in the cluster assigment! \n");
  size_t iCluster = 0;
  size_t minHammd = fNumClusters;
  for (size_t icl=0; icl<fNumClusters; ++icl) {
    const size_t hammd = ComputeDistance(aScoreData, icl, fNumClusters-1);
    if (hammd<minHammd) {
      minHammd = hammd;
      iCluster = icl;
    }
  }
  return iCluster; 
}

// sign based encoding plus min Hamming distance gives hard assigment i.e 0 or 1.
template <typename T>
size_t  KscEncodingAndQM<T>::ClusterDataPoint(const T* aScoreData, size_t /*dim*/, T& aMemberships) {
  aMemberships = 1.;
  return ClusterDataPoint(aScoreData, fNumClusters-1);
}

template <typename T>
size_t  KscEncodingAndQM<T>::ClusterDataPoint(const T* aScoreData, size_t /*dim*/, T* aMemberships) {
  size_t iCluster = ClusterDataPoint(aScoreData, fNumClusters-1);
  aMemberships[iCluster] = 1.; // all othes left to be zero
  return iCluster;
}



// --------------------------------------------------------------------------- //
template <typename T>
void  KscEncodingAndQM<T>::ClusterDataSet(Matrix<T>& aScoreMatrix, Matrix<T, false>& aCMMatrix, size_t flag) {
  const size_t numDataPoint = aScoreMatrix.GetNumCols(); // number of score points
  //const size_t dimDataPoint = aScoreMatrix.GetNumRows(); // K-1 <- dimension of score points
  //const size_t numClusters  = this->GetNumClusters();
  //
  if (flag==0) {
    for (size_t ip=0; ip<numDataPoint; ++ip) {
       const T* dum = aScoreMatrix.GetPtrToBlock(ip);
       size_t iCluster = ClusterDataPoint(dum, fNumClusters-1);
       aCMMatrix.SetElem(ip, 0, static_cast<T>(iCluster));
    }
  } else if (flag==1) {
    T cm = 0.;
    for (size_t ip=0; ip<numDataPoint; ++ip) {
       const T* dum = aScoreMatrix.GetPtrToBlock(ip);
       size_t iCluster = ClusterDataPoint(dum, fNumClusters-1, cm);
       aCMMatrix.SetElem(ip, 0, static_cast<T>(iCluster));
       aCMMatrix.SetElem(ip, 1, cm);
     }     
  } else {
    // note: each row of the aCMMatrix is assumed to be set to 0.
    for (size_t ip=0; ip<numDataPoint; ++ip) {
       T* cmRow = aCMMatrix.GetPtrToBlock(ip)+1;
       const T* dum = aScoreMatrix.GetPtrToBlock(ip);
       size_t iCluster = ClusterDataPoint(dum, fNumClusters-1, cmRow);
       aCMMatrix.SetElem(ip, 0, static_cast<T>(iCluster));
     }     
  }
}



// aDataM is assumed to store the data(vectors) to cluster in its cols in memory continus way
template <typename T>
void  KscEncodingAndQM<T>::KMeans(std::vector< std::vector<T> >& thePrototypeVect, const Matrix<T>& aDataM, bool doNormalise) {
   const size_t numDataPoint = aDataM.GetNumCols();
   const size_t numClusters  = this->GetNumClusters();
   const size_t maxItr       = 100;
//   aDataM.WriteToFile("xxx.dat", true);
   std::vector<size_t>  theCardinalities(numClusters, 0);
   std::vector<size_t>  theAssigment(numDataPoint, numClusters+1);
   std::vector< std::vector<T> > theNewPrototypeVect(thePrototypeVect.size(), std::vector<T> (numClusters-1));
   
   size_t itr = 0;
   while(itr<maxItr) {
     // reset cardinalities (used for mean computation) and counter of changes 
     // in the assigments
     std::fill(theCardinalities.begin(), theCardinalities.end(), 1);
     size_t numModified = 0;
     // assigne data to clusters
     for (size_t id=0; id<numDataPoint; ++id) {
       const T* aData  = aDataM.GetPtrToBlock(id);
       size_t iCluster = 0;
       T      minDist  = ComputeDistance(aData, thePrototypeVect[0].data(), numClusters-1);
       for (size_t ic=1; ic<numClusters; ++ic) {
         const T dist = ComputeDistance(aData, thePrototypeVect[ic].data(), numClusters-1);
         if (dist<minDist) {
           minDist  = dist;
           iCluster =   ic; 
         }
       }
       if (theCardinalities[iCluster]==0) {
         for (size_t ic=0; ic<numClusters-1; ++ic) {
           theNewPrototypeVect[iCluster][ic] = aData[ic];
         }
       } else {
         for (size_t ic=0; ic<numClusters-1; ++ic) {
           theNewPrototypeVect[iCluster][ic] += aData[ic];
         }         
       }
       if (theAssigment[id] != iCluster) {
         ++numModified;
       }
       theAssigment[id] = iCluster;
       ++theCardinalities[iCluster];
     }
     // compute new mean into the original prototype
//     std::cerr << "  cardianlities:  itr = " << itr << " >> ";
//std::cerr<< "\n*********** "<< std::endl;
     for (size_t ic=0; ic<numClusters; ++ic) {
       assert ( theCardinalities[ic] > 0 && " Cardinality of a cluster is zero => should never happen");
       const T invCardinality = 1./static_cast<T>(theCardinalities[ic]);
       T norm  = 0.;
       for (size_t is=0; is<numClusters-1; ++is) {
         const T val = theNewPrototypeVect[ic][is]*invCardinality;
         thePrototypeVect[ic][is] = val;
//         std::cerr<< thePrototypeVect[ic][is] << " ";
         norm += val*val;
       }
       if (doNormalise) {
         const T invNorm = 1./sqrt(norm);
         for (size_t is=0; is<numClusters-1; ++is) {
           thePrototypeVect[ic][is] *= invNorm;
         }
       }
//       std::cerr <<  theCardinalities[ic] << "  ";        
     }
//     std::cerr<<std::endl;
     ++itr;         
     if (numModified==0) {
       break;
     }
   }
}




#endif