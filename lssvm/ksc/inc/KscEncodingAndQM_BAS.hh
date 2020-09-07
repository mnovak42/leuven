
#ifndef KSCENCODINGANDQM_BAS_HH
#define KSCENCODINGANDQM_BAS_HH

#include "KscEncodingAndQM.hh"

/**
 * @class  KscEncodingAndQM_BAS
 * @brief  Angual similarity based KSC cluster membership encoding and Balanced
 *         Angual Similarity clastering quality measure.
 *
 * Follows the base KscEncodingAndQM class interfaces to implement a cluster 
 * membership encoding and the corresponding model evaluation criterion. 
 * Similarity to the KscEncodingAndQM_AMS, the encoding is based on the 
 * **collinearity of the score space representation of the training data that 
 * belong to the same cluster**. However, **the cluster prototype directions** are 
 * **determined based on the reduced set coefficients instead of** using **the score**
 * **data**. Therefore, this encoding **can only be used when the reduced set method 
 * is utilised to obtain a sparse KSC model**.
 * 
 * After the partition of the reduced set coeffitients data into the desired number
 * of clustres by using the sign based cluster membership encoding, 
 * a **mean reduced set coefficient vector** is comuted **for each clusters** that 
 * will be used **as cluster prototype directions** (see more at KscEncodingAndQM_BAS<T>::GenerateCodeBook 
 * interface method implementation). 
 *
 * In contrast to the binary, sign and Hamming distances based cluster indicator 
 * implemented in the base class, a **soft cluster membership indicator** is 
 * introduced (see more at the KscEncodingAndQM_BAS<T>::ClusterDataPoint interface 
 * method implementation) **by computing the Euclidean distance** of any normalised 
 * score data **from the cluster prototype directions** (see more at the 
 * KscEncodingAndQM_BAS<T>::ComputeDistance 
 * interface method implementation). The data is assigned to the cluster yielding 
 * the smallest distance. 
 *
 * This cluster membership encoding comes with **Balanced Angular Similarity** 
 * **model evaluation critarion** (see more at the 
 * KscEncodingAndQM_BAS<T>::ComputeQualityMeasure interface method implemntation).
 * 
 * @author M. Novak
 * @date   February 2020
 */


template <typename T>
class KscEncodingAndQM_BAS : public KscEncodingAndQM<T> {

public:
  
  /**@brief Constructor */
  KscEncodingAndQM_BAS() :  KscEncodingAndQM<T>(kBAS, "BAS") {}
  
  /**
   * @brief Inerface method implementation to generate the cluster membership encoding.
   *
   * Similar to the KscEncodingAndQM_AMS<T>::GenerateCodeBook but specialised for 
   * the case when the **reduced set method** is used to obtain a **sparese KSC model**.
   *
   * Using a reduced set \f$ \mathcal{R} = \{\mathbf{x}_r\}_{r=1}^{R} \subset \{\mathbf{x}_i\}_{i=1}^{N_{tr}}, R \ll N_{tr} \f$
   * and the corresponding coeffitients \f$ \{ \mathbf{\zeta}_k \}_{k=1}^{K-1}, \mathbf{\zeta}_k \in \mathbb{R}^{R} \f$ 
   * such that \f$ \Omega_{\Psi\Psi}\mathbf{\zeta}^{(k)} =  \Omega_{\Psi\Phi}\mathbf{\beta}^{(k)}\f$
   * with \f$ \Omega_{\Psi\Psi}, \Omega_{\Psi\Phi} \f$ being the within reduced set
   * and reduced set - training set kernel matrices respectively and \f$\beta^{(k)}\f$
   * is the \f$k\f$-th (approximated)eigenvector of the \f$ D^{-1}M_D\Omega_{\Phi\Phi}\f$
   * matrix. The approximated score points, associated to the \f$\mathbf{x}_i\f$ 
   * input data  are computed as 
     \f[
        \tilde{z}^{*^{(k)}}_i = \sum_{r=1}^{R} \zeta_r^{(k)} K(\mathbf{x}_r, \mathbf{x}_i) + \tilde{b}^{(k)}, k=1,\dots,K-1
     \f]
   * It can be shown, that the \f$ R \ll N_{tr}, K-1 \f$ dimensional \f$ \mathbf{\tau}^*\f$ 
   * **reduced set coefficient points** (formed as the\f$ R \f$ rows of the reduced set 
   * coeffitient matrix), can be **used to find the cluster proptotype directions** 
   * **instead of the score data points** (since the reduced set coeffitients plays the 
   * role of the \f$ \mathbf{\beta}^{(k)}, k=1,\dots,K\f$ eigenvectors  when 
   * the reduced set method is used). 
   * 
   * Therefore, following a sign based encoding (using the base class KscEncodingAndQM::GenerateCodeBook method) 
   * and sign based clustering the \f$ R,  \mathbf{\tau}^*\f$ reduced set 
   * coefficient points into the \f$\mathcal{D}_1,\dots,\mathcal{D}_K\f$,
   * disjoint sets, the cluster prototype directions are computed for each clustrs as  
     \f[ 
        \mathbf{u}_k' =\frac{1}{|\mathcal{D}_k|} \sum_{i, \mathbf{x}_i \in \mathcal{D}_k} \mathbf{\tau}_i^* ,k=1,\dots,K
     \f] 
   * The final prototype directions are obtained with a normalisation as 
   * \f$ \mathbf{u}_k = \mathbf{u}_k'/\| \mathbf{u}_k'\|_2\f$.
   *
   * @param[in] encodeMatrix reference to the training set score data matrix. 
   * @param[in] number of required clusters.
   *
   */
  void    GenerateCodeBook(const Matrix<T>& aReducedSetCoefM, size_t numClusters) override;
  
  /** 
   * @brief Implementation of the interface method to cluster a data point given 
   *        its representation in the score variable space. 
   *
   * The **Euclidean distance** between the cluster prototype direction vectors 
   * \f$ \mathbf{u}_k \f$ (determined previously by calling the GenerateCodeBook method)
   * and the (approximated)score variable space representation \f$ \tilde{\mathbf{z}}_{t}^{*} \in \mathbb{R}^{(K-1)}\f$ 
   * of any input data point \f$ \mathbf{x}_t \in \mathbb{R}^{d} \f$ is computed as
     \f[
          \texttt{d}( \tilde{\mathbf{z}}^*_t, \mathbf{u}_k )
          = \left\|
             \frac{ \tilde{\mathbf{z}}^{*}_t }{ \| \tilde{\mathbf{z}}^{*} \|_2 } 
              - \mathbf{u}_k 
            \right\|_2
     \f]
     implemented in the KscEncodingAndQM_BAS<T>::ComputeDistance interface method.
   * The data is assigned to the cluster yielding the smallest distance. 
     \f[
         \mathbf{x}_t \to k = \arg \max_{k} (\texttt{cm}^{(k)}(\tilde{\mathbf{z}}_{t}^{*})), k=1,\dots,K
     \f]     
   *
   * The index of this cluster is returned by this method.
   *
   * @param[in] aScoreData  pointer to a memory where a fNumClusters-1 dimensional 
   *     score data is stored (in memory continous way) that is to be clusterred.
   * @param[dim] dim  dimension of the score data (must be fNumClusters-1).
   * @return Index of the cluster the input data is assigned to.    
   */
  size_t  ClusterDataPoint(const T* aScoreData, size_t dim) override;

  /**
   * @brief Interface methods to cluster a data point given its representation 
   *        in the score variable space and provide cluster membership strength. 
   *
   * Same as above with the difference that cluster membership strength indicator 
   * \f$ \texttt{cm}^{(k)} \f$ for the selected cluster \f$ k \in \{1,\dots,K\}\f$
   * will also be written into the address specified by the corresponding input 
   * argument. 
   *
   * This cluster membership strength is defined as 
     \f[
        \texttt{cm}^{(k)} (\tilde{\mathbf{z}}_{t}^{*}) = 1 -  \frac{\texttt{d}^{2}( \tilde{\mathbf{z}}^*_t, \mathbf{u}_{k_{2}} )}{\texttt{d}^{1}( \tilde{\mathbf{z}}^*_t, \mathbf{u}_{k_{1}} )}
     \f] 
   * where \f$_{k_{1}}\f$ and \f$_{k_{2}}\f$ indicates the clusters yielding the 
   * smallest \f$ \texttt{d}^{1}\f$ and the second smallest \f$\texttt{d}^{2}\f$ 
   * distances measured from the given \f$\tilde{\mathbf{z}}_{t}^{*}\f$ approximated 
   * score data point. Note, that **this membership strength penalizes** situation 
   * when **data points** are **located near a decision boundary**.
   *
   * Note, that the score space is single dimensional when the desired 
   * number of clusters is two \f$K=2 \to K-1=1\f$ and the corresponding cluster 
   * prototype directions are \f$+1, -1\f$. Since the Euclidean distance of the 
   * normalised score data takes the binary \f${0,+2}\f$ values in this case 
   * and the above cluster indicator strength becoms the constant 1. Therefore, 
   * this membership strength can be used only in \f$K>2\f$.  
   *
   * @param[in] aScoreData  pointer to a memory where a fNumClusters-1 dimensional 
   *     score data is stored (in memory continous way) that is to be clusterred.
   * @param[in] dim  dimension of the score data (must be fNumClusters-1).
   * @param[in/out] aMembership reference to fill the cluster membership strength. 
   * @return Index of the cluster the input data is assigned to.    
   */
  size_t  ClusterDataPoint(const T* aScoreData, size_t dim, T& aMembership) override;

// This is not implemented because we cannot assigne membership strengths to each clusters.
//  size_t  ClusterDataPoint(const T* aScoreData, size_t dim, T* aMemberships) override;

  /**
   *@brief Implementation of the intrface method to compute the BAS quality measure 
   *       for model selection.
   *
   * The Balanced Angular Similarity(BAS) model selection criterion is impemented in 
   * this method that can be used to measure how far the clustering result of a 
   * data set, obtained by a given KSC model, is from the ideal case. This 
   * information can be used for model selection.
   * 
   * The Angular Similarity (AS) is defined as 
     \f[
        \texttt{AS} = \frac{1}{K}\sum_{k=1}^{K} \frac{1}{|\mathcal{A}_k|}\sum_{i\in \mathcal{A}_k} \texttt{cm}^{(k)}_i
     \f]
   * where the input data set is partitioned into the \f$\mathcal{A}_1,\dots, \mathcal{A}_K \f$
   * clusters and \f$ \texttt{cm}^{(k)}_i\f$ (computed in the KscEncodingAndQM_BAS<T>::ClusterDataPoint methods)
   * is the value of the cluster membership strength for the \f$ i\f$-th data point assigned to 
   * the \f$ k\f$-th cluster.
   *
   * Beyond this AS value, that measures the average within cluster collinearity 
   * of the corresponding score variables, a second term is introduced as 
   * \f$ \texttt{BL} = \texttt{min}(|\mathcal{A}_k|)/\texttt{max}(|\mathcal{A}_k|), k=1,\dots,K \f$
   * to measure how balanced is the clustering result.
   *
   * The final quality measure is 
     \f[
         \texttt{BAS} = [1-\eta]\texttt{AS} + \eta\texttt{BL} 
     \f]
   * with \f$ \eta \in [0,1]\f$ as an input parameter determines the importance 
   * of the balance over the collinearity term.
   *
   * @param[in] aCMMatrix reference to the matrix that stores the assigned 
   *   cluster indices as its first (zeroth) column and the corresponding cluster 
   *   membership strength values as its second (first) column. This matrix was 
   *   filled by the KscEncodingAndQM_BAS<T>::ClusterDataSet method with the 
   *   corresponding \texttt{flag=1} value.
   * @param[in] aScoreMatrix not used in this method
   * @param[in] theSecondVarForBLFM not used in this method
   * @return The computed Balanced Angular Similarity KSC model evaluation criterion.
   */
  T       ComputeQualityMeasure(Matrix<T, false>& aCMMatrix, const Matrix<T>* aScoreMatrix=nullptr, const Matrix<T>* theSecondVarForBLFM=nullptr) override;

  /**
   * @brief Implementation of the interface method to compute the distance of 
   *        a data point from a given cluster.
   *
   * The **Euclidean distance** between the cluster prototype direction vectors 
   * \f$ \mathbf{u}_k \f$ (determined previously by calling the GenerateCodeBook method)
   * and the (approximated)score variable space representation \f$ \tilde{\mathbf{z}}_{t}^{*} \in \mathbb{R}^{(K-1)}\f$ 
   * of any input data point \f$ \mathbf{x}_t \in \mathbb{R}^{d} \f$ is computed as
     \f[
          \texttt{d}( \tilde{\mathbf{z}}^*_t, \mathbf{u}_k )
          = \left\|
             \frac{ \tilde{\mathbf{z}}^{*}_t }{ \| \tilde{\mathbf{z}}^{*} \|_2 } 
              - \mathbf{u}_k 
            \right\|_2
     \f]
   *
   * @param[in] aScoreData pointer to a memeory space that stores the (approximated) 
   *    score variable space representation of a data point in a memory continuos way.
   * @param[in] pCluster index of the cluster from which the distance needs to be computed (\f$ k\f$).
   * @param[in] dim dimension of the input score data (must be KscEncodingAndQM<T>::fNumClusters-1).   
   * @return The Euclidean distance computed between the normalised, approximated score 
   *    variable space representation of a data point and a given cluster specified by its index. 
   */  
  T       ComputeDistance(const T* aScoreData, size_t pCluster, size_t dim) override; 
  T       ComputeDistance(const T* av, const T* bv, size_t dim) override;

      
private:

  /**@brief The code book generated by calling the KscEncodingAndQM_BAS<T>::GenerateCodeBook method.
    *
    * The normalised \f$ \mathbf{u}_k, k=1,\dots,K\f$ vectors.
    */
  std::vector< std::vector<T> > fThePrototypeVectorBook;
  /**@brief A utility vector to store some intermediate infomation on the membershipt*/
  std::vector<T>                fTheMemberships;
};


template <typename T>
void  KscEncodingAndQM_BAS<T>::GenerateCodeBook(const Matrix<T>& aReducedSetCoefM, size_t numClusters) {
  const size_t theNumTrData       = aReducedSetCoefM.GetNumCols();
  const size_t theNumClusters     = numClusters;
  const size_t theNumClustersMone = numClusters-1;
  //
  // call the base class method to generate the top K sign codings
  KscEncodingAndQM<T>::GenerateCodeBook(aReducedSetCoefM, numClusters);
  const std::vector< std::vector<bool> >& theSignBasedCodeBook = this->GetTheSignBasedCodeBook();
  // set size of the cluster membership vector that will be used during the assigment
  fTheMemberships.resize(theNumClusters);
  //
  // compute the corresponding K, K-1 dimensional prototype vectors (directions 
  // i.e. normalised in case of K>2!)
  fThePrototypeVectorBook.clear();
  fThePrototypeVectorBook.resize(theNumClusters, std::vector<T>(theNumClustersMone, 0.));
  std::vector<size_t> theCodeWordFrequencies(numClusters, 0);
  for (size_t idat=0; idat<theNumTrData; ++idat) {
    // find out which code word this reducded set coef fits (if any)
    for (size_t icl=0; icl<theNumClusters; ++icl) {
      size_t is=0;
      for (; is<theNumClustersMone; ++is) {
        if ( theSignBasedCodeBook[icl][is] != (aReducedSetCoefM.GetElem(is, idat) > 0.) ) {
          break;
        } 
      }
      // this reduce set coef. point has the same signes as this icl-th code word:
      //  - add the coordinates to the corresponding prototype vector coordiantes
      if (is==theNumClustersMone) {
        for (size_t ic=0; ic<theNumClustersMone; ++ic) {
          fThePrototypeVectorBook[icl][ic] += aReducedSetCoefM.GetElem(ic, idat);
        }
        ++theCodeWordFrequencies[icl];
        break;
      }
      // do nothing otherwise: this score point do not match any of the top K 
      //                       code words based on the signs
    }
  }
  // compute the mean i.e. divide by the cardianlities/frequencies and normalise
  // the prototype vectors
  for (size_t icl=0; icl<theNumClusters; ++icl) {
    T norm  = 0.;
    T ifreq = 1./static_cast<T>(theCodeWordFrequencies[icl]);
    for (size_t is=0; is<theNumClustersMone; ++is) {
      T val = fThePrototypeVectorBook[icl][is]*ifreq; 
      fThePrototypeVectorBook[icl][is] = val;
      norm += val*val;
    }
    norm = 1./std::sqrt(norm);
    for (size_t is=0; is<theNumClustersMone; ++is) {
      fThePrototypeVectorBook[icl][is] *= norm;
    }
  }
  
  //
  // cluster all the R reduced set coefs points into K clusters using these 
  // prototypes as seeds 
/*  
  for (size_t j=0; j<numClusters; ++j) {
    std::cerr<<std::endl;
    for (size_t i=0; i<numClusters-1; ++i)
      std::cerr << fThePrototypeVectorBook[j][i] << " ";
  }
  std::cerr << " \n after " << std::endl;
*/  
  this->KMeans(fThePrototypeVectorBook, aReducedSetCoefM, true);
 /*
  for (size_t j=0; j<numClusters; ++j) {
    std::cerr<<std::endl;
    for (size_t i=0; i<numClusters-1; ++i)
      std::cerr << fThePrototypeVectorBook[j][i] << " ";
  }
  std::cerr << " ---- " << std::endl;
 */

  //
  // compute the final prototypes and normalised them
}



template <typename T>
size_t  KscEncodingAndQM_BAS<T>::ClusterDataPoint(const T* aScoreData, size_t /*dim*/) {
  const size_t numClusters  = this->GetNumClusters();
  // assert (numClusters==dim+1 && " Dimension of the score point is not K-1! ");
  size_t iCluster = 0;
  T      minDist  = 2.; 
  for (size_t icl=0; icl<numClusters; ++icl) {
    const T dist = ComputeDistance(aScoreData, icl, numClusters-1);
    if (minDist>dist) {
      minDist  = dist;
      iCluster = icl;
    }
  }
  return iCluster;
}


// aMemberships will contain the soft cluster mebership value for the assigned 
// cluster if K>2 and only the const 1 value in ase of K=2
template <typename T>
size_t  KscEncodingAndQM_BAS<T>::ClusterDataPoint(const T* aScoreData, size_t /*dim*/, T& aMembership) {
  const size_t numClusters  = this->GetNumClusters();
  // assert (numClusters==dim+1 && " Dimension of the score point is not K-1! ");
  size_t iCluster = 0;
  T      minDist  = 2.; 
  for (size_t icl=0; icl<numClusters; ++icl) {
    const T dist = ComputeDistance(aScoreData, icl, numClusters-1);
    if (minDist>dist) {
      minDist  = dist;
      iCluster = icl;
    }
    fTheMemberships[icl] = dist;
  }
  // order the distance upt to the second (first two minimal)
  for (size_t i=0; i<2; ++i) {
    for (size_t j=i; j<numClusters; ++j) {
      if (fTheMemberships[i]>fTheMemberships[j]) {
        T dum = fTheMemberships[i];
        fTheMemberships[i] = fTheMemberships[j];
        fTheMemberships[j] = dum;
      }
    }
  }
  //
  aMembership = 1. - minDist/fTheMemberships[1]; 
  return iCluster;
}

/*
// aMemberships will contain the soft cluster mebership values for the k=1,...,K
// clusters if K>2 and only the 1,0 or 0,1 in case of K=2.
template <typename T>
size_t  KscEncodingAndQM_BAS<T>::ClusterDataPoint(const T* aScoreData, size_t, T* aMemberships) {
  const size_t numClusters  = this->GetNumClusters();
  // assert (numClusters==dim+1 && " Dimension of the score point is not K-1! ");
  T prod    = 1.;
  for (size_t icl=0; icl<numClusters; ++icl) {
    const T dist = ComputeDistance(aScoreData, icl, numClusters-1);
    if (dist==0.) { 
      aMemberships[icl]  = 1.;
      return icl;
    }
    aMemberships[icl] = dist;
    prod             *= dist;
  }
  size_t iCluster = 0;
  T      maxCm    = 0.; // not divided by the sum_i prod/x_i these are just the prod/x_i
  T      sumCm    = 0.;
  for (size_t icl=0; icl<numClusters; ++icl) {
    aMemberships[icl] = prod/aMemberships[icl];
    sumCm     += aMemberships[icl];
    if (maxCm<aMemberships[icl]) {
      maxCm    =  aMemberships[icl];
      iCluster = icl;
    }
  }
  // for the cm_i, we should divide now maxCm by sum   
  for (size_t icl=0; icl<numClusters; ++icl) {
    aMemberships[icl] /= sumCm; // cm_i
  }
  //
  return iCluster;
}
*/


template <typename T>
T KscEncodingAndQM_BAS<T>::ComputeDistance(const T* av, const T* bv, size_t dim) {
  // norm of the score point (the prototype vectors are already normalised)
  T norm = 0.;
  T dum  = 0.;
// ========================================================================== //  
// d = 1-projection  => [0:2]
/*
  for (size_t is=0; is<dim; ++is) {
    const T ais = av[is];
    dum  += ais*bv[is];
    norm += ais*ais;
  }
  return 1. - dum/sqrt(norm);
*/
// ========================================================================== //
// distance = ||sp-sc_i|| => [0:2]
// note: cluster prototypes sp are normalised and the score sc_i will be
// 
  // 1. normalise the score point vector
  for (size_t is=0; is<dim; ++is) {
    norm += av[is]*av[is];
  }
  norm = 1./std::sqrt(norm);
  // 2. compute distance: norm of the difference => [0:2]
  T norm1 = 0.;
  for (size_t is=0; is<dim; ++is) {
    const T ais = av[is]*norm;
    dum    = ais-bv[is];
    norm1 += dum*dum;
  }
  return std::sqrt(norm1);
}

// Computes the cosine distance between the given score point and the prototype 
// direction that corresponds to pCluster-th cluster.
// Note, that the prototyp directions are normalised while the score point is not.
// Also note, that in cae of K=2 the prototype directions are -1 and +1 so 
// cosine_distance = 1-0 => 0 or 1-(-1) => 2 therefore there is no soft membership
// in case of K=2!
template <typename T>
T KscEncodingAndQM_BAS<T>::ComputeDistance(const T* aScoreData, size_t pCluster, size_t /*dim*/) {
  const size_t numClusters  = this->GetNumClusters();
  return ComputeDistance(aScoreData, fThePrototypeVectorBook[pCluster].data(), numClusters-1);
/*  
  // norm of the score point (the prototype vectors are already normalised)
  T norm = 0.;
  T dum  = 0.;
// ========================================================================== //  
// NOTE: d = 1-projection  => [0:2]
//
//  for (size_t is=0; is<numClusters-1; ++is) {
//    const T score = aScoreData[is];
//    dum  += score*fThePrototypeVectorBook[pCluster][is];
//    norm += score*score;
//  }
//  return 1. - dum/sqrt(norm);

// ========================================================================== //
// NOTE: distance = ||sp-sc_i|| => [0:2]
// note: cluster prototypes sp are normalised and the score sc_i will be
// 
  // 1. normalise the score point vector
  for (size_t is=0; is<numClusters-1; ++is) {
    norm += aScoreData[is]*aScoreData[is];
  }
  norm = 1./sqrt(norm);
  // 2. compute distance: norm of the difference => [0:2]
  T norm1 = 0.;
  for (size_t is=0; is<numClusters-1; ++is) {
    const T score = aScoreData[is]*norm;
    dum    = score-fThePrototypeVectorBook[pCluster][is];
    norm1 += dum*dum;
  }
  return sqrt(norm1);
  */
}

 
template <typename T>
T KscEncodingAndQM_BAS<T>::ComputeQualityMeasure(Matrix<T, false>& aCMMatrix, const Matrix<T>* /*aScoreMatrix*/, const Matrix<T>* /*theSecondVarForBLFM*/) {
  const size_t numDataPoint = aCMMatrix.GetNumRows(); // number of clustered data ponts
  const size_t numCols      = aCMMatrix.GetNumCols(); 
  const size_t numClusters  = this->GetNumClusters();
  const size_t outlierThres = this->GetOutlierThreshold();
  const T      etaBalance   = this->GetCoefEtaBalance();
  //
  // Compute the angular similarity part
  T theAS = 0.;
  if (numCols!=2) {
    std::cerr << " *** WARNING in KscEncodingAndQM_BAS::ComputeQualityMeasure \n"
              << "       ClusterDataSet should have been called with flag=1   \n"
              << "       requesting to compute all membership strengths and   \n"
              << "       store in the (firth) cols of the resulted matrix.    \n"
              << "       Quality measure will be zero without these !         \n"
              << std::endl;
    return theAS;         
  }

  // collect cardinalities and sum of cluster membership strengths
  std::vector<size_t> theCardinalities(numClusters, 0);
  std::vector<T>      theSumCmPerClusters(numClusters, 0.);
  for (size_t idat=0; idat<numDataPoint; ++idat) {
    const size_t iCluster = static_cast<size_t>(aCMMatrix.GetElem(idat,0));
    ++theCardinalities[iCluster];
    theSumCmPerClusters[iCluster] += aCMMatrix.GetElem(idat,1);
//    theSumCmPerClusters[iCluster] += (numCols==2) ? aCMMatrix.GetElem(idat,1) : aCMMatrix.GetElem(idat,iCluster+1); 
  }
  for (size_t icl=0; icl<numClusters; ++icl) {
    theAS += (theCardinalities[icl]>outlierThres) ? theSumCmPerClusters[icl]/theCardinalities[icl] : 0.;
  }
  theAS /= numClusters;   
  //
  // Compute the Balance part: min(Cardinality)/max(Cardianlity)
  size_t minCardinality = numDataPoint;
  size_t maxCardinality = 0;
  for (size_t i=0; i<numClusters; ++i) {
    minCardinality = minCardinality>theCardinalities[i] ? theCardinalities[i] : minCardinality;
    maxCardinality = maxCardinality<theCardinalities[i] ? theCardinalities[i] : maxCardinality; 
  }
  const T theBalance = static_cast<T>(minCardinality)/static_cast<T>(maxCardinality);
  //
  // Compute the final BAS and return 
  const T theQMValue = (1.-etaBalance)*theAS + etaBalance*theBalance;
  this->fTheQualityMeasureValue = theQMValue;
  return theQMValue;    
}




#endif
