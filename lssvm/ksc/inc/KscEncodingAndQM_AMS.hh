
#ifndef KSCENCODINGANDQM_AMS_HH
#define KSCENCODINGANDQM_AMS_HH

#include "KscEncodingAndQM.hh"

/**
 * @class  KscEncodingAndQM_AMS
 * @brief  Angual similarity based KSC cluster membership encoding and Average 
 *         Membership Strength clastering quality measure.
 *
 * Follows the base KscEncodingAndQM class interfaces to implement a cluster
 * membership encoding based on the **collinearity of the score space 
 * representation of the training data that belong to the same cluster**. 
 * 
 * After the partition of the training set score data into the desired number of 
 * clustres by using the sign based cluster membership encoding, 
 * a **mean score vector** is comuted **for each clusters** that will be used 
 * **as cluster prototype** (see more at KscEncodingAndQM_AMS<T>::GenerateCodeBook 
 * interface method implementation). 
 *
 * In contrast to the binary, sign and Hamming distances based cluster indicator 
 * implemented in the base class, a **soft cluster membership indicator** is 
 * introduced (see more at the KscEncodingAndQM_AMS<T>::ClusterDataPoint interface 
 * method implementation) **by computing the cosine distance** of any score data 
 * **from the cluster prototypes** (see more at the KscEncodingAndQM_AMS<T>::ComputeDistance 
 * interface method implementation). The data is assigned to the cluster yielding 
 * the highest cluster membership indicator. 
 *
 * This cluster membership encoding and sof membership indicator comes with the 
 * so called **Average** cluster **Membership Strength** computation as the 
 * **model evaluation critarion** (see more at the 
 * KscEncodingAndQM_AMS<T>::ComputeQualityMeasure interface method implemntation).
 * 
 * @author M. Novak
 * @date   February 2020
 */


template <typename T>
class KscEncodingAndQM_AMS : public KscEncodingAndQM<T> {

public:
  
  /**@brief Constructor */
  KscEncodingAndQM_AMS() :  KscEncodingAndQM<T>(kAMS, "AMS") {}
  
  /**
   * @brief Inerface method implementation to generate the cluster membership encoding.
   *
   * The base class interface method implementation KscEncodingAndQM::GenerateCodeBook
   * **is called** first **to generate the sign based encoding** of the clusters using
   * the **training score data** provided as input argument. Then a cluster prototype, 
   * as the average score vector, is computed for each clusters as
     \f[ 
         \mathbf{s}_k =\frac{1}{|\mathcal{A}_k|} \sum_{i, \mathbf{z}_i^* \in \mathcal{A}_k} \mathbf{z}_{i}^{*} ,k=1,\dots,K
     \f]
   * where \f$ \{ \mathcal{A}_1, \dots,\mathcal{A}_K \} \f$ is the partiton of the 
   * training score data based on their signs. These cluster prototype score vectors
   * are normalised \f$ \mathbf{s}_k = \mathbf{s}_k/\|\mathbf{s}_k\|_2 \f$, unless \f$ K=2\f$ (when the Euclidean distance between these 
   * prototype and a given score point will be compute for the cluster assignment 
   * instead of the cosine distance).
   *
   * @param[in] encodeMatrix reference to the training set score data matrix. 
   * @param[in] number of required clusters.
   *
   */
  void    GenerateCodeBook(const Matrix<T>& aScoreVariableM, size_t numClusters) override;

  /** 
   * @brief Implementation of the interface method to cluster a data point given 
   *        its representation in the score variable space. 
   *
   * The **cosine distance** between the cluster prototype vectors \f$ \mathbf{s}_k \f$
   * (determined previously by calling the GenerateCodeBook method) and the score 
   * variable space representation \f$ \mathbf{z}_{t}^{*} \in \mathbb{R}^{(K-1)}\f$ 
   * of any input data point \f$ \mathbf{x}_t \in \mathbb{R}^{d} \f$ is computed as
     \f[
        \texttt{d}_{\texttt{cos}}(\mathbf{z}^*_t, \mathbf{s}_k) = 1-\frac{\mathbf{z}^{*^{T}}_t \mathbf{s}_k}{\|\mathbf{z}^*_t \|_2\|\mathbf{s}_k \|_2}
     \f]
     implemented in the KscEncodingAndQM_AMS<T>::ComputeDistance interface method.
   * Then the cluster membership indicator for this data 
     \f[
        \texttt{cm}^{(k)} (\mathbf{z}_{t}^{*}) = \frac{ \prod_{p\neq k}  
                                   \texttt{d}_{\texttt{cos}}(\mathbf{z}^*_t, \mathbf{s}_p) 
                                  }{ 
                                   \sum_{p=1}^{K} 
                                   \prod_{p\neq k}  \texttt{d}_{\texttt{cos}}(\mathbf{z}^*_t, \mathbf{s}_p) 
                                 }
     \f] 
   * is computed for all \f$ k=1,\dots,K\f$ clustres and the data is assigned 
   * to the cluster yielding the highest cluster membership indicator value
     \f[
         \mathbf{x}_t \to k = \arg \max_{k} (\texttt{cm}^{(k)}(\mathbf{z}_{t}^{*})), k=1,\dots,K
     \f]     
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
   *        in the score variable space and provide cluster membership indicator. 
   *
   * Same as above with the difference that soft cluster membership indicator 
   * \f$ \texttt{cm}^{(k)} \f$ for the selected cluster \f$ k \in \{1,\dots,K\}\f$
   * will also be written into the address specified by the corresponding input 
   * argument.
   *
   * @param[in] aScoreData  pointer to a memory where a fNumClusters-1 dimensional 
   *     score data is stored (in memory continous way) that is to be clusterred.
   * @param[in] dim  dimension of the score data (must be fNumClusters-1).
   * @param[in/out] aMembership reference to fill the cluster membership strength. 
   * @return Index of the cluster the input data is assigned to.    
   */
  size_t  ClusterDataPoint(const T* aScoreData, size_t dim, T& aMembership) override;

  /**
   * @brief Interface method to cluster a data point given its representation 
   *        in the score variable space and provide cluster membership indicator. 
   *
   * Same as above with the difference that the soft cluster membership indicator 
   * \f$ \texttt{cm}^{(k)} \f$ for all clusters \f$ k \in \{1,\dots,K\}\f$
   * will also be written into the address specified by the corresponding input 
   * argument.
   *
   * @param[in] aScoreData  pointer to a memory where a fNumClusters-1 dimensional 
   *     score data is stored (in memory continous way) that is to be clusterred.
   * @param[in] dim  dimension of the score data (must be fNumClusters-1).
   * @param[in/out] aMembership a pointer to a continuos memory where the 
   *     cluster membership strength to be filled.
   * @return Index of the cluster the input data is assigned to. 
   */
  size_t  ClusterDataPoint(const T* aScoreData, size_t dim, T* aMemberships) override;

  /**
   *@brief Implementation of the intrface method to compute the AMS quality measure 
   *       for model selection.
   *
   * The Average Membership Strength(AMS) is impemented in this method that can be 
   * used to measure how far the clustering result of a data set, obtained by a given KSC model, 
   * is from the ideal case. This information can be used for model selection.
   * 
   * The AMS is defined as 
     \f[
        \texttt{AMS} = \frac{1}{K}\sum_{k=1}^{K} \frac{1}{|\mathcal{A}_k|}\sum_{i\in \mathcal{A}_k} \texttt{cm}^{(k)}_i
     \f]
   * where the input data set is partitioned into the \f$\mathcal{A}_1,\dots, \mathcal{A}_K \f$
   * clusters and \f$ \texttt{cm}^{(k)}_i\f$ (computed in the KscEncodingAndQM_AMS<T>::ClusterDataPoint methods)
   * is the soft cluster membership indicator value for the \f$ i\f$-th data point
   * assigned to the \f$ k\f$-th cluster.
   *
   * Beyond this AMS value that measures the average within cluster collinearity 
   * of the corresponding score variables, a second term is introduced as 
   * \f$ \texttt{BL} = \texttt{min}(|\mathcal{A}_k|)/\texttt{max}(|\mathcal{A}_k|), k=1,\dots,K \f$
   * to measure how balanced is the clustering result.
   *
   * The final quality measure is 
     \f[
         [1-\eta]\texttt{AMS} + \eta\texttt{BL} 
     \f]
   * with \f$ \eta \in [0,1]\f$ as an input parameter determines the importance 
   * of the balance over the collinearity term.
   *
   * @param[in] aCMMatrix reference to the matrix that stores the assigned 
   *   cluster indices as its first (zeroth) column and the corresponding soft cluster 
   *   membership indicator values as its second (first) column. This matrix was 
   *   filled by the KscEncodingAndQM_AMS<T>::ClusterDataSet method with the 
   *   corresponding \texttt{flag>0} value.
   * @param[in] aScoreMatrix not used in this method
   * @param[in] theSecondVarForBLFM not used in this method
   * @return The computed balanced Average Membership Strength KSC model evaluation criterion.
   */
  T       ComputeQualityMeasure(Matrix<T, false>& aCMMatrix, const Matrix<T>* aScoreMatrix=nullptr, const Matrix<T>* theSecondVarForBLFM=nullptr) override;

  /**
   * @brief Implementation of the interface method to compute the distance of 
   *        a data point from a given cluster.
   *
   * The **cosine distance** between a cluster prototype vectors \f$ \mathbf{s}_k \f$
   * (determined previously by calling the KscEncodingAndQM_AMS<T>::GenerateCodeBook method) and the score 
   * variable space representation \f$ \mathbf{z}_{t}^{*} \in \mathbb{R}^{(K-1)}\f$ 
   * of an input data point \f$ \mathbf{x}_t \in \mathbb{R}^{d} \f$ is computed as
     \f[
        \texttt{d}_{\texttt{cos}}(\mathbf{z}^*_t, \mathbf{s}_k) = 1-\frac{\mathbf{z}^{*^{T}}_t \mathbf{s}_k}{\|\mathbf{z}^*_t \|_2\|\mathbf{s}_k \|_2}
     \f]
   *   
   * @param[in] aScoreData pointer to a memeory space that stores the score variable 
   *    space representation of a data point in a memory continuos way.
   * @param[in] pCluster index of the cluster from which the distance needs to be computed (\f$ k\f$).
   * @param[in] dim dimension of the input score data (must be KscEncodingAndQM<T>::fNumClusters-1).   
   * @return The cosine distance computed between the score variable space representation
   *    of a data point and a given cluster specified by its index. 
   */  
  T       ComputeDistance(const T* aScoreData, size_t pCluster, size_t dim) override;
  T       ComputeDistance(const T* av, const T* bv, size_t dim) override;
  
    
private:

  /**@brief The code book generated by calling the KscEncodingAndQM_AMS<T>::GenerateCodeBook method.
    *
    * The normalised (unless \f$ K=2\f$) \f$ \mathbf{s}_k, k=1,\dots,K\f$ vectors.
    */
  std::vector< std::vector<T> > fThePrototypeVectorBook;
  /**@brief A utility vector to store some intermediate infomation on the membershipt*/
  std::vector<T>                fTheMemberships;

    
};


//  Form the book of prototype vectors (each reperesenting one cluster) by 
//  clustering the N_tr, K-1 dimensional score points into K groups based on 
//  their signes and calculate the K, K-1 dimensional "mean vectors" 
//  \mathbf{s}_p \in \mathbb{R}^{K-1}, p=1,..,K, that correspond to the K 
//  cluster prototypes:
//   - 1. using the signe coding of the K-1 dimensional score points the K most  
//     frequent signe codings will be determined 
//   - 2. the corresponding, s_p mean prototype vectors will be computed and 
//     stored in fThePrototypeVectorBook (normalised unless K=2)
template <typename T>
void  KscEncodingAndQM_AMS<T>::GenerateCodeBook(const Matrix<T>& aScoreVariableM, size_t numClusters) {
  const size_t theNumTrData       = aScoreVariableM.GetNumCols();
  const size_t theNumClusters     = numClusters;
  const size_t theNumClustersMone = numClusters-1;
  
// aScoreVariableM.WriteToFile("xxx.dat", true);
  //
  // call the base class method to generate the top K sign codings
  KscEncodingAndQM<T>::GenerateCodeBook(aScoreVariableM, numClusters);
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
        if ( theSignBasedCodeBook[icl][is] != (aScoreVariableM.GetElem(is, idat) > 0.) ) {
          break;
        } 
      }
      // this score point has the same signes as this icl-th code word:
      //  - add the coordinates to the corresponding prototype vector coordiantes
      if (is==theNumClustersMone) {
        for (size_t ic=0; ic<theNumClustersMone; ++ic) {
          fThePrototypeVectorBook[icl][ic] += aScoreVariableM.GetElem(ic, idat);
        }
        ++theCodeWordFrequencies[icl];
        break;
      }
      // do nothing otherwise: this score point do not match any of the top K 
      //                       code words based on the signs
    }
  }
  // compute the mean i.e. divide by the cardianlities/frequencies and normalise
  // the prototype vectors if K>2
  for (size_t icl=0; icl<theNumClusters; ++icl) {
    T norm  = 0.;
    T ifreq = 1./static_cast<T>(theCodeWordFrequencies[icl]);
    for (size_t is=0; is<theNumClustersMone; ++is) {
      T val = fThePrototypeVectorBook[icl][is]*ifreq; 
      fThePrototypeVectorBook[icl][is] = val;
      norm += val*val;
    }
    if (theNumClusters>2) {
      norm = 1./std::sqrt(norm);
      for (size_t is=0; is<theNumClustersMone; ++is) 
        fThePrototypeVectorBook[icl][is] *= norm;
    }
  }
  
/*  
  for (size_t i=0; i<numClusters-1; ++i)
    std::cerr << fThePrototypeVectorBook[0][i] << " ";
  std::cerr << " after " << std::endl;
*/  
//  this->KMeans(fThePrototypeVectorBook, aScoreVariableM, (theNumClusters>2));
/*  
for (size_t j=0; j<numClusters; ++j) {
  std::cerr<<std::endl;
  for (size_t i=0; i<numClusters-1; ++i)
    std::cerr << fThePrototypeVectorBook[j][i] << " ";
}
  std::cerr << " ---- " << std::endl;
*/  

}

//   In ClusterDataPoint:
//   - then the cosie (or Euclidean in case of K=2) distance between all these
//     prototype vectors and a score point that corresponds to the the i-th data 
//     is computed and the corresponding soft cluster memberships are computed
//     together with the cardinalities
//   - this is done for all data points (score variables) and the AMS of the 
//     training data can be computed 

// the first vector will be normalised
template <typename T>
T KscEncodingAndQM_AMS<T>::ComputeDistance(const T* av, const T* bv, size_t dim) {
  T dist = 0.;
  if (dim>1) {
    // Use the cosine distance
    // norm of the score point (the prototype vectors are already normalised)
    T norm = 0.;
    T dum  = 0.;
    for (size_t is=0; is<dim; ++is) {
      const T ais = av[is];
      dum  += ais*bv[is];
      norm += ais*ais;
    }
    dist =  1. - dum/std::sqrt(norm);
  } else {
    // K=2 -> K-1 = 1: use the Euclidean distance
    const T dx = av[0] - bv[0];
    dist =  std::sqrt(dx*dx);     
  }
  return std::max(0.,dist);
}


// Computes the distance (Euclidean distance for K=2 and cosine distance for K>2)
// between the given score point and the prototype direction that corresponds to 
// pCluster-th cluster.
template <typename T>
T KscEncodingAndQM_AMS<T>::ComputeDistance(const T* aScoreData, size_t pCluster, size_t /*dim*/) {
  const size_t numClusters = this->GetNumClusters();
  return ComputeDistance(aScoreData, fThePrototypeVectorBook[pCluster].data(), numClusters-1);
/*
  if (numClusters>2) {
    // Use the cosine distance
    // norm of the score point (the prototype vectors are already normalised)
    T norm = 0.;
    T dum  = 0.;
    for (size_t is=0; is<numClusters-1; ++is) {
      const T score = aScoreData[is];
      dum  += score*fThePrototypeVectorBook[pCluster][is];
      norm += score*score;
    }
    return 1. - dum/sqrt(norm);
  } else {
    // K=2 -> K-1 = 1: use the Euclidean distance
    const T dx = aScoreData[0] - fThePrototypeVectorBook[pCluster][0];
    return sqrt(dx*dx);     
  }
  */
}


template <typename T>
size_t  KscEncodingAndQM_AMS<T>::ClusterDataPoint(const T* aScoreData, size_t /*dim*/) {
  const size_t numClusters = this->GetNumClusters();
  //assert (numClusters==dim+1 && " Dimension of the score point is not K-1! ");
  T prod = 1.;  
  for (size_t icl=0; icl<numClusters; ++icl) {
    const T dist = ComputeDistance(aScoreData, icl, numClusters-1);
    if (dist==0.) { 
      return icl;
    }
    fTheMemberships[icl] = dist;
    prod *= dist;
  }
  size_t iCluster = 0;
  T      maxCm    = 0.; // not divided by the sum_i prod/x_i these are just the prod/x_i
  T      sumCm    = 0.;
  for (size_t icl=0; icl<numClusters; ++icl) {
    fTheMemberships[icl] = prod/fTheMemberships[icl];
    sumCm     += fTheMemberships[icl];
    if (maxCm<fTheMemberships[icl]) {
      maxCm    =  fTheMemberships[icl];
      iCluster = icl;
    }
  }
  // for the cm_i, we should divide now maxCm by sum   
//  for (size_t icl=0; icl<numClusters; ++icl) 
//    aMemberships[icl] /= sumCm; // cm_i
  //
  return iCluster;
}


// aMemberships will contain the soft cluster mebership value for the assigned 
// cluster if K>2 and only the const 1 value in ase of K=2
template <typename T>
size_t  KscEncodingAndQM_AMS<T>::ClusterDataPoint(const T* aScoreData, size_t /*dim*/, T& aMembership) {
  const size_t numClusters = this->GetNumClusters();
  //assert (numClusters==dim+1 && " Dimension of the score point is not K-1! ");
  T prod = 1.;
  for (size_t icl=0; icl<numClusters; ++icl) {
    const T dist = ComputeDistance(aScoreData, icl, numClusters-1);
    if (dist==0.) { 
      aMembership = 1.;
      return icl;
    }
    fTheMemberships[icl] = dist;
    prod                *= dist;
  }
  size_t iCluster = 0;
  T      maxCm    = 0.; // not divided by the sum_i prod/x_i these are just the prod/x_i
  T      sumCm    = 0.;
  for (size_t icl=0; icl<numClusters; ++icl) {
    fTheMemberships[icl] = prod/fTheMemberships[icl];
    sumCm     += fTheMemberships[icl];
    if (maxCm<fTheMemberships[icl]) {
      maxCm    =  fTheMemberships[icl];
      iCluster = icl;
    }
  }
  // for the cm_i, we should divide now maxCm by sum   
//  for (size_t icl=0; icl<numClusters; ++icl) { 
//    fTheMemberships[icl] /= sumCm; // cm_i
//  }
  
  aMembership = fTheMemberships[iCluster]/sumCm; 
  //
  return iCluster;
}


// aMemberships will contain the soft cluster mebership values for the k=1,...,K
// clusters if K>2 and only the 1,0 or 0,1 in case of K=2.
template <typename T>
size_t  KscEncodingAndQM_AMS<T>::ClusterDataPoint(const T* aScoreData, size_t /*dim*/, T* aMemberships) {
  const size_t numClusters = this->GetNumClusters();
  //assert (numClusters==dim+1 && " Dimension of the score point is not K-1! ");
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

// For AMS quality measure, the ClusterDataSet method must be invoked with 
// flag>0 i.e. compute the cluster membership strengths for all clusters.
// Here this will be checked based on the number of cols in the aCMMatrix that 
// should be number of clusters + 1 (c[0]=cluster index; c[1,..,K]=strength )  
template <typename T>
T KscEncodingAndQM_AMS<T>::ComputeQualityMeasure(Matrix<T, false>& aCMMatrix, const Matrix<T>* /*aScoreMatrix*/, const Matrix<T>* /*theSecondVarForBLFM*/) {
  const size_t numDataPoint = aCMMatrix.GetNumRows(); // number of clustered data ponts
  const size_t numCols      = aCMMatrix.GetNumCols(); 
  const size_t numClusters  = this->GetNumClusters();
  const size_t outlierThres = this->GetOutlierThreshold();
  const T      etaBalance   = this->GetCoefEtaBalance();
  
  T theAMS = 0.;
  if (numCols==1) {
    std::cerr << " *** WARNING in KscEncodingAndQM_AMS::ComputeQualityMeasure \n"
              << "       ClusterDataSet should have been called with flag>0   \n"
              << "       requesting to compute all membership strengths and   \n"
              << "       and store in the 1,...,K cols of the resulted matrix.\n"
              << "       Quality measure will be zero without these !         \n"
              << std::endl;
    return theAMS;         
  }
  
  // collect cardinalities and sum of cluster membership strengths
  std::vector<size_t> theCardinalities(numClusters, 0);
  std::vector<T>      theSumCmPerClusters(numClusters, 0.);
  for (size_t idat=0; idat<numDataPoint; ++idat) {
    const size_t iCluster = static_cast<size_t>(aCMMatrix.GetElem(idat,0));
    ++theCardinalities[iCluster];
    theSumCmPerClusters[iCluster] += (numCols==2) ? aCMMatrix.GetElem(idat,1) : aCMMatrix.GetElem(idat,iCluster+1); 
  }
  for (size_t icl=0; icl<numClusters; ++icl) {
    theAMS += (theCardinalities[icl]>outlierThres) ? theSumCmPerClusters[icl]/theCardinalities[icl] : 0.;
  }
  theAMS /= numClusters;   
  //
  // 2. Penalize extermly unbalanced situations:
  size_t minCardinality = numDataPoint;
  size_t maxCardinality = 0;
  for (size_t i=0; i<numClusters; ++i) {
    minCardinality = minCardinality>theCardinalities[i] ? theCardinalities[i] : minCardinality;
    maxCardinality = maxCardinality<theCardinalities[i] ? theCardinalities[i] : maxCardinality; 
  }
  // [max(C)-min(C)]/max(C)+1
  const T theBalance = static_cast<T>(minCardinality)/static_cast<T>(maxCardinality);
  //  
  // Compute the final Balanced AMS and return
  const T theQMValue = (1.-etaBalance)*theAMS + etaBalance*theBalance;
  this->fTheQualityMeasureValue = theQMValue;
  return theQMValue;    
}




#endif
