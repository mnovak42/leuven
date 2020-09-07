
#ifndef KSCENCODINGANDQM_BLF_HH
#define KSCENCODINGANDQM_BLF_HH

#include <iostream>

#include "KscEncodingAndQM.hh"

/**
 * @class  KscEncodingAndQM_BLF
 * @brief  Sign based cluster membership encoding scheme and Balanced Line Fit
 *         (BLF) model quality measure.
 *
 * Follows the base KscEncodingAndQM class interfaces to implement a cluster
 * membership encoding based on the signs of the training score data components: 
 * data that belong to the same cluster are located in the same orthant of the 
 * score space. The cluster assigment then is done by computing the Hamming 
 * distance between these sign based code words and the binarised score space 
 * representation of the data. The data will be assigned to the cluster that 
 * gives the smallest Hamming distance. In order to generate a qualty measure, 
 * the collinearity of the score data that belongs to the same cluster is 
 * measured by computing a Line Fit and additional term accounts the Balance 
 * of teh resulted clusters (see more at the KscEncodingAndQM_BLF<T>::ComputeQualityMeasure
 * interface implementation).
 *
 * Since the base class implements the necessary sing based code book generation 
 * in its KscEncodingAndQM<T>::GenerateCodeBook method, the corresponding Hamming 
 * distance computation in its KscEncodingAndQM<T>::ComputeDistance method and the
 * corresponding cluster assigments in its KscEncodingAndQM<T>::ClusterDataPoint 
 * methods, the only interface method is the KscEncodingAndQM_BLF<T>::ComputeQualityMeasure 
 * to be implemented here. 
 *
 */
template <typename T>
class KscEncodingAndQM_BLF : public KscEncodingAndQM<T> {

public:
  
  /**@brief Constructor. */
  KscEncodingAndQM_BLF() :  KscEncodingAndQM<T>(kBLF, "BLF") {}

  /**
  *@brief Implementation of the intrface method to compute the BLF quality measure 
  *       for model selection.
  * 
  * The Line Fit part of this quality measure is motivated by the fact that the 
  * score space representation of the data points, that belong to the same cluster,
  * are collinear in the ideal case. By measuring the collinearity of the score 
  * variables that have been assigned to the same cluster, one can have an indicator 
  * on how far the given KSC model and its results are from the ideal case. 
  * The collinearity of the score variables assigned the same cluster is 
  * measured by determining the fraction of variance contained along the first 
  * principal direction to the total. It can be done by forming the covariance 
  * matrix of the corresponding score data point
   \f[
       \texttt{Cov}^{(k)} := \frac{1}{|\mathcal{A}_k|} Z^{(k)^T}Z^{(k)}, k=1,\dots,K
   \f]
  * where the clustering resulted the partition \f$ \mathcal{A}_1,\dots,\mathcal{A}_K\f$
  * of the score data points and the rows of the matrix \f$ Z^{(k)} \in \mathbb{R}^{|\mathcal{A}_k|\times(K-1)}\f$
  * are the score data points that belong to the \f$ k\f$-th cluster. 
  * The required ratio of the variance can be obtained by computing the eigenvalues 
  * \f$ \lambda_1^{(k)} \geq \lambda_2^{(k)} \geq \dots \lambda_{K-1}^{(k)} \f$
  * of the covariance matrix for the \f$ k\f$-th cluster and taking the ratio 
  * \f$ \lambda_1^{(k)}/\sum_{p=1}^{K-1} \lambda_p^{(k)} \f$. This ratio is equal 
  * to 1 in the ideal case when all the variance is contained along the first
  * principal direction and equal to \f$ 1/(K-1)\f$ when evenly distributed along
  * all the \f$ K-1\f$ principal directions. These scalled to the \f$ [0,1]\f$
  * intervall by defining the Line Fit as 
    \f[
       \texttt{LF}(K>2) = \frac{1}{K}\frac{K-1}{K-2} \sum_{k=1}^{K} 
         \left[ 
           \frac{ 
             \lambda_1^{(k)}
           }{
             \sum_{p=1}^{K-1} \lambda_p^{(k)}
           } - \frac{1}{K-1}
         \right]
    \f]
  * In case of \f$ K=2\f$ there is a single score variable i.e. the score data 
  * a single  dimensionals \f$K-1=1\f$ so the above procedure is not applicable. 
  * However, taking the \f$ \mathbf{z}_i = \sum_{j=1}^{N_{tr}} K(\mathbf{z}_i, \mathbf{z}_j) + b^{(1)}\f$ 
  * variable beyond the single score variable \f$ \mathbf{z}_i = \sum_{j=1}^{N_{tr}} \mathbf{\beta}^{(1)}K(\mathbf{z}_i, \mathbf{z}_j) + b^{(1)}\f$ 
  * (where \f$ \mathbf{\beta}^{(1)}\f$ is the leading eigenvector of the \f$ D^{-1}M_d\Omega\f$ matrix and \f$ b^{(1)}\f$ is the corresponding bias term)
  * the above procedure becomes applicable and the corresponding Line Fit 
    \f[
       \texttt{LF}(K=2) = 
         \sum_{k=1}^{2}
         \left[ 
           \frac{ 
             \lambda_1^{(k)}
           }{
             \lambda_1^{(k)} + \lambda_2^{(k)}
           }
           -\frac{1}{2}
         \right]
    \f]  
  * 
  * In order to provide the possibility to give more weights to KSC models that 
  * result more balanced clustering, a balance (BL) terem  can be introduced as
  * \f$ \texttt{BL} = \texttt{min}(|\mathcal{A}_k|)/\texttt{max}(|\mathcal{A}_k|), k=1,\dots,K \f$
  * and the final quality measure is 
    \f[
        \texttt{BLF} = [1-\eta]\texttt{LF} + \eta\texttt{BL} 
    \f]
  * with \f$ \eta \in [0,1]\f$ as an input parameter determines the importance 
  * of the balance over the collinearity i.e. line fit term.
  *
  * @param[in] aCMMatrix reference to the matrix that stores the assigned 
  *   cluster indices as its first (zeroth) column. Note, that only this column 
  *   is used in the quality measure computation and this infomation is
  *   filled by the KscEncodingAndQM_BFL<T>::ClusterDataSet method with the 
  *   corresponding \texttt{flag>=0} value.
  * @param[in] aScoreMatrix pointer to the score data matrix that was clusterred
  *   with the KscEncodingAndQM<T>::ClusterDataSet method and the corresponding 
  *   cluster assigment was filled in the \f$ \texttt{aCMMatrix}\f$ matrix
  * @param[in] theSecondVarForBLFM pointer to the second variable matrix used 
  *   when the required number of clusters is two (**used only in case of K=2**)
  * @return The computed Balanced Line Fit KSC model evaluation criterion.
  */
  T ComputeQualityMeasure(Matrix<T, false>& aCMMatrix, const Matrix<T>* aScoreMatrix, const Matrix<T>* theSecondVarForBLFM) override;

};


// computes the balanced line fit (aScoreMatrix is the score variable matrix or 
// in case of K=2 the score and extra variable matrix with these 2 rows 
template <typename T>
T  KscEncodingAndQM_BLF<T>::ComputeQualityMeasure(Matrix<T, false>& aCMMatrix, const Matrix<T>* aScoreMatrix, const Matrix<T>* theSecondVarForBLFM)  {
  const size_t numDataPoint = aCMMatrix.GetNumRows(); // number of clustered data ponts
  const size_t numClusters  = this->GetNumClusters(); // number of clusters
  const size_t outlierThres = this->GetOutlierThreshold();
  const T      etaBalance   = this->GetCoefEtaBalance();
  
  const bool   is2Clusters  = (numClusters==2);       // K=2 ? 
  // set dimension of the covariance matrix
  const size_t dimOfVars    = is2Clusters ? numClusters : numClusters-1; 
  // for each cluster: form the conariance matrix of the score points that belong 
  // to the given cluster, compute the fraction of variance along the leading 
  // eignevector. 
  //
  // The line fit part
  T theLF = 0.;
  // the cov. matrix and its eigenvalues
  BLAS theBlas;
  Matrix<T> theCovarM(dimOfVars, dimOfVars);
  theBlas.Malloc(theCovarM);
  Matrix<T> theEigenvals(dimOfVars, 1);
  theBlas.Malloc(theEigenvals);
  // cardinalities will be collected as well
  std::vector<size_t> theCardinalities(numClusters, 0);
  std::vector<size_t> theDataIndices(numDataPoint);
  for (size_t icl=0; icl<numClusters; ++icl) {  
    // check which data belongs to this cluster, collect their indices, compute 
    // the corresponding mean score variable and cardinality  
    std::vector<T>    theMeanScoreVector(dimOfVars, 0.); 
    for (size_t idat=0; idat<numDataPoint; ++idat) {
      const size_t iCluster = static_cast<size_t>(aCMMatrix.GetElem(idat,0));
      if (icl==iCluster) {
        theDataIndices[theCardinalities[icl]] = idat;
        for (size_t is=0; is<numClusters-1; ++is) {
          theMeanScoreVector[is] += aScoreMatrix->GetElem(is, idat);
        }
        if (is2Clusters) {
          theMeanScoreVector[1] += theSecondVarForBLFM->GetElem(idat);
        }
        ++theCardinalities[icl];
      }
    }
    // form the mean centered matrix of the score variables of this cluster Z^(k) (K-1)x(|A_k|)
    const size_t theCardinality = theCardinalities[icl];
    if (theCardinality<=outlierThres) 
      continue;
    // - compute the mean score vector of this cluster
    for (size_t is=0; is<std::max(numClusters-1,std::size_t(2)); ++is) {
      theMeanScoreVector[is] /= theCardinality;
    }
    // - Z^(k) (K-1)x(|A_k|)
    Matrix<T> theMeanCentScoreM(dimOfVars, theCardinality);
    theBlas.Malloc(theMeanCentScoreM);
    for (size_t i=0; i<theCardinality; ++i) {
      for (size_t is=0; is<numClusters-1; ++is) {
        theMeanCentScoreM.SetElem(is,i, aScoreMatrix->GetElem(is, theDataIndices[i]) - theMeanScoreVector[is]);
      }
      if (is2Clusters) {
        theMeanCentScoreM.SetElem(1,i, theSecondVarForBLFM->GetElem(theDataIndices[i]) - theMeanScoreVector[1]);
      }
    }
    // copute the covariance matrix as Z^(k) Z^(k)^T /|A_k|
    theBlas.XGEMM(theMeanCentScoreM, theMeanCentScoreM, theCovarM, 
                  1./theCardinality, 0., false, true);
    theBlas.Free(theMeanCentScoreM);
    // Compute all eigenvalues of the covariance matrix (ascending order)
    const size_t numEigenVals = theBlas.XSYEVR(theCovarM, theEigenvals);
    T sumEigenvals = 0.;
    for (size_t i=0; i<numEigenVals; ++i)  { 
      sumEigenvals += theEigenvals.GetElem(i);
    }
    // sum up the top/sum eigenvalues
    theLF += theEigenvals.GetElem(dimOfVars-1)/sumEigenvals;
  }
  // free memory allocated for the covaraince matrix and its eigenvalues
  theBlas.Free(theCovarM);
  theBlas.Free(theEigenvals);
  //
  if (is2Clusters) {
    theLF -= 1.;
  } else {
    theLF = ((1.-1./numClusters)*theLF  - 1.)/(numClusters-2.);  
  }
  //
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
  // Compute final BLF and return
  const T theQMValue = (1.-etaBalance)*theLF + etaBalance*theBalance; 
  this->fTheQualityMeasureValue = theQMValue;
  return theQMValue;    
}

#endif