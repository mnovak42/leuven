#include <iostream>

#include "types.hh"
#include "Matrix.hh"


#include "Kernels.hh"

#include "sys/time.h"
#include "time.h"

#include "KscWkpca.hh"
#include "KscEncodingAndQM_BLF.hh"
#include "KscEncodingAndQM_AMS.hh"

// for input argument parsing
#include "KscIchol_TrainInputPars.hh"
#include "cxxopts.hh"



// Define data type to be used in the computtaions (double or float)
typedef double DTYPE;
//typedef float DTYPE;
//
// Define data type of the input data (can be any)
typedef double INP_DTYPE;
// or: use the inout data as float but all the other computation is in double
//typedef float INP_DTYPE;

//
// Training the (non-sparse) KCS model on a given training data set using the
// RBF kernel:
//  1. Trains the KSC model on the training data set by using the given hyper
//     parameters (RBF kernel parameter and required number of clusters).
//
//  How to: execute `./Ksc_Train --help` to see the required/optional input
//          arguments.
//
int main(int argc, char **argv) {
  // ===========================================================================
  // Obtain input arguments:
  // -----------------------
  //   Obtain required and optional input argument given to the program by
  //   parsing the input string.
    KscIchol_TrainInputPars<DTYPE, INP_DTYPE> theInParams;
    if (0 > theInParams.GetOpts(argc, argv)) {
      return EXIT_FAILURE;
    }
    if (theInParams.fTheVerbosityLevel > 0) {
      std::cout << theInParams << std::endl;
    }
  //   Define auxiliary variables to measure the computation times, allocate/de-
  //   allocate matrix memory, store number and dimension of the training data.
    struct timeval start;   // initial time stamp - for timing
    struct timeval finish;  // final time stamp   - for timing
    BLAS           theBlas; // only for Matrix memory managmenet here in main
    const bool     theUseGPU    = theInParams.fUseGPU;            // use GPU in training ?
    const size_t   theNumTrData = theInParams.fTheTrDataNumber;   // #training data
    const size_t   theDimTrData = theInParams.fTheTrDataDimension;// its dimension
  // ===========================================================================


  // ===========================================================================
  // Input data for training:
  // ------------------------
  //   Create input training data matrix, allocate memory and load
  //   Note:
  //    - the matrix must be row-major: each input data occupies one row of the
  //      matrix in a memory continous way.
  //    - with type of INP_DTYPE: input data will be stored in this type and the
  //      kernel function will receive two pointers to two rows of the matrix
  //      with this type (i.e. const INP_DTYPE*) together with their (common)
  //      dimension.
    if (theInParams.fTheVerbosityLevel > 1) {
      std::cout << "\n ---- Starts: allocating memory for and loading the training data." << std::endl;
    }
    Matrix<INP_DTYPE, false> theInputTrainingDataM(theNumTrData, theDimTrData);
    theBlas.Malloc(theInputTrainingDataM);
    theInputTrainingDataM.ReadFromFile(theInParams.fTheTrDataFile);
    if (theInParams.fTheVerbosityLevel > 1) {
      std::cout << " ---- Finished: allocating memory for and loading the training data:" << std::endl;
      std::cout << "      ---> Dimensions of M  :(" << theInputTrainingDataM.GetNumRows()
                                                    << " x "
                                                    << theInputTrainingDataM.GetNumCols()
                                                    << ")" << std::endl;
    }
  // ===========================================================================


  // ===========================================================================
  // Training the KSC model:
  // ----------------------
  //   Training the KSC model using the setting (desired number of clusters,
  //   cluster membership encoding, kernel parameter, etc.) given by the input
  //   arguments. An RBF kernel is used.
    KscWkpca<KernelRBF <DTYPE, INP_DTYPE>, DTYPE, INP_DTYPE > theKsc;
    // Set all required members:
    // 1. the 1D RBF kernel paraneter (i.e. bandwidth)
    theKsc.SetKernelParameters(theInParams.fTheClusterRBFKernelPar);
    // 2. the input data matrix (which permutations have already been applied on)
    theKsc.SetInputTrainingDataMatrix(&theInputTrainingDataM);
    // 4. number of desired clusters
    theKsc.SetNumberOfClustersToFind(theInParams.fTheClusterNumber);
    // 5. the cluster membership encoding scheme and model evaluation
    switch (theInParams.fTheClusterEncodingScheme) {
      case 0: theKsc.SetEncodingAndQualityMeasureType(KscQMType::kBLF);
              break;
      case 1: theKsc.SetEncodingAndQualityMeasureType(KscQMType::kAMS);
              break;
      default: theKsc.SetEncodingAndQualityMeasureType(KscQMType::kAMS);
    }
    // the weight to be given to the balance term in the model evaluation
    theKsc.SetQualityMeasureEtaBalance(theInParams.fTheClusterEvalWBalance);
    // the cardinality threshold below which clusters are considered to be
    // outliers and contibute with zero to the clustering quality measure
    theKsc.SetQualityMeasureOutlierThreshold(theInParams.fTheClusterEvalOutlierThreshold);
    // set request to use GPU during the training phase
    theKsc.SetUseGPU(theUseGPU);
    if (theInParams.fTheVerbosityLevel > 0) {
      std::cout << "\n ---- Starts: training the KSC model." << std::endl;
    }
    gettimeofday(&start, NULL);
    theKsc.Train(theInParams.fTheNumBLASThreads, true, theInParams.fTheClusterLevel, theInParams.fTheVerbosityLevel);
    gettimeofday(&finish, NULL);
    double durationTr = ((double)(finish.tv_sec-start.tv_sec)*1000000 + (double)(finish.tv_usec-start.tv_usec)) / 1000000;
    // check is results needs to be written (if theInParams.fTheClusterLevel>0)
    if (theInParams.fTheClusterLevel>0) {
      // this must be the same matrix (even in memory) that was set i.e. the
      // thePermInputTrainingDataM
//      const Matrix<INP_DTYPE, false>* thePermTrDataM = theKscWkpcaIchol.GetPermutedTrDataMatrix();
      const Matrix<DTYPE, false>*     theClusterResM = theKsc.GetTheClusterMembershipMatrix();
//      thePermTrDataM->WriteToFile(theInParams.fTheClusterResDataFile);
      theClusterResM->WriteToFile(theInParams.fTheClusterResFile);
    }

    if (theInParams.fTheVerbosityLevel > 0) {
      std::cout << " ---- Finished: training the KSC model"                   << std::endl;
      std::cout << "      ---> Duration         : " << durationTr << " [s]"   << std::endl;
      const KscEncodingAndQM<DTYPE>* theEncoding =  theKsc.GetEncodingAndQualityMeasure();
      std::cout << "      ---> The encoding(QM) : " << theEncoding->GetName() << std::endl;
      std::cout << "      --->   Quality value  : " << theEncoding->GetTheQualityMeasureValue() << std::endl;
      std::cout << "      --->   Eta balance    : " << theEncoding->GetCoefEtaBalance()         << std::endl;
      std::cout << "      --->   Outlier thres. : " << theEncoding->GetOutlierThreshold() << std::endl;
      if (theInParams.fTheClusterLevel>0) {
        std::cout << "      ---> Result is writen : " << std::endl;
        std::cout << "      --->   Clustering     : " << theInParams.fTheClusterResFile     << std::endl;
        std::cout << "      --->   Training data  : " << theInParams.fTheClusterResDataFile <<std::endl;
      }
      std::cout << std::endl;
    }
  // ===========================================================================

  // free remaining allocated memeory
  theBlas.Free(theInputTrainingDataM);


  return EXIT_SUCCESS;
}
