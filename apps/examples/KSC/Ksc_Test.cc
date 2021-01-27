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
#include "KscIchol_TestInputPars.hh"
#include "cxxopts.hh"

//#define CHI2

// Define data type to be used in the computaions (double or float)
typedef double DTYPE;
//typedef float DTYPE;
//
// Define data type of the input data (can be any)
typedef double INP_DTYPE;
// or: use the inout data as float but all the other computation is in double
//typedef float INP_DTYPE;

//
// Training the (non-sparse) KCS model on a given training data set using the RBF
// kernel and apply the trained model to cluster the test data set.
//  1. Trains the KSC model on the training data set by using the given hyper
//     parameters (RBF kernel parameter and required number of clusters)
//  2. Clusters the test data set using the KSC model trained in the previous
//     step.
//
//  How to: execute `./Ksc_Test --help` to see the required/optional input
//          arguments
//
int main(int argc, char **argv) {
  // ===========================================================================
  // Obtain input arguments:
  // -----------------------
  //   Obtain required and optional input argument given to the program by
  //   parsing the input string.
    KscIchol_TestInputPars<DTYPE, INP_DTYPE> theInParams;
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
    BLAS           theBlas; // only for Matrix memory managmenet here in the main
    const bool     theUseGPU      = theInParams.fUseGPU;              // use GPU in training
    const size_t   theNumTrData   = theInParams.fTheTrDataNumber;     // #training data
    const size_t   theDimTrData   = theInParams.fTheTrDataDimension;  // its dimension
    const size_t   theNumTestData = theInParams.fTheTestDataNumber;   // #test data
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
  //   arguments. An RBF kernel is used (the Chi2 kernel can also be used here).
  #ifdef CHI2
    KscWkpca<KernelChi2 <DTYPE, INP_DTYPE>, DTYPE, INP_DTYPE > theKsc;
  #else
    KscWkpca<KernelRBF <DTYPE, INP_DTYPE>, DTYPE, INP_DTYPE > theKsc;
  #endif
    // Set all required members:
    // 1. the RBF kernel paraneter (i.e. bandwidth)
    theKsc.SetKernelParameters(theInParams.fTheClusterRBFKernelPar);
    // 2. the input data matrix (which permutations have already been applied on)
    theKsc.SetInputTrainingDataMatrix(&theInputTrainingDataM);
    // 4. desired number of clusters
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

    if (theInParams.fTheVerbosityLevel > 0) {
      std::cout << " ---- Finished: training the KSC model"                   << std::endl;
      std::cout << "      ---> Duration         : " << durationTr << " [s]"   << std::endl;
      const KscEncodingAndQM<DTYPE>* theEncoding =  theKsc.GetEncodingAndQualityMeasure();
      std::cout << "      ---> The encoding(QM) : " << theEncoding->GetName() << std::endl;
      std::cout << "      --->   Quality value  : " << theEncoding->GetTheQualityMeasureValue() << std::endl;
      std::cout << "      --->   Eta balance    : " << theEncoding->GetCoefEtaBalance()         << std::endl;
      std::cout << "      --->   Outlier thres. : " << theEncoding->GetOutlierThreshold()       << std::endl;
      std::cout << std::endl;
    }
  // ===========================================================================


  // ===========================================================================
  // Input data for test (i.e. to cluster):
  // --------------------------------------
  //   Create input test data matrix, allocate memory and load.
  //   Note: (the same as for the training data)
    if (theInParams.fTheVerbosityLevel > 1) {
      std::cout << "\n ---- Starts: allocating memory for and loading the test data." << std::endl;
    }
    Matrix<INP_DTYPE, false> theInputTestDataM(theNumTestData, theDimTrData);
    theBlas.Malloc(theInputTestDataM);
    theInputTestDataM.ReadFromFile(theInParams.fTheTestDataFile);
    if (theInParams.fTheVerbosityLevel > 1) {
      std::cout << " ---- Finished: allocating memory for and loading the test data:" << std::endl;
      std::cout << "      ---> Dimensions of M  :(" << theInputTestDataM.GetNumRows()
                                                    << " x "
                                                    << theInputTestDataM.GetNumCols()
                                                    << ")" << std::endl;
    }
  // ===========================================================================


  // ===========================================================================
  // Testing (out-of-sample extension) i.e. cluster assigment of the test data:
  // ---------------------------------------------------------------------------
    if (theInParams.fTheVerbosityLevel > 0) {
      std::cout << "\n ---- Starts: clustering the test data with the KSC model." << std::endl;
    }
    //
    gettimeofday(&start, NULL);
    theKsc.Test(theInputTestDataM, theInParams.fTheNumBLASThreads, theInParams.fTheClusterLevel, theInParams.fTheVerbosityLevel);
    gettimeofday(&finish, NULL);
    durationTr = ((double)(finish.tv_sec-start.tv_sec)*1000000 + (double)(finish.tv_usec-start.tv_usec)) / 1000000;
    // write results into file
    const Matrix<DTYPE, false>* theClusterResM = theKsc.GetTheClusterMembershipMatrix();
    theClusterResM->WriteToFile(theInParams.fTheClusterResFile);
    // print out information
    if (theInParams.fTheVerbosityLevel > 0) {
      std::cout << " ---- Finished: test data cluster assignment"             << std::endl;
      std::cout << "      ---> Duration         : " << durationTr << " [s]"   << std::endl;
      const KscEncodingAndQM<DTYPE>* theEncoding =  theKsc.GetEncodingAndQualityMeasure();
      std::cout << "      ---> The encoding(QM) : " << theEncoding->GetName() << std::endl;
      std::cout << "      --->   Quality value  : " << theEncoding->GetTheQualityMeasureValue() << std::endl;
      std::cout << "      --->   Eta balance    : " << theEncoding->GetCoefEtaBalance()         << std::endl;
      std::cout << "      ---> Result is writen : " << std::endl;
      std::cout << "      --->   Clustering     : " << theInParams.fTheClusterResFile           << std::endl;
      std::cout << std::endl;
    }
  // ===========================================================================

  // free allocated memory
  theBlas.Free(theInputTrainingDataM);
  theBlas.Free(theInputTestDataM);

  return EXIT_SUCCESS;
}
