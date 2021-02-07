
#include <iostream>

#include "types.hh"
#include "Matrix.hh"


#include "IncCholesky.hh"
#include "Kernels.hh"

#include "sys/time.h"
#include "time.h"


// Define data type for the test(double or float)
typedef double DTYPE;
//typedef float DTYPE;
//
// Define data type of the input data (can be any)
typedef double INP_DTYPE;
// or: use the inout data as float but all the other computation is in double
//typedef float INP_DTYPE;


int main() {
  double duration;
  struct timeval start,finish;


  // file name to read/write
//  const std::string  inFname = "inMatrix.dat";
  const std::string  inFname = "training_data_100000_std";
//  const std::string outFname = "outMatrix.dat";
  //
  // number of row and cols in the 'inMatrix.dat' file
  size_t numData = 100000;
  size_t numDim  = 2;
  // create the input data matrix and allocate memory
  Matrix<INP_DTYPE, false> theInputDataM(numData, numDim);
  // only for memory managmenet
  BLAS theBlas;
  theBlas.Malloc(theInputDataM);
  // read data
  theInputDataM.ReadFromFile(inFname);

  // IChol
  IncCholesky<KernelRBF <DTYPE, INP_DTYPE>, DTYPE, INP_DTYPE> theIChol;
  // set kernle parameters: bandwidth in case of KernelRBF
  INP_DTYPE rbfBW = 0.006;
  theIChol.SetKernelParameters(rbfBW);
  // set the input data matrix
  theIChol.SetInputDataMatrix(&theInputDataM);
  // perform the decompositon
  double tolError = 1.0E-3;
  size_t maxItr   = 1000;//260;
  bool   transp   = true;

  printf("\n ---- Starts: incomplete Cholesky decomposition of the Kernel matrix\n");
  gettimeofday(&start, NULL);
  theIChol.Decompose(tolError, maxItr, transp);
  gettimeofday(&finish, NULL);
  printf("\n ---- Finished: incomplete Cholesky decomposition of the Kernel matrix\n");

  duration = ((double)(finish.tv_sec-start.tv_sec)*1000000 + (double)(finish.tv_usec-start.tv_usec)) / 1000000;
  printf("\n\n == TIMING: %lf [s]\n", duration);
  std::cerr<< "== Final error = " << theIChol.GetFinalResidual() << std::endl;
  std::cerr<< "== G = " << theIChol.GetICholMatrix()->GetNumRows() << " x "<< theIChol.GetICholMatrix()->GetNumCols()  << std::endl;

  // print the ICholMatrix
  //theIChol.GetICholMatrix()->WriteToFile("theICholM");

  std::cerr<< " --- Forming the permutated input data matrix. " << std::endl;
  // write the input data ino file by taking into account the permutations perfomred during the ICD
  // (row major order)
  Matrix<INP_DTYPE, false> thePermutedInputDataM(theInputDataM.GetNumRows(), theInputDataM.GetNumCols());
  theBlas.Malloc(thePermutedInputDataM);
  const size_t sizeOfRow = sizeof(DTYPE)*theInputDataM.GetNumCols();
  for (size_t ir=0; ir<theInputDataM.GetNumRows(); ++ir)
    memcpy(thePermutedInputDataM.GetPtrToBlock(ir), theInputDataM.GetPtrToBlock(theIChol.GetPermutationVector()[ir]), sizeOfRow);

  std::cerr<< " --- Writing the permutated input data into file. " << std::endl;
  thePermutedInputDataM.WriteToFile("PermInpData.dat");

  // free allocated data
  theBlas.Free(theInputDataM);

  return 0;
}
