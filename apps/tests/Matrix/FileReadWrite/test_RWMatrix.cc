
#include <iostream>

#include "types.hh"
#include "Matrix.hh"

// Define data type for the test(double or float)
typedef double DTYPE;
//typedef float DTYPE;

int main() {
  // file name to read/write
  const std::string  inFname = "../data/inMatrix.dat";
  const std::string outFname = "../data/outMatrix.dat";
  //
  // number of row and cols in the 'inMatrix.dat' file
  size_t numData = 6;
  size_t numDim  = 2;
  // create the matrix and allocate memory
  Matrix<DTYPE> A(numData, numDim);
  // only for memory managmenet
  BLAS theBlas;
  theBlas.Malloc(A);
  // read/write data 
  A.ReadFromFile(inFname);
  A.WriteToFile(outFname);
  A.WriteToFile(outFname+"_1",3, 6);
  // free allocated data
  theBlas.Free(A);
  
  return 0;
}