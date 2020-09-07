
#ifndef MATRIX_HH
#define MATRIX_HH

#include "definitions.hh"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>


// Data structure to store col/row-vector/matrix (alloc/free depend on the acltual 
// version of BLAS implementation used). Col-major order by default!

// the allcoator (the one that calls e.g. BLAS::Malloc/Calloc for fData memory 
// allocation) is resposnible to free the corresponding memory (e.g. by calling 
// BLAS::Free).

// Matrix class templated on the data type and order i.e. col- or row-major.
// For this later option, the class is specialised below for performance reasons.

// Base class for CRTP: will implement the default col-major methods and store 
// all data
template <typename M, class T>
class MatrixBase {
public:  
    // prefered CTR
    MatrixBase (const size_t nr, const size_t nc=1) 
    : fData(nullptr), 
  //      fIsColMajor(isColMajor),
      fNRows(nr), 
      fNCols(nc),
      fSize(fNRows*fNCols), 
      fINRows(1./fNRows), 
      fINCols(1./fNCols) {}

    // compute the continuos (col or row major) index by given the row and col indices
    inline size_t Indx(const size_t ir, const size_t ic=0) const {
      assert( ir<fNRows && "\n **** MatrixBase ::Indx(): !(ir<fNRows) ****\n");
      assert( ic<fNCols && "\n **** MatrixBase ::Indx(): !(ic<fNCols) ****\n");
      return static_cast<const M*>(this)->Indx(ir, ic);
    }
    // compute the col index given the continuos index
    inline size_t ColIndx(const size_t indx) const {
      assert( indx<fSize && "\n **** MatrixBase ::ColIndx(): !(indx<fSize) ****\n");
      return static_cast<M*>(this)->ColIndx(indx);
    }
    // compute the row index given the continuos index 
    inline size_t RowIndx(const size_t indx) const {
      assert( indx<fSize && "\n **** MatrixBase ::RowIndx(): !(indx<fSize) ****\n");
      // computes the col index
      return indx - ColIndx(indx)*fNRows;
    }
    
    // method to obtain pointers to a give col/row (icr will be interpreted as 
    // col index for col major and as row index for row major Matrices
    T* GetPtrToBlock(const size_t icr) const {
      return static_cast<M*>(this)->GetPtrToBlock(icr);
    }


    // access elements by giving the: row and col indices
    inline T GetElem(const size_t ir, const size_t ic) const { 
      assert( ir<fNRows && "\n **** MatrixBase ::GetElem(): !(ir<fNRows) ****\n");
      assert( ic<fNCols && "\n **** MatrixBase ::GetElem(): !(ic<fNCols) ****\n");
      return fData[Indx(ir, ic)];  
    }
    // access elements by giving the: continuos idx
    inline T GetElem(const size_t indx) const { 
      assert( indx<fSize && "\n **** MatrixBase ::GetElem(): !(indx<fSize) ****\n");
      return fData[indx];  
    }
    // set element by giving the: row, col indices and the value
    inline void SetElem(const size_t ir, const size_t ic, const T val) { 
      assert( ir<fNRows && "\n **** MatrixBase ::SetElem(): !(ir<fNRows) ****\n");
      assert( ic<fNCols && "\n **** MatrixBase ::SetElem(): !(ic<fNCols) ****\n");
      fData[Indx(ir, ic)] = val;  
    }
    // set elements by giving the: continuos idx and the value
    inline void SetElem(const size_t indx, const T val) { 
      assert( indx<fSize && "\n **** MatrixBase ::Elem(): !(indx<fSize) ****\n");
      fData[indx] = val;  
    }
    
    // Memory allocation needs to be done outside
    // It is assumed that the number of
    //  - lines in the file = fNRows
    //  - cols  in the file = fNCols  
    // and each line will be one of row of the matrix
    // Note: data will be memory continuos along the dimensions using row major 
    //       while along the input data number when using col-major order. 
    void ReadFromFile(const std::string& fname) {
      char buffer[16184]; // 16184
      std::ifstream inFile(fname, std::ios::in);
      inFile.rdbuf()->pubsetbuf(buffer, 16184); 
      if ( !inFile.is_open() ) {
        std::cerr << "  *** Matrix::ReadFromFile: Cannot open the file = " << fname << std::endl;
        exit(1);
      }
      T dum;
      size_t curNumData = 0;
      for (size_t in=0; in<fNRows; ++in) {   // row index = index of data point
        for (size_t id=0; id<fNCols; ++id) { // col index = dimension
          inFile >> dum;
          SetElem(in, id, dum);
        }
        ++curNumData;
      }
      // cross-check if number of data read equal to the size (number of rows) of the Matrix
      if ( curNumData!=fNRows ) {
        std::cerr<<"  *** Matrix::ReadFromFile: Number of data read = " << curNumData 
                 << " smaller than the number of rows in the matrix = " << fNRows << std::endl; 
      }
      inFile.close();
    }
    
    //
    // Writes matrix inot file (ascii format) 
    void WriteToFile(const std::string& fname, bool doTranspose=false, int precision=8, int width=0) const {
      char buffer[2048]; // 16184
      std::ofstream outFile(fname, std::ios::out);
      outFile.rdbuf()->pubsetbuf(buffer, 2048); 
      if ( !outFile.is_open() ) {
        std::cerr<< "  *** Matrix::WriteToFile: Cannot open the file = " << fname << std::endl;
        exit(1);
      }
      outFile.setf(std::ios::scientific);
      //outFile.setf(std::ios::fixed);
      outFile.precision(precision);
      const int wdt = (width==0) ? precision+8 : width;
      if (doTranspose) {
        for (size_t in=0; in<fNCols; ++in) {   // col index = dimension 
          for (size_t id=0; id<fNRows; ++id) { // row index = index of data point
            outFile << std::setw(wdt) << GetElem(id, in) << " ";
          }
          outFile << std::endl;
        }  
        outFile.close();        
      } else {
        for (size_t in=0; in<fNRows; ++in) {   // row index = index of data point
          for (size_t id=0; id<fNCols; ++id) { // col index = dimension
            outFile << std::setw(wdt) << GetElem(in, id) << " ";
          }
          outFile << std::endl;
        }  
        outFile.close();
      }
    }

    
    T*        GetDataPtr() const { return fData;       }
    T**       GetDataPtrAdrs()   { return &fData;      }
    void      SetDataPtr(T* ptr) { fData = ptr;        }
    bool      IsColMajor() const { return fIsColMajor; }
    size_t    GetNumRows() const { return fNRows;      }
    size_t    GetNumCols() const { return fNCols;      }
    size_t    GetSize()    const { return fSize;       }
    double    GetINRows()  const { return fINRows;     }
    double    GetINCols()  const { return fINCols;     }
    
public:
// Data members 
    T*        fData;
    
    bool      fIsColMajor;  
            
    size_t    fNRows;
    size_t    fNCols;
    size_t    fSize;
     
    double    fINRows;
    double    fINCols;
};





// derived class templated  on col-row majority (col-major by defult)
template <class T, bool isColMajor = true >
class Matrix : public MatrixBase< Matrix <T,isColMajor>, T > {
public:  
   // just calls the base ctr
    Matrix(const size_t nr, const size_t nc=1) 
    : MatrixBase< Matrix <T,isColMajor>, T >(nr, nc) { 
      this->fIsColMajor = isColMajor;
    }

    // compute the continuos: col-major(LD=#Rows) data index
    inline size_t Indx(const size_t ir, const size_t ic=0) const {
//      assert( ir<fNRows && "\n **** Matrix::Indx(): !(ir<fNRows) ****\n");
//      assert( ic<fNCols && "\n **** Matrix::Indx(): !(ic<fNCols) ****\n");
      return ic*this->fNRows + ir;
    }
    // compute the col index given the continuos index
    inline size_t ColIndx(const size_t indx) const {
//      assert( indx<fSize && "\n **** Matrix::ColIndx(): !(indx<fSize) ****\n");
      return static_cast<size_t>(indx*this->fINRows);
    }
    // compute the row index given the continuos index and (optionaly) the col. index    
    inline size_t RowIndx(const size_t indx) const {
//      assert( indx<fSize && "\n **** Matrix::RowIndx(): !(indx<fSize) ****\n");
      // computes the col index
      return indx - ColIndx(indx)*this->fNRows;
    }
    
    // method to obtain pointers to the a give col
    T* GetPtrToBlock(const size_t ic) const {
      assert( ic<this->fNCols && "\n **** Matrix::GetPtrToCol: !(ic<fNCols) ****\n");
      return &(this->fData[ic*this->fNRows]);
    }

};


// partial specialisation for Row-major order
template <class T>
class Matrix<T, false> : public MatrixBase< Matrix <T,false>, T > {
public:  
    Matrix(const size_t nr, const size_t nc=1) 
    : MatrixBase< Matrix <T,false>, T >(nr, nc) { 
      this->fIsColMajor = false;
    }
    
    // compute the continuos, row-major data index by giving the row and col indx
    inline size_t Indx(const size_t ir, const size_t ic=0) const {
//      assert( ir<fNRows && "\n **** Matrix::Indx(): !(ir<fNRows) ****\n");
//      assert( ic<fNCols && "\n **** Matrix::Indx(): !(ic<fNCols) ****\n");
      return ir*this->fNCols + ic;
    }
    // compute the row index given the continuos index
    inline size_t RowIndx(const size_t indx) const {
//      assert( indx<fSize && "\n **** Matrix::RowIndx(): !(indx<fSize) ****\n");
      return static_cast<size_t>(indx*this->fINCols);
    }
    // compute the col index given the continuos index
    inline size_t ColIndx(const size_t indx) const {
//      assert( indx<fSize && "\n **** Matrix::ColIndx(): !(indx<fSize) ****\n");
      return  indx - RowIndx(indx)*this->fNCols;
    }
    
    // method to obtain pointers to the a give row
    T* GetPtrToBlock(const size_t ir) const {
      assert( ir<this->fNRows && "\n **** Matrix::GetPtrToRow: !(ir<fNRows) ****\n");
      return &(this->fData[ir*this->fNCols]);
    }

};




#endif // MATRIX_HH
