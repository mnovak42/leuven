
#include "types.hh"
#include <cmath>

// for std::memcpy
#include <cstring>

template < class TKernel, typename T, typename TInputD >
void IncCholesky < TKernel, T, TInputD >::Decompose(double tolError, size_t maxIter, bool transpose) {
  // check if the fInputData ptr has been set (is not nullptr) 
  if ( !fInputDataM ) {
    std::cerr << " *** IncCholesky::Decompose: \n" 
              << "     Input data matrix must be set before decompositon! "
              << std::endl;
    exit(1);
  }
  // set the number and dimension of the input data local variables
  const size_t numInpData = fInputDataM->GetNumRows();
  const size_t dimInpData = fInputDataM->GetNumCols();
  // set/reset permutations and diagonals of the G^{T}G approximated Kernel Matrix
  fPermutationVect.resize(numInpData);
  std::vector<T>   diagGG(numInpData);          
  for (size_t in=0; in<numInpData; ++in) {
    fPermutationVect[in] = in;
    diagGG[in]           = 1.;  // normalised kernel assumed i.e. k(x,x)=x^x=1
  }
  // The matrix G will be generated in RxN upper triangular for using col major 
  // format during the work: better to compute the sum_{t=1}^{i-1} R_{ti} R{tj}
  const size_t preSize   = 100;     // initial and increment size
  size_t curCapacity     = preSize; // current row capacity in G matrix
  Matrix<T>* workGM      = new Matrix<T>(curCapacity, numInpData); // G matrix
  BLAS theBlas;
  theBlas.Calloc(*workGM); // allocate memory
  // 
  size_t     itr = 0;          // iteration counter
  size_t     piv = 0;          // pivot: currently selected input data index
  double   resid = numInpData; // residual: sum of the diagonals of the approximaed Kernel matrix 
  T      maxDiag = 0.;         //  
  while (resid > tolError && itr < maxIter) {
    // check if the capacity has been reached: increase it 
    if ( itr==curCapacity-1 ) {
      curCapacity += preSize;
      Matrix<T>* newM = new Matrix<T>(curCapacity, numInpData);
      theBlas.Calloc(*newM);
      const size_t sizeOneCol = sizeof(T)*workGM->GetNumRows();
//      const size_t nn = workGM->GetNumRows();
      for (size_t ic=0; ic<numInpData; ++ic) {
        std::memcpy(newM->GetPtrToBlock(ic), workGM->GetPtrToBlock(ic), sizeOneCol);
//        for (size_t ir=0; ir<nn; ++ir) {
//          newM->SetElem(ir, ic, workGM->GetElem(ir, ic));
//        }
      }
      theBlas.Free(*workGM);
      delete workGM;
      workGM = newM;
    }
    // check if order has been changed (pivoting): update permutations and the
    // already computed cols. of the G matrix (note, piv is set to be itr before 
    // looking for maximum diagonals at the end)
    if ( itr!=piv ) {
      // swap data indices in the permutation vector (will be used at the Kernel eval.)
      size_t indx = fPermutationVect[piv];
      fPermutationVect[piv] = fPermutationVect[itr];
      fPermutationVect[itr] = indx;
      // update already computed rows of G according to this change in the cols.
      // i.e. the itr-th and piv-th cols of G
      for (size_t ir=0; ir<=itr; ++ir) {
        T dum = workGM->GetElem(ir, piv);
        workGM->SetElem(ir, piv, workGM->GetElem(ir, itr));
        workGM->SetElem(ir, itr, dum);
      }
      // swap diagonals as well
      T dum = diagGG[itr];
      diagGG[itr] = diagGG[piv];
      diagGG[piv] = dum;
    }
    // calculate the itr-th row of the G matrix (piv is the index of the selected input data):
    // - G_{itr,j}    = 0 for all j<itr (<= R is upper triangular: projections to the previous q-vectors)
    // - G_{itr, itr} = 0 i.e. j=itr  is the sqrt{diagGG[piv]} where diagGG[piv] is the orthogonal projection = K_{piv,piv} - sum_{t=1}^{itr-1} R_{t,piv}^2
    // - G_{itr,j}    = K_{itr,j} - sum_{t=1}^{itr-1} R_{t,itr} R_{t,j} for all j>itr:N
    // 1. the diagonal = sqt(diagGG) where diagGG stores the squared orthogonal projections
    const T invDiag = 1./std::sqrt(diagGG[itr]);
    workGM->SetElem(itr, itr, 1./invDiag);
    // 2. compute each col > diag for the given row = itr
    const TInputD* pivInpData = fInputDataM->GetPtrToBlock(fPermutationVect[itr]);
    // reset some variables
    resid     = 0.;
    piv       = itr+1; // set to be the next data point
    maxDiag   = 0.;   
    for (size_t ic=itr+1; ic<numInpData; ++ic) {
      // (input data should be continuos along its dimension in memory)
      const T kernelM = fKernel->Evaluate(pivInpData, fInputDataM->GetPtrToBlock(fPermutationVect[ic]), dimInpData);
      // sum of R_t,itr x R_t,ic for 
      T dum = kernelM;
      for (size_t ir=0; ir<itr; ++ir) { // matrix G is col major i.e. along a col in memory
        dum -= workGM->GetElem(ir, ic)*workGM->GetElem(ir, itr);
      }
      dum *= invDiag;
      // set the G_itr,ic element of G
      workGM->SetElem(itr, ic, dum);
      // update the  corresponding diagnal element of the approximated kenel matrix
      diagGG[ic] -= dum*dum;
      // find maximal diagonal i.e. point with maximum residual norm
      if (diagGG[ic] > maxDiag) {
        maxDiag = diagGG[ic];
        piv = ic;
      }
      // compute sum of residual (should we use this or individual?)
      resid += diagGG[ic];
    
    }
    // normalise the sum residual i.e. trace{approximated kernel matrix} = sum eigenvalues_i
    resid /= numInpData;
    // increase iteration counter
    ++itr;
  }  
  // Decomposition is scompleted here.
  //
  // form the incomplete Cholesky matrix member (after the decomopsition is done)
  if (fICholM) {
    theBlas.Free(*fICholM);
    delete fICholM;
  }

  // itr x numInpData matrix or its traspose if transpose=true
  fICholM = transpose ? new Matrix<T>(numInpData, itr) : new Matrix<T>(itr, numInpData);
  theBlas.Malloc(*fICholM);
  if (transpose) {
    // rows of fICholM are the N, (Rx1)^T vectors
    // it's a bit slower since we canntt go alonge cache lines
    // note: fICholM is in col-major order so depending on if transpose was set 
    //       to true or not 
    //   - traspose=false : fICholM is RxN (R<N) and col-continuos -> alonge the R projections
    //   - traspose=true  : fICholM is NxR (R<N) and col-continuos -> alonge the components of the R projections
    for (size_t ic=0; ic<numInpData; ++ic) { // col to row
      for (size_t ir=0; ir<itr; ++ir) {
        fICholM->SetElem(ic, ir, workGM->GetElem(ir, ic));
      }
    }
  } else {
    // cols of fICholM are the N, (Rx1) vectors
    // this is the same as generated: each cols can be copied since they are continuos in memory
    for (size_t ic=0; ic<numInpData; ++ic) { // col to col
      std::memcpy(fICholM->GetPtrToBlock(ic), workGM->GetPtrToBlock(ic), sizeof(T)*itr);
    }
  }
  // free working memory
  theBlas.Free(*workGM);
  delete workGM;
  // set the final approximation error as the trace{K-G^TG}/N i.e. sum of the 
  // eignevalues of K-G^TG per sum of the eigenvalues of K (assuming norm. ker. func.)
  fFinalResidual = resid;
}

