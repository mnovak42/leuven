// 
// The common implementation part of CBLAS and FBALS interfaces. 
// This file is included both into CBLAS.hh and FBLAS.hh
//


#if USE_CBLAS_WRAPPER
  typedef CBLAS XBLAS; 
#else 
  typedef FBLAS XBLAS;
#endif

//  - for aligned memory allocation in case of MKL BLAS implementation
#if USE_MKL_BLAS
  #include "mkl_service.h"
  #include "mkl_cblas.h"
#elif  USE_OPEN_BLAS
extern "C" {
  #include "cblas.h"
}
#endif 

template < class T, bool is>
void XBLAS::Malloc(Matrix<T,is>& m) {
#if USE_MKL_BLAS
  m.SetDataPtr((T *)mkl_malloc( m.GetNumRows()*m.GetNumCols()*sizeof( T ), kMKLAlignment ));
#else
  m.SetDataPtr((T *)malloc( m.GetNumRows()*m.GetNumCols()*sizeof( T ) ));
#endif
}

template < class T, bool is>
void XBLAS::Calloc(Matrix<T,is>& m) {
#if USE_MKL_BLAS
  m.SetDataPtr((T *)mkl_calloc( m.GetNumRows()*m.GetNumCols(), sizeof( T ), kMKLAlignment ));
#else
  m.SetDataPtr((T *)calloc( m.GetNumRows()*m.GetNumCols(), sizeof( T ) ));
#endif
}

template < class T, bool is>
void XBLAS::Free(Matrix<T,is>& m) {
#if USE_MKL_BLAS
  if (m.GetDataPtr()) {
    mkl_free((void*)m.GetDataPtr());
    m.SetDataPtr(nullptr);
  }
#else
  if (m.GetDataPtr()) {
    free((void*)m.GetDataPtr());
    m.SetDataPtr(nullptr);
  }
#endif
}
