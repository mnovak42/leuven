
#ifndef TYPES_HH
#define TYPES_HH

#include "definitions.hh"

//
// One BLAS Wrapper needs to be selected out of the two:
//  = FBLAS (USE_CBLAS_WRAPPER=FF): 
//      - direct call of the Fortran BLAS functions -> col-major order only! 
//      - can be used (linked) with any BLAS implementations 
//  = CBLAS (USE_CBLAS_WRAPPER=ON): (RECOMMENDED: if MKL BLAS or OpenBLAS are available)
//      - Fortran BLAS functions are called through the C-BLAS interface -> 
//        supported both col- and row-major order
//      - can be used (linked) only with limited BLAS implementations (that have
//        their C-BLAS interface (MKL, OpenBLAS)
//
// After selecting the BLAS Wrapper, one of the BLAS implementations needs to be 
// selected (among those that are available on the local system) for linking.
// The possible options are:
//  = NETLIB-BLAS (USE_NETLIB_BLAS): 
//      - the reference BLAS implementation that is available everywhere 
//      - doesn't have C-BlAS interface -> can be used only with the FBLAS 
//        wrapper 
//      - therefore, supports only COLUMN-MAJOR order 
//      - very slow compared to others 
//  = OpenBLAS (USE_OPEN_BLAS): 
//      - optimised BLAS implementation (alternative to the refernce NETLIB one)
//      - have its C-BLAS interface -> can be used either with the FBLAS or the 
//        CBLAS Wrappers -> supports both ROW- and COLUMN-MAJOR order when used 
//        with the CBLAS Wrapper
//      - supports multiple threads (MT), significantly faster than the NETLIB 
//        version (even in single thread mode)
//  = ATLAS-BLAS (USE_ATLAS_BLAS): 
//      - ....
//  = Intel MKL-BLAS (USE_MKL_BLAS):  (RECOMMENDED)
//      - optimised BLAS implementation (alternative to the refernce NETLIB one)
//      - have its C-BLAS interface -> can be used either with the FBLAS or the 
//        CBLAS Wrappers -> supports both ROW- and COLUMN-MAJOR order when used 
//        with the CBLAS Wrapper
//      - supports multiple threads (MT), significantly faster than the NETLIB 
//        version (even in single thread mode) and even a bit faster than the 
//        OpenBLAS implementation
//

// Chekc the CPU BLAS Wrapper (FBLAS, CBLAS)
#if USE_CBLAS_WRAPPER
  #pragma message(" Compiled with CBLAS CPU BLAS Wrapper (calls through the C_BLAS interface, both row- and col-major order)!")
  #if USE_MKL_BLAS
      #pragma message(" Compiled with MKL BLAS support!")
  #elif  USE_OPEN_BLAS
      #pragma message(" Compiled with OpenBlas BLAS support!")
  #elif  USE_ATLAS_BLAS
      // ATLAS is special because it has the full C-BLAS interface but it has only 
      // partial LAPACK implementation and do not have the LAPACKE C-interface for
      // the full LAPACK. So for the simplicity, we do not support the CBLAS-wrapper 
      // when ATLAS BLAS-LAPACK is used as CPU option. 
      #error "ATLAS BLAS do not have the LAPACKE C-interface! Use the FBLAS Wrapper!"
  #elif  USE_NETLIB_BLAS
      // NETLIB BLAS do not have C-BLAS interface so it cannot be used with CBLAS
      // Wrapper. Only the FBLAS Wrapper can be used with NETLIB BLAS (the order 
      // must be fixed to column-major).
      #error "NETLIB BLAS do not have C-BLAS interface! Use the FBLAS Wrapper!"
  #else
      #pragma message(" **** Undefined CPU BLAS implementation!") 
      #error  UNDEFINED (CPU) BLAS IMPLEMENTATION SEE THE BLAS OTIONS IN THE DOC
  #endif
  #include "CBLAS.hh"
  typedef CBLAS BLAS;
#else  // USE_CBLAS_WRAPPER = OFF
    #pragma message(" Compiled with FBLAS CPU BLAS Wrapper (direct Fortran calls, col-major order)!")
    // Check the BLAS implementation (MKL BLAS, OpenBLAS, NETLIB BLAS)
    #if USE_MKL_BLAS
        #pragma message(" Compiled with MKL BLAS support!")
    #elif  USE_OPEN_BLAS
        #pragma message(" Compiled with OpenBlas BLAS support!")
    #elif  USE_ATLAS_BLAS
        #pragma message(" Compiled with AtlasBlas BLAS support!")
    #elif  USE_NETLIB_BLAS
        #pragma message(" Compiled with NETLIB BLAS support!")
    #else
        #pragma message(" **** Undefined CPU BLAS implementation!") 
        #error  UNDEFINED (CPU) BLAS IMPLEMENTATION SEE THE BLAS OTIONS IN THE DOC
    #endif
    #include "FBLAS.hh"
    typedef FBLAS BLAS;  
#endif

// The selected GPU (CUDA) BLAS implementation (optional)
#if USE_CUBLAS 
    #include "CUBLAS.hh"
    typedef CUBLAS BLAS_gpu;
    #pragma message(" Compiled with CUBLAS GPU BLAS support!")
#else
    #pragma message(" Compiled WITHOUT GPU BLAS support!")
#endif 


#endif