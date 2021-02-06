################################################################################
## Testing of the leuven library and framework
################################################################################

if (${CMAKE_VERSION} VERSION_GREATER "3.17.0")
  cmake_policy( SET CMP0104 OLD )
endif ()

## ----------------------------------------------------------------------------
## Retrieve configuration i.e. dependencies locations set at the end of 'utils'
## ----------------------------------------------------------------------------

# CPU BLAS/LAPACK libraries and possible include dirs
get_property(CPU_BLAS_LIBRARY        GLOBAL PROPERTY  CPU_BLAS_LIBRARY_PROP)
get_property(CPU_BLAS_INCLUDE_DIR    GLOBAL PROPERTY  CPU_BLAS_INCLUDE_DIR_PROP)

# CUDA libraries, include dirs and nvcc compiler flags
get_property(CUDA_LIBRARIES          GLOBAL PROPERTY CUDA_LIBRARIES_PROP)
get_property(CUDA_INCLUDE_DIRS       GLOBAL PROPERTY CUDA_INCLUDE_DIRS_PROP)
get_property(CUDA_NVCC_FLAGS_leuven  GLOBAL PROPERTY CUDA_NVCC_FLAGS_leuven_PROP)
get_property(CPU_BLAS_INCLUDE_DIR    GLOBAL PROPERTY CPU_BLAS_INCLUDE_DIR_PROP)

# all (i.e. the leuven, CPU and CUDA BLAS/LAPACK) include directories
get_property(leuven_INCLUDE_DIR      GLOBAL PROPERTY leuven_INCLUDE_DIR_PROP)

# Fortran required
get_property(leuven_FORTRAN_REQUIRED GLOBAL PROPERTY leuven_FORTRAN_REQUIRED_PROP)

if (leuven_FORTRAN_REQUIRED)
  enable_language(Fortran)
endif()


## ----------------------------------------------------------------------------
## Find available thread libraries and prefer pthreads
##
set (CMAKE_THREAD_PREFER_PTHREAD TRUE)
set (THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package (Threads REQUIRED)


## ----------------------------------------------------------------------------
## Add some unit tests for CPU/CUDA BLAS/LAPACK interface testing
## ----------------------------------------------------------------------------

set (UNIT_TESTS "xgemm" "xgeqrf" "xgesvd" "xorgqr" "xsyevr" "xsysv" )

if (USE_CUBLAS)
  enable_language(CUDA)
endif ()

foreach (tname ${UNIT_TESTS})
  message(STATUS ${tname})
  add_executable (test_${tname} ${CMAKE_SOURCE_DIR}/apps/tests/BlasLapack/${tname}/test_${tname}.cc)
  add_test       (${tname}  test_${tname} )
  #
  target_include_directories (test_${tname} PUBLIC "${leuven_INCLUDE_DIR}")
  if (leuven_FORTRAN_REQUIRED)
    set_property(TARGET test_${tname} PROPERTY LINKER_LANGUAGE Fortran)
  endif ()
  if (USE_CUBLAS)
    target_link_libraries (test_${tname} leuven;${CUDA_LIBRARIES} Threads::Threads)
#    target_link_libraries (test_${tname} PRIVATE leuven CUDA_LIBS)
    if ( NOT (${tname} STREQUAL "xorgqr") )
      target_compile_options (test_${tname} PUBLIC "-DON_GPU=ON")
    endif ()
  else ()
    target_link_libraries (test_${tname} leuven Threads::Threads)
  endif ()
endforeach ()


if (USE_CUBLAS)

  file (GLOB headers ${CMAKE_SOURCE_DIR}/apps/tests/BlasLapack/xgeqrf/*.h)
  add_executable (test_xgeqrf_v2 ${CMAKE_SOURCE_DIR}/apps/tests/BlasLapack/xgeqrf/test_xgeqrf_v2.cc ${CMAKE_SOURCE_DIR}/apps/tests/BlasLapack/xgeqrf/cuKers.cu ${headers})
  add_test       (xgeqrf_v2  test_xgeqrf_v2)
  #
  target_include_directories (test_xgeqrf_v2 PUBLIC "${leuven_INCLUDE_DIR}")
  if (leuven_FORTRAN_REQUIRED)
    set_property(TARGET test_xgeqrf_v2 PROPERTY LINKER_LANGUAGE Fortran)
  endif ()
#  if (USE_CUBLAS)
    target_link_libraries (test_xgeqrf_v2 leuven ${CUDA_LIBRARIES} Threads::Threads)
    target_compile_options (test_xgeqrf_v2 PUBLIC "-DON_GPU=ON")

    set_property(TARGET test_xgeqrf_v2 PROPERTY CUDA_SEPARABLE_COMPILATION ON)
#  endif ()


  file (GLOB headers ${CMAKE_SOURCE_DIR}/apps/tests/BlasLapack/xorgqr/*.h)
  add_executable (test_xorgqr_gpu ${CMAKE_SOURCE_DIR}/apps/tests/BlasLapack/xorgqr/test_xorgqr.cc ${CMAKE_SOURCE_DIR}/apps/tests/BlasLapack/xorgqr/cuKers.cu ${headers})
  add_test       (xorgqr_gpu  test_xorgqr_gpu)
  #
  target_include_directories (test_xorgqr_gpu PUBLIC "${leuven_INCLUDE_DIR}")
  if (leuven_FORTRAN_REQUIRED)
    set_property(TARGET test_xorgqr_gpu PROPERTY LINKER_LANGUAGE Fortran)
  endif ()
  #  if (USE_CUBLAS)
    target_link_libraries (test_xorgqr_gpu leuven ${CUDA_LIBRARIES} Threads::Threads)
    target_compile_options (test_xorgqr_gpu PUBLIC "-DON_GPU=ON")

    set_property(TARGET test_xorgqr_gpu PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  #  endif ()


endif ()
