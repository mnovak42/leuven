


get_property(CPU_BLAS_LIBRARY        GLOBAL PROPERTY  CPU_BLAS_LIBRARY_PROP)
get_property(CPU_BLAS_INCLUDE_DIR    GLOBAL PROPERTY  CPU_BLAS_INCLUDE_DIR_PROP)

get_property(CUDA_LIBRARIES          GLOBAL PROPERTY CUDA_LIBRARIES_PROP)
get_property(CUDA_INCLUDE_DIRS       GLOBAL PROPERTY CUDA_INCLUDE_DIRS_PROP)
get_property(CUDA_NVCC_FLAGS_leuven  GLOBAL PROPERTY CUDA_NVCC_FLAGS_leuven_PROP)
get_property(CPU_BLAS_INCLUDE_DIR    GLOBAL PROPERTY CPU_BLAS_INCLUDE_DIR_PROP)

get_property(leuven_INCLUDE_DIR      GLOBAL PROPERTY leuven_INCLUDE_DIR_PROP)

set (UNIT_TESTS "xgemm" "xgeqrf" "xgesvd" "xsyevr" "xsysv")

if (USE_CUBLAS)
  enable_language(CUDA)
endif ()

foreach (tname ${UNIT_TESTS})
  message(STATUS ${tname})    
  add_executable (test_${tname} ${CMAKE_SOURCE_DIR}/apps/tests/BlasLapack/${tname}/test_${tname}.cc)
  add_test       (${tname}  test_${tname} )
  # 
  target_include_directories (test_${tname} PUBLIC "${leuven_INCLUDE_DIR}")
  if (USE_CUBLAS)
    target_link_libraries (test_${tname} ${PROJECTNAME} ${CUDA_LIBRARIES})
    target_compile_options (test_${tname} PUBLIC "-DON_GPU=ON")
  else ()
    target_link_libraries (test_${tname} ${PROJECTNAME})
  endif ()
endforeach ()



add_executable (test_xgeqrf_v2 ${CMAKE_SOURCE_DIR}/apps/tests/BlasLapack/xgeqrf/test_xgeqrf_v2.cu)
add_test       (xgeqrf_v2  test_xgeqrf_v2 )
# 
target_include_directories (test_xgeqrf_v2 PUBLIC "${leuven_INCLUDE_DIR}")
if (USE_CUBLAS)
  target_link_libraries (test_xgeqrf_v2 ${PROJECTNAME} ${CUDA_LIBRARIES})
  target_compile_options (test_xgeqrf_v2 PUBLIC "-DON_GPU=ON")
  
  set_property(TARGET test_xgeqrf_v2 PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif ()
