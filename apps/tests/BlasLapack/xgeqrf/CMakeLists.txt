#-------------------------------------------------------------------------------
# Set cmake minimum version
#
cmake_minimum_required (VERSION 3.1.0 FATAL_ERROR)


#-------------------------------------------------------------------------------
# Define LIBNAME as the library name which we are dependent on here 
#
set (LIBNAME "leuven")


#-------------------------------------------------------------------------------
# Find the library 
#
find_package (${LIBNAME} CONFIG REQUIRED)
if (NOT ${LIBNAME}_FOUND)
  message (FATAL_ERROR "${LIBNAME} was not found")
endif ()


#-------------------------------------------------------------------------------
# Set the CXX flags that were used when building the library
#
set (CMAKE_CXX_FLAGS "${${LIBNAME}_CMAKE_CXX_FLAGS}")


#-------------------------------------------------------------------------------
# We need Fortran compiler (linking) in case of FBLAS-wrapper
#
if (FORTRAN_COMPILER_REQUIRED)
  enable_language (Fortran)
endif (FORTRAN_COMPILER_REQUIRED)


#-------------------------------------------------------------------------------
# Option to use the GPU (only if the lib. was built with USE_CUBLAS) in the test
#
option(ON_GPU "Option to make the test on GPU" OFF)
if (USE_CUBLAS AND ON_GPU)
   add_definitions (-DON_GPU)
   enable_language (CUDA)
   # use the same nvcc flags that were used to build the library
   set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS_Leuven})
endif (USE_CUBLAS AND ON_GPU)


#-------------------------------------------------------------------------------
# Locate sources and headers for this project
#
include_directories (${PROJECT_SOURCE_DIR}/include 
                     ${${LIBNAME}_INCLUDE_DIR})
file (GLOB headers ${PROJECT_SOURCE_DIR}/*.h)


#-------------------------------------------------------------------------------
# Add the executable, and link it to the ${LIBNAME} libraries
#
add_executable (test_xgeqrf  test_xgeqrf.cc  ${headers})
target_link_libraries (test_xgeqrf  ${${LIBNAME}_LIBRARIES})
if (USE_CUBLAS AND ON_GPU)
  add_executable (test_xgeqrf_v2 test_xgeqrf_v2.cc cuKers.cu ${headers})
  set_property(TARGET test_xgeqrf_v2 PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries (test_xgeqrf_v2 ${${LIBNAME}_LIBRARIES})
endif (USE_CUBLAS AND ON_GPU)


#-------------------------------------------------------------------------------
# In case of fortran compiler i.e. when FBLAS wrapper was used to build the lib.
#
if (FORTRAN_COMPILER_REQUIRED)
  set_property(TARGET test_xgeqrf PROPERTY LINKER_LANGUAGE Fortran)
  if (USE_CUBLAS AND ON_GPU)
    set_property(TARGET test_xgeqrf_v2 PROPERTY LINKER_LANGUAGE Fortran)
  endif (USE_CUBLAS AND ON_GPU)
endif (FORTRAN_COMPILER_REQUIRED)
