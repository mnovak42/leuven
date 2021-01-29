#-------------------------------------------------------------------------------
# Set cmake minimum version
#
cmake_minimum_required (VERSION 3.1.0 FATAL_ERROR)
project(test-matrixRW)


#-------------------------------------------------------------------------------
# Find the library
#
find_package (leuven CONFIG REQUIRED)
if (NOT leuven_FOUND)
  message (FATAL_ERROR "leuven was not found")
endif ()


#-------------------------------------------------------------------------------
# Find available thread libraries and prefer pthreads
#
set (CMAKE_THREAD_PREFER_PTHREAD TRUE)
set (THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package (Threads REQUIRED)


#-------------------------------------------------------------------------------
# Set the CXX flags that were used when building the library
#
set (CMAKE_CXX_FLAGS "${leuven_CMAKE_CXX_FLAGS}")


#-------------------------------------------------------------------------------
# We need Fortran compiler (linking) in case of FBLAS-wrapper
#
if (FORTRAN_COMPILER_REQUIRED)
  enable_language(Fortran)
endif (FORTRAN_COMPILER_REQUIRED)


#-------------------------------------------------------------------------------
# Option to use the GPU (only if the lib. was built with USE_CUDA) in the test
#
option(ON_GPU "Option to make the test on GPU" OFF)
if (USE_CUBLAS AND ON_GPU)
   add_definitions(-DON_GPU)
endif (USE_CUBLAS AND ON_GPU)


#-------------------------------------------------------------------------------
# Locate sources and headers for this project
#
include_directories (${PROJECT_SOURCE_DIR}/include
                     ${leuven_INCLUDE_DIR})
file (GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cc)
file (GLOB headers ${PROJECT_SOURCE_DIR}/include/*.hh)


#-------------------------------------------------------------------------------
# Add the executable, and link it to the leuven libraries
#
add_executable (test_RWMatrix test_RWMatrix.cc ${sources} ${headers})
target_link_libraries (test_RWMatrix ${leuven_LIBRARIES} Threads::Threads)


#-------------------------------------------------------------------------------
# In case of fortran compiler i.e. when FBLAS wrapper was used to build the lib.
#
if (FORTRAN_COMPILER_REQUIRED)
  set_property(TARGET test_RWMatrix PROPERTY LINKER_LANGUAGE Fortran)
endif (FORTRAN_COMPILER_REQUIRED)
