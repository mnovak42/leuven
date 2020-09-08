#-------------------------------------------------------------------------------
# Set cmake minimum version
#
cmake_minimum_required (VERSION 3.1.0 FATAL_ERROR)


#-------------------------------------------------------------------------------
# Define LIBNAME as the library name which we are dependent on here 
#
SET (LIBNAME "leuven")


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
  enable_language(Fortran)
endif (FORTRAN_COMPILER_REQUIRED)


#-------------------------------------------------------------------------------
# Option to use the GPU (only if the lib. was built with USE_CUDA) in the test
#
option(ON_GPU "Option to make the test on GPU" OFF)
if (USE_CUBLAS AND ON_GPU)
   add_definitions(-DON_GPU)
endif (USE_CUBLAS AND ON_GPU)

message (STATUS "${leuven_LIBRARIES}")

#-------------------------------------------------------------------------------
# Locate sources and headers for this project
#
include_directories (${PROJECT_SOURCE_DIR}/include 
                     ${${LIBNAME}_INCLUDE_DIR})
file (GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cc)
file (GLOB headers ${PROJECT_SOURCE_DIR}/include/*.hh)


#-------------------------------------------------------------------------------
# Add the executable, and link it to the ${LIBNAME} libraries
#
add_executable (test_xgemm test_xgemm.cc ${sources} ${headers})
target_link_libraries (test_xgemm ${${LIBNAME}_LIBRARIES})


#-------------------------------------------------------------------------------
# In case of fortran compiler i.e. when FBLAS wrapper was used to build the lib.
#
if (FORTRAN_COMPILER_REQUIRED)
  set_property(TARGET test_xgemm PROPERTY LINKER_LANGUAGE Fortran)
endif (FORTRAN_COMPILER_REQUIRED)

