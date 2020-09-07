################################################################################
# A simple cmake script to locate OpenBLAS blas implementation and headers
# It will set the following variables if OpenBLAS root directly is set properly:
# OpenBLAS_FOUND 
# OpenBLAS_LIBRARY
# OpenBLAS_INCLUDE_DIR
################################################################################


################################################################################
# Set locations where the OpenBLAS include directory and library can be located 
Set (OpenBLAS_SEARCH_INCLUDE_DIR_PATHS
  ${OpenBLAS_DIR}/include
  ${OpenBLAS_DIR}/include/openblas
  $ENV{OpenBLASROOT}
  $ENV{OpenBLASROOT}/include
  /opt/OpenBLAS/include
  /usr/local/include/openblas
  /usr/include/openblas
  /usr/local/include/openblas-base
  /usr/include/openblas-base
  /usr/include/x86_64-linux-gnu
  /usr/local/include
  /usr/include
  /usr/local/opt/openblas/include
)

Set (OpenBLAS_SEARCH_LIBRARY_PATHS
  ${OpenBLAS_DIR}/lib
  ${OpenBLAS_DIR}/lib64
  $ENV{OpenBLASROOT}cd
  $ENV{OpenBLASROOT}/lib
  $ENV{OpenBLASROOT}/lib64
  /opt/OpenBLAS/lib
  /usr/local/lib64
  /usr/local/lib
  /lib/openblas-base
  /lib64/
  /lib/
  /usr/lib/openblas-base
  /usr/lib/x86_64-linux-gnu
  /usr/lib64
  /usr/lib
  /usr/local/opt/openblas/lib
)


################################################################################
# Try to find the OpenBLAS include directory and OpenBLAS library using the paths
Find_path (OpenBLAS_INCLUDE_DIR NAMES cblas.h PATHS ${OpenBLAS_SEARCH_INCLUDE_DIR_PATHS} NO_DEFAULT_PATH)
Find_library(OpenBLAS_LIBRARY NAMES openblas PATHS ${OpenBLAS_SEARCH_LIBRARY_PATHS} NO_DEFAULT_PATH)


set(OpenBLAS_FOUND TRUE)
set(OpenBLAS_INCLUDE_FOUND TRUE)

################################################################################
# Check include files
if (NOT OpenBLAS_INCLUDE_DIR)
    set (OpenBLAS_INCLUDE_FOUND FALSE)
    if (USE_CBLAS_WRAPPER AND OpenBLAS_LIBRARY)
      message (FATAL_ERROR
      "\nOpenBLAS library could be found at ${OpenBLAS_LIBRARY}. However, the  "
      " OpenBLAS include directory, required when using the CBLAS wrapper, was "
      " not found! You still can use this OpenBLAS library but only through the "
      " FBLAS wrapper: set the cmake option USE_CBLAS_WRAPPER=OFF explicitly.\n")
    endif ()
endif ()


################################################################################
# Check the library
if (NOT OpenBLAS_LIBRARY)
    set (OpenBLAS_FOUND FALSE)
Endif ()


MARK_AS_ADVANCED (
  OpenBLAS_INCLUDE_DIR
  OpenBLAS_LIBRARY
  OpenBLAS
)