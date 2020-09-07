################################################################################
# A simple cmake script to locate ATLAS blas, lapack implementation and headers
# It will set the following variables if ATLAS root directly is set properly:
# AtlasBLAS_FOUND 
# AtlasBLAS_LIBRARY
# AtlasBLAS_INCLUDE_DIR
################################################################################


################################################################################
# Set locations where the AtlasBLAS include directory and library can be located 
Set (AtlasBLAS_SEARCH_INCLUDE_DIR_PATHS
  ${AtlasBLAS_DIR}/include
  $ENV{AtlasBLASROOT}
  $ENV{AtlasBLASROOT}/include
  /opt/AtlasBLAS/include
  /usr/local/include/atlas
  /usr/include/atlas
  /usr/local/include/atlas-base
  /usr/include/atlas-base
  /usr/include/x86_64-linux-gnu
  /usr/local/include
  /usr/include
  /usr/local/opt/atlas/include
)

Set (AtlasBLAS_SEARCH_LIBRARY_PATHS
  ${AtlasBLAS_DIR}/lib
  ${AtlasBLAS_DIR}/lib64
  $ENV{AtlasBLASROOT}cd
  $ENV{AtlasBLASROOT}/lib
  $ENV{AtlasBLASROOT}/lib64
  /opt/AtlasBLAS/lib
  /usr/local/lib64
  /usr/local/lib
  /lib/atlas-base
  /lib64/
  /lib/
  /usr/lib/atlas-base
  /usr/lib/x86_64-linux-gnu
  /usr/lib64
  /usr/lib
  /usr/local/opt/atlas/lib
)


################################################################################
# Try to find the AtlasBLAS include directory and AtlasBLAS library using the paths
Find_path (AtlasBLAS_INCLUDE_DIR NAMES cblas.h PATHS ${AtlasBLAS_SEARCH_INCLUDE_DIR_PATHS} NO_DEFAULT_PATH)
Find_library(AtlasBLAS_LIBRARY_atlas NAMES atlas PATHS ${AtlasBLAS_SEARCH_LIBRARY_PATHS} NO_DEFAULT_PATH)
Find_library(AtlasBLAS_LIBRARY_lapack NAMES lapack PATHS ${AtlasBLAS_SEARCH_LIBRARY_PATHS} NO_DEFAULT_PATH)
Find_library(AtlasBLAS_LIBRARY_cblas NAMES cblas PATHS ${AtlasBLAS_SEARCH_LIBRARY_PATHS} NO_DEFAULT_PATH)
Find_library(AtlasBLAS_LIBRARY_ptcblas NAMES ptcblas PATHS ${AtlasBLAS_SEARCH_LIBRARY_PATHS} NO_DEFAULT_PATH)
Find_library(AtlasBLAS_LIBRARY_f77blas NAMES f77blas ptf77blas PATHS ${AtlasBLAS_SEARCH_LIBRARY_PATHS} NO_DEFAULT_PATH)
Find_library(AtlasBLAS_LIBRARY_ptf77blas NAMES ptf77blas PATHS ${AtlasBLAS_SEARCH_LIBRARY_PATHS} NO_DEFAULT_PATH)


set(AtlasBLAS_LIBRARY 
  ${AtlasBLAS_LIBRARY_atlas}
  ${AtlasBLAS_LIBRARY_lapack} 
  ${AtlasBLAS_LIBRARY_cblas}
  ${AtlasBLAS_LIBRARY_ptcblas}
  ${AtlasBLAS_LIBRARY_f77blas}
  ${AtlasBLAS_LIBRARY_ptf77blas}
 )


set(AtlasBLAS_FOUND TRUE)
set(AtlasBLAS_INCLUDE_FOUND TRUE)

################################################################################
# Check include files
if (NOT AtlasBLAS_INCLUDE_DIR)
    set (AtlasBLAS_INCLUDE_FOUND FALSE)
    if (USE_CBLAS_WRAPPER AND AtlasBLAS_LIBRARY)
      message (FATAL_ERROR
      "\nAtlasBLAS library could be found at ${AtlasBLAS_LIBRARY}. However, the  "
      " AtlasBLAS include directory, required when using the CBLAS wrapper, was "
      " not found! You still can use this AtlasBLAS library but only through the "
      " FBLAS wrapper: set the cmake option USE_CBLAS_WRAPPER=OFF explicitly.\n")
    endif ()
endif ()


################################################################################
# Check the library
if (NOT AtlasBLAS_LIBRARY)
    set (AtlasBLAS_FOUND FALSE)
Endif ()


MARK_AS_ADVANCED (
  AtlasBLAS_INCLUDE_DIR
  AtlasBLAS_LIBRARY
  AtlasBLAS
)