################################################################################
# A simple cmake script to locate the reference NETLIB BLAS implementation 
# library (libblas,liblapack). The following variables will be set if the NETLIB
# BLAS implementation library was found:
# NETLIB_BLAS_FOUND
# NETLIB_BLAS_LIBRARY
################################################################################


################################################################################
# Set locations where the NETLIB BLAS library can be located 
Set (NETLIB_BLAS_SEARCH_LIBRARY_PATHS
  ${NETLIB_BLAS_DIR}/lib
  ${NETLIB_BLAS_DIR}/lib64
  $ENV{NETLIB_BLASROOT}cd
  $ENV{NETLIB_BLASROOT}/lib
  $ENV{NETLIB_BLASROOT}/lib64
  /opt/BLAS/lib
  /usr/local/lib64
  /usr/local/lib
  /lib/blas
  /lib64/
  /lib/
  /usr/lib/blas
  /usr/lib/x86_64-linux-gnu
  /usr/lib64
  /usr/lib
  /usr/local/opt/blas/lib
)


################################################################################
# Try to find the BLAS, LAPACK library using the paths (in our order)
Find_library(NETLIB_BLAS_LIBRARY NAMES libblas blas PATHS ${NETLIB_BLAS_SEARCH_LIBRARY_PATHS} NO_DEFAULT_PATH)
Find_library(NETLIB_LAPACK_LIBRARY NAMES liblapack lapack PATHS ${NETLIB_BLAS_SEARCH_LIBRARY_PATHS} NO_DEFAULT_PATH)


set (NETLIB_BLAS_LIBRARY 
  ${NETLIB_BLAS_LIBRARY} 
  ${NETLIB_LAPACK_LIBRARY}
)


set(NETLIB_BLAS_FOUND TRUE)

################################################################################
# Check the library was found
if (NOT NETLIB_BLAS_LIBRARY)
    set (NETLIB_BLAS_FOUND FALSE)
endif ()


MARK_AS_ADVANCED (
  NETLIB_BLAS_LIBRARY
  NETLIB_BLAS
)