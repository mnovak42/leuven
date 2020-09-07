################################################################################
# A simple cmake script to locate Intel Math Kernel Library
# It will set the following variables if MKL root directly is set properly:
# MKL_FOUND 
# MKLROOT_PATH
# MKL_INCLUDE_DIR
# MKL_LIBRARY_DIR
# MKL_LIBRARIES
# LIB_MKL_RT
# LIB_PTHREAD 
################################################################################


################################################################################
# Find the MKL root directory. Looks for 3 places in the following order:
# - the MKL_ROOT_DIR cmake variable
#	- the environment variable MKLROOT
#	- the directory /opt/intel/mkl
################################################################################
if (MKL_ROOT_DIR)
  set(MKLROOT_PATH ${MKL_ROOT_DIR})
elseif (DEFINED ENV{MKLROOT})
  set(MKLROOT_PATH $ENV{MKLROOT})
elseif (EXISTS "/opt/intel/mkl")
  set(MKLROOT_PATH "/opt/intel/mkl")
endif ()

set(MKL_FOUND FALSE)
if (MKLROOT_PATH) 
  set(MKL_FOUND TRUE)
endif ()


################################################################################
# Find include path and libraries. It will set (if exists):
# - MKL_INCLUDE_DIR, MKL_LIBRARY_DIR, MKL_LIBRARIES, LIB_MKL_RT, LIB_PTHREAD
################################################################################
if (MKL_FOUND)

  # = First set expected include, library and icc paths 
  set(EXPECT_MKL_INCPATH "${MKLROOT_PATH}/include")
  #
  if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(EXPECT_MKL_LIBPATH "${MKLROOT_PATH}/lib")
  endif (CMAKE_SYSTEM_NAME MATCHES "Darwin")
  #
  set(EXPECT_ICC_LIBPATH "$ENV{ICC_LIBPATH}")
  #
  if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    if (CMAKE_SIZEOF_VOID_P MATCHES 8)
      set(EXPECT_MKL_LIBPATH "${MKLROOT_PATH}/lib/intel64")
    else (CMAKE_SIZEOF_VOID_P MATCHES 8)
      set(EXPECT_MKL_LIBPATH "${MKLROOT_PATH}/lib/ia32")
    endif (CMAKE_SIZEOF_VOID_P MATCHES 8)
  endif (CMAKE_SYSTEM_NAME MATCHES "Linux")
  
  # = Check if the expected paths are exists 
  if (IS_DIRECTORY ${EXPECT_MKL_INCPATH})
	  set(MKL_INCLUDE_DIR ${EXPECT_MKL_INCPATH})
	endif (IS_DIRECTORY ${EXPECT_MKL_INCPATH})
	#
	if (IS_DIRECTORY ${EXPECT_MKL_LIBPATH})
		set(MKL_LIBRARY_DIR ${EXPECT_MKL_LIBPATH})
	endif (IS_DIRECTORY ${EXPECT_MKL_LIBPATH})
	
	# = find specific library files
	find_library(LIB_MKL_RT NAMES mkl_rt HINTS ${MKL_LIBRARY_DIR})
	find_library(LIB_PTHREAD NAMES pthread)	  
endif (MKL_FOUND)

set(MKL_LIBRARIES 
	${LIB_MKL_RT} 
	${LIB_PTHREAD})


################################################################################
# Deal with QUIET and REQUIRED argument
################################################################################
include(FindPackageHandleStandardArgs)
#
find_package_handle_standard_args(MKL DEFAULT_MSG 
    MKL_LIBRARY_DIR
    LIB_MKL_RT
    LIB_PTHREAD
    MKL_INCLUDE_DIR)
#    
mark_as_advanced(LIB_MKL_RT LIB_PTHREAD MKL_INCLUDE_DIR)

