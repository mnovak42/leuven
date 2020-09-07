################################################################################
# Setup the project
################################################################################
cmake_minimum_required (VERSION 3.3.0 FATAL_ERROR)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
   message(FATAL_ERROR "You don't want to configure in the source directory!")
endif ()

SET (PROJECTNAME "leuven")
SET (LIBNAME "lib${PROJECTNAME}")
project (${PROJECTNAME}) #  LANGUAGES CXX CUDA)



if(NOT WIN32)
  string(ASCII 27 Esc)
  set(ColourReset "${Esc}[m")
  set(ColourBold  "${Esc}[1m")
  set(Red         "${Esc}[31m")
  set(Green       "${Esc}[32m")
  set(Yellow      "${Esc}[33m")
  set(Blue        "${Esc}[34m")
  set(Magenta     "${Esc}[35m")
  set(Cyan        "${Esc}[36m")
  set(White       "${Esc}[37m")
  set(BoldRed     "${Esc}[1;31m")
  set(BoldGreen   "${Esc}[1;32m")
  set(BoldYellow  "${Esc}[1;33m")
  set(BoldBlue    "${Esc}[1;34m")
  set(BoldMagenta "${Esc}[1;35m")
  set(BoldCyan    "${Esc}[1;36m")
  set(BoldWhite   "${Esc}[1;37m")
endif()

################################################################################
# Set default CMAKE_BUILD_TYPE to Release:
################################################################################
if (NOT CMAKE_BUILD_TYPE)
  set (CMAKE_BUILD_TYPE Release CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebug RelWithDebInfo MinSizeRel."
    FORCE)
endif (NOT CMAKE_BUILD_TYPE)

if (CMAKE_BUILD_TYPE STREQUAL "" AND NOT CMAKE_CXX_FLAGS MATCHES "-O[123]")
   message (STATUS "WARNING! It seems you are compiling without optimization. Please set CMAKE_BUILD_TYPE.")
endif (CMAKE_BUILD_TYPE STREQUAL "" AND NOT CMAKE_CXX_FLAGS MATCHES "-O[123]")




set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall -std=c++11 -pthread")
#set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG -W -Wall -O3 -std=c++11 -pthread -march=native -Rpass=loop-vectorize -Rpass-analysis=loop-vectorize")
set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG -W -Wall -O3 -std=c++11 -pthread -march=native")
set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -W -Wall -O0 -std=c++11 -pthread -g")


if (NOT CMAKE_BUILD_TYPE)
    set (leuven_CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
elseif (CMAKE_BUILD_TYPE MATCHES Release) 
    set (leuven_CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELEASE}")
elseif (CMAKE_BUILD_TYPE MATCHES Debug) 
    set (leuven_CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_DEBUG}")
else ()
    set (leuven_CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELEASE}")
endif (NOT CMAKE_BUILD_TYPE)



###
## The user needs to chose a CPU BLAS implementation available on their system.
## The recommended CPU BLAS implementation is the Intel MKL one but others, like 
## OpenBLAS or the reference NETLIB BLAS implementation are also supported and 
## can be selected with the appropriate cmake option (more at CPU BLAS options). 
##
## Optionally, the user can select GPU (CUDA) implementation of the BLAS library
## in combination of the CPU one using one of the appropriate cmake options.
## THESE are all col-major 

#### NOTES :
#
# MKL will be located based on the MKLROOT env variable (set at MKL INSTALL)
#
# FindCUDA should bot be there but if it is there then it is used so we need 
#    to set -DCUDA_TOOLKIT_ROOT_DIR=/Developer/NVIDIA/CUDA-10.1 i.e. the same 
#    that CUDA install will add to the ~/.bash_profile without the /bin ending.


# We might need a fortran compiler as well in case of NETLIB BLAS (on MAC sure)
# Fortran compiler required (FORTRAN_REQUIRED) will be set in the generated
# cmakeConfig.cmake if the application need fortran compiler (or linking) (i.e. 
# in case all cases except MKL).



################################################################################
add_subdirectory (utils)
add_subdirectory (lssvm)

