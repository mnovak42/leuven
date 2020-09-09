################################################################################
# Add source, header files & include directories to the leuven framework.
################################################################################

## ----------------------------------------------------------------------------
# Add some project sources headers and set include directories
## ----------------------------------------------------------------------------
file (GLOB CXXHEADERS ${CMAKE_SOURCE_DIR}/utils/inc/global/*.hh)
include_directories (${CMAKE_SOURCE_DIR}/utils/inc ${CMAKE_SOURCE_DIR}/utils/inc/global)
message (STATUS "${Green}===== WRAPPER ===== ${ColourReset}")
option (USE_CBLAS_WRAPPER "Cmake option to select the C-BLAS capable CPU BLAS Wrapper." OFF)
if (USE_CBLAS_WRAPPER)
  file (GLOB CXXSOURCES ${CMAKE_SOURCE_DIR}/utils/src/CBLAS.cc)
  list (APPEND CXXHEADERS ${CMAKE_SOURCE_DIR}/utils/inc/CBLAS.hh ${CMAKE_SOURCE_DIR}/utils/inc/XBLAS.tpp ${CMAKE_SOURCE_DIR}/utils/inc/CBLAS.tpp)
  message (STATUS "Building with the CBLAS Wrapper")
else ()
  file (GLOB CXXSOURCES ${CMAKE_SOURCE_DIR}/utils/src/FBLAS.cc)
  list (APPEND CXXHEADERS ${CMAKE_SOURCE_DIR}/utils/inc/FBLAS.hh ${CMAKE_SOURCE_DIR}/utils/inc/XBLAS.tpp ${CMAKE_SOURCE_DIR}/utils/inc/FBLAS.h ${CMAKE_SOURCE_DIR}/utils/inc/FBLAS.tpp)
  message (STATUS "Building with the FBLAS Wrapper")
endif ()
message ("")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")


################################################################################
# CPU BLAS options:
# ------------------------------------------------------------------------------
# A BLAS implementation is needed and the user can specify one of the 
# followings (depending on which of them are available on their system):
# - Intel (R) MKL with -DCPU_BLAS_OPTION=USE_MKL_BLAS option (RECOMMENDED)
# - OpenBlas with the -DCPU_BLAS_OPTION=USE_OPEN_BLAS option
# - ATLAS with the -DCPU_BLAS_OPTION=USE_ATLAS_BLAS option
# - the reference NETLIB BLAS with -DCPU_BLAS_OPTION=USE_NETLIB_BLAS option
# The last option is defined by default. The reason of this is that actually not
# the version of the BLAS implementation itself is important but the fact if the
# implementation does have or doesn't have C-BLAS interface because it determine 
# the wrapper (FBLAS or CBLAS) used in the project. Note, that the FBLAS wrapper 
# can handle any BLAS implementations since it makes direct calls to the Fortran 
# BLAS interfaces. Among the 4 options listed, NETLIB BLAS is the one that will 
# select the FBLAS wrapper (since it doesn't have C-BLAS interface) so it can be 
# used as a "wild card" option fo using any BLAS implementations different than 
# the above 4 options (number of threads depend on what the actual BLAS is).  
################################################################################

## ------------------------------------------------------------------------------
## Define options for (explicit) CPU BLAS/LAPACK
## ------------------------------------------------------------------------------

unset (CPU_BLAS_LIBRARY CACHE)

# CPU BLAS option selection:
unset (USE_MKL_BLAS CACHE)
unset (USE_OPEN_BLAS CACHE)
unset (USE_ATLAS_BLAS CACHE)
unset (USE_NETLIB_BLAS CACHE)
#unset (USE_ANY_BLAS CACHE)
#
option (CPU_BLAS_OPTION "The selected CPU BLAS implementation")
SET (CPU_BLAS_OPTION_VALUES "USE_MKL_BLAS" "USE_OPEN_BLAS" "USE_ATLAS_BLAS" "USE_NETLIB_BLAS")
SET (CPU_BLAS_OPTION_STRING_VALUES "MKL-BLAS" "OpenBLAS" "ATLAS-BLAS" "NETLIB-BLAS")
if (NOT CPU_BLAS_OPTION)
  set (CPU_BLAS_OPTION_INDEX 3)
else ()
  LIST (FIND CPU_BLAS_OPTION_VALUES ${CPU_BLAS_OPTION} CPU_BLAS_OPTION_INDEX)
endif ()
#
if (${CPU_BLAS_OPTION_INDEX} EQUAL -1)
  message (FATAL_ERROR "Option ${CPU_BLAS_OPTION} not supported, valid entries are ${CPU_BLAS_OPTION_VALUES}")
elseif (${CPU_BLAS_OPTION_INDEX} EQUAL 0)
  option (USE_MKL_BLAS "Use Intel(R) MKL BLAS implementation for CPU BLAS support" ON)
elseif (${CPU_BLAS_OPTION_INDEX} EQUAL 1)
  option (USE_OPEN_BLAS "Use OpenBLAS BLAS implementation for CPU BLAS support" ON)
elseif (${CPU_BLAS_OPTION_INDEX} EQUAL 2)
  option (USE_ATLAS_BLAS "Use ATLAS BLAS implementation for CPU BLAS support" ON)
else ()
  option (USE_NETLIB_BLAS "Use NETLIB BLAS implementation for CPU BLAS support" ON)
endif ()
#
list (GET CPU_BLAS_OPTION_STRING_VALUES ${CPU_BLAS_OPTION_INDEX} CPU_BLAS_OPTION_STRING) 
message (STATUS "${Green}==== The selected CPU BLAS Option =${ColourReset} ${CPU_BLAS_OPTION_STRING} ${Green} ==== ${ColourReset}")
message (STATUS " ")
#
 
################################################################################
# MKL BLAS as optional CPU BLAS implementation (good for both CBLAS and FBLAS).
# MKL BLAS will be searched at location based on (in the order):
# - the MKL_ROOT_DIR cmake variable (set to the dir where mkl include, lib, etc)
# - the MKLROOT environment variable 
# - try /opt/intel/mkl location
################################################################################

if (USE_MKL_BLAS AND NOT CPU_BLAS_LIBRARY)
  message (STATUS "${Green}========  MKL ======= ${ColourReset}")
  find_package (MKL REQUIRED)
  if (MKL_FOUND)
    add_definitions (-DUSE_MKL_BLAS)
#    message (STATUS "Found MKL at ${MKLROOT_PATH}")
    include_directories (${MKL_INCLUDE_DIR})
    set (CPU_BLAS_INCLUDE_DIR ${MKL_INCLUDE_DIR})
    set (CPU_BLAS_LIBRARY ${MKL_LIBRARIES})
    message (STATUS "MKL INCLUDE DIRECTORY = ${MKL_INCLUDE_DIR}")
    message (STATUS "MKL LIBRARY DIRECTORY = ${MKL_LIBRARY_DIR}")
    message (STATUS "MKL LIBRARIES = ${MKL_LIBRARIES}")
    message (STATUS " ")
  else (MKL_FOUND)
    message(FATAL_ERROR "REQUIRED INTEL MKL LIBRARIES/HEADER WERE NOT FOUND")
  endif (MKL_FOUND)  
endif (USE_MKL_BLAS AND NOT CPU_BLAS_LIBRARY)


################################################################################
# OpenBLAS as optional CPU BLAS implementation (good for both CBLAS and FBLAS).
#
# If you installed openBLAS manually, set the OpenBLAS_DIR cmake variable to 
# point to this install directory where the include, lib and share directories 
# are located.
# 
# OpenBLAS will be searched at location based on (in the order):
# - the OpenBLAS_DIR cmake variable (set to the dir where include, lib, share)
# - then the OpenBLASROOT environment variable (the dir where include, lib, etc)
# - many standard locations as (/usr/lib; /opt//usr/local/lib; etc.)
# Note, that when OpenBLAS is located with this code, it is assumed that the 
# OpenBLAS is supposed to be used through the CBLAS wrapper so OpenBlas include 
# directory is also necessary to find (for the C-BLAS interface header). In case 
# the OpenBlas include directory is not found an explicit error will be generated
# and user will advice to use the FBLAS wrapper instead.  
################################################################################

if (USE_OPEN_BLAS AND NOT CPU_BLAS_LIBRARY)
  message (STATUS "${Green}========  OpenBLAS ======= ${ColourReset}") 
  find_package (OpenBLAS REQUIRED)
  if (OpenBLAS_FOUND)
    add_definitions (-DUSE_OPEN_BLAS)
#    message (STATUS "Found OpenBLAS at ${OpenBLAS_DIR}")
    include_directories (${OpenBLAS_INCLUDE_DIR})
    set (CPU_BLAS_INCLUDE_DIR ${OpenBLAS_INCLUDE_DIR})
    set (CPU_BLAS_LIBRARY ${OpenBLAS_LIBRARY})
    message (STATUS "OpenBLAS INCLUDE DIRECTORY = ${OpenBLAS_INCLUDE_DIR}")
    message (STATUS "OpenBLAS LIBRARY = ${OpenBLAS_LIBRARY}")
    message(STATUS " ")
  else (OpenBLAS_FOUND)
    message(FATAL_ERROR "REQUIRED OpenBLAS LIBRARY/HEADER WERE NOT FOUND")
  endif (OpenBLAS_FOUND)
endif (USE_OPEN_BLAS AND NOT CPU_BLAS_LIBRARY)


################################################################################
# USE_ATLAS_BLAS as optional CPU BLAS implementation.
#
# Note, that since the C-interface of Atlas LAPACK don't follow the LAPACKE C-
# interface, we do not use ATLAS with the CBLAS-wrapper. Therefore, only col-
# major matrix order is supported (although the CBLAS part follows the proper 
# C-BLAS interface)  
# ATLAS BLAS will be searched at location based on (in the order):
# - the AtlasBLAS_DIR cmake variable (set to the dir where the include and 
#   lib sub-directories are located)
# - then the AtlasBLASROOT environment variable (set as above)
# - many standard locations as (/usr/lib; /opt/usr/local/lib; etc.)
################################################################################

if (USE_ATLAS_BLAS AND NOT CPU_BLAS_LIBRARY)
  message (STATUS "${Green}========  ATLAS BLAS ======= ${ColourReset}") 
  find_package (AtlasBLAS REQUIRED)
  message(STATUS " WAS FOUND = ${AtlasBLAS_FOUND}")
  if (AtlasBLAS_FOUND)
    add_definitions (-DUSE_ATLAS_BLAS)
#    include_directories (${AtlasBLAS_INCLUDE_DIR})
#    set (CPU_BLAS_INCLUDE_DIR ${AtlasBLAS_INCLUDE_DIR})
    set (CPU_BLAS_LIBRARY ${AtlasBLAS_LIBRARY})
#    message (STATUS "AtlasBLAS INCLUDE DIRECTORY = ${AtlasBLAS_INCLUDE_DIR}")
    message (STATUS "ATLAS BLAS LIBRARY = ${AtlasBLAS_LIBRARY}")
    message(STATUS " ")
  else (AtlasBLAS_FOUND)
    message(FATAL_ERROR "REQUIRED ATLAS BLAS LIBRARY WAS NOT FOUND")
  endif (AtlasBLAS_FOUND)
endif (USE_ATLAS_BLAS AND NOT CPU_BLAS_LIBRARY)


################################################################################
# USE_NETLIB_BLAS as optional CPU BLAS implementation (good only for FBLAS).
#
# NETLIB BLAS will be searched at location based on (in the order):
# - the NETLIB_BLAS_DIR cmake variable (set to the dir where the BLAS library 
#   file, libblas is located)
# - then the NETLIB_BLASROOT environment variable (set as above)
# - many standard locations as (/usr/lib; /opt//usr/local/lib; etc.)
# Note, that the NETLIB BLAS can only be used with the FBLAS wrapper since it
# doesn't have the a C-BLAS interface. So when this option is used together with
# the CBLAS wrapper option, an explicit error will be generated and the user is
# advised to use the FBLAS wrapper instead. 
################################################################################

if (USE_NETLIB_BLAS AND NOT CPU_BLAS_LIBRARY)
  message (STATUS "${Green}========  NETLIB BLAS (or any BLAS) ======= ${ColourReset}") 
  find_package (NETLIB_BLAS REQUIRED)
  message(STATUS " WAS FOUND = ${NETLIB_BLAS_FOUND}")
  if (NETLIB_BLAS_FOUND)
    add_definitions (-DUSE_NETLIN_BLAS)
    unset (CPU_BLAS_INCLUDE_DIR)
    set (CPU_BLAS_LIBRARY ${NETLIB_BLAS_LIBRARY})
    message (STATUS "NETLIB BLAS LIBRARY = ${NETLIB_BLAS_LIBRARY}")
    message(STATUS " ")
  else (NETLIB_BLAS_FOUND)
    message(FATAL_ERROR "REQUIRED NETLIB BLAS LIBRARY WAS NOT FOUND")
  endif (NETLIB_BLAS_FOUND)
endif (USE_NETLIB_BLAS AND NOT CPU_BLAS_LIBRARY)


## ----------------------------------------------------------------------------
## check if CPU BLAS library was found: if  CPU_BLAS_LIBRARY was set 
## check if CBLAS Wrapper can be used: if  CPU_BLAS_INCLUDE_DIR was set 
## check if the proper wrapper is used 
## ----------------------------------------------------------------------------

message (STATUS "${Green}===== CHECKING CPU BLAS WRAPPER AND LIBRARY CONSISTENCY =====${ColourReset}")
if (CPU_BLAS_LIBRARY)
  if (USE_CBLAS_WRAPPER AND NOT CPU_BLAS_INCLUDE_DIR)
    if (USE_ATLAS_BLAS)
      message (FATAL_ERROR 
      " ${BoldRed} **** ERROR- CPU BLAS Wrapper and library mismatch: \n${ColourReset}"
      "${ColourBold}CBLAS WRAPPER${ColourReset} WAS SELECTED BUT ONLY THE ${ColourBold}FBLAS-WRAPPER${ColourReset} IS SUPPORTED " 
      "IN CASE OF THE ${ColourBold}ATLAS-BLAS${ColourReset}. THIS IS BECAUSE, ALTHOUGH ATLAS HAS COMPLETE CBLAS INTERFACE " 
      "IT DOSEN'T HAVE THE ${ColourBold}LAPACKE C-INTERFACE${ColourReset}.\n"
      "SWITCH TO THE ${ColourBold}FBLAS WRAPPER${ColourReset} OR CHANGE CPU BLAS OPRION.")
    else (USE_ATLAS_BLAS)
      message (FATAL_ERROR 
      " ${BoldRed} **** ERROR- CPU BLAS Wrapper and library mismatch: \n${ColourReset}"
      "${ColourBold}CBLAS WRAPPER${ColourReset} WAS SELECTED BUT THE C-BLAS INTERFACE OF THE SELECTED BLAS "
      "IMPLEMENTATION ${ColourBold}${CPU_BLAS_OPTION_STRING}${ColourReset} COULD NOT BE FOUND."
      "SWITCH TO THE ${ColourBold}FBLAS WRAPPER${ColourBold} OR CHANGE CPU BLAS OPRION.")
    endif (USE_ATLAS_BLAS)
  endif (USE_CBLAS_WRAPPER AND NOT CPU_BLAS_INCLUDE_DIR)  
else (CPU_BLAS_LIBRARY)
  message (FATAL_ERROR
    "${BoldRed} **** ERROR- CPU BLAS library was not found: \n${ColourReset}" 
    "A CPU BLAS implementation, required by the project, was not set or found! See the Doc.!")
endif (CPU_BLAS_LIBRARY)
message (STATUS " ")


## ----------------------------------------------------------------------------
## See if Fortran compiler (linking) is needed for the application 
## ----------------------------------------------------------------------------

unset(FORTRAN_COMPILER_REQUIRED CACHE)
if (NOT USE_MKL_BLAS)
  set (FORTRAN_COMPILER_REQUIRED true)
endif (NOT USE_MKL_BLAS)
##message (STATUS "====> FORTAN ? = ${FORTRAN_COMPILER_REQUIRED}")



################################################################################
# GPU BLAS options: 
# ------------------------------------------------------------------------------
# Optional ....can be cuBLAS only at the moment
#
################################################################################


################################################################################
# cuBLAS as optional GPU BLAS implementation.
# The CUDA toolkit (including cuBLAS) location is normally added to the PATH 
# that will be searched by the FindCUDA.cmake script. In case CUDA is not found,
# one can specify the CUDA toolkit location by the CUDA_TOOLKIT_ROOT_DIR. See 
# more details in the CMake/Modules/FindCUDA.cmake file.
################################################################################

option (USE_CUBLAS "Use NVIDIA cuBLAS implementation for GPU BLAS support" OFF)
if (USE_CUBLAS)
  message (STATUS "${Green}========  CUDA, cuBLAS ======= ${ColourReset}")
  find_package (CUDA 6.0 REQUIRED)
  if (CUDA_FOUND)
    add_definitions (-DUSE_CUBLAS)
    message (STATUS "Found CUDA at ${CUDA_TOOLKIT_ROOT_DIR} (${CUDA_VERSION_STRING})")
    include_directories (AFTER SYSTEM ${CUDA_INCLUDE_DIRS})
    message (STATUS "CUDA INCLUDE DIRECTORY = ${CUDA_INCLUDE_DIRS}")
    message (STATUS "CUDA LIBRARIES = ${CUDA_LIBRARIES}")
    message (STATUS "CUDA CUBLAS LIBRARIES = ${CUDA_CUBLAS_LIBRARIES}")
    message (STATUS "CUDA cudart static LIBRARY = ${CUDA_cudart_static_LIBRARY}")
    message (STATUS "CUDA cusolver LIBRARY = ${CUDA_cusolver_LIBRARY}")
    message (STATUS " ")
  else (CUDA_FOUND)
    message(FATAL_ERROR "REQUIRED CUDA AND/OR cuBLAS LIBRARIES WERE NOT FOUND")
  endif (CUDA_FOUND)  
endif (USE_CUBLAS)


################################################################################
## Setting up the library with the selected CPU/GPU BLAS/LAPACK configurations
## and install 
################################################################################

## ----------------------------------------------------------------------------
## Generate "definitions.hh" configuration file and add to the include files:
## - this will contain all the CMake configuration options used at build time
## ----------------------------------------------------------------------------

configure_file (${CMAKE_SOURCE_DIR}/cmake/definitions.hh.in
    ${CMAKE_CURRENT_SOURCE_DIR}/inc/global/definitions.hh @ONLY)
list (APPEND CXXHEADERS ${CMAKE_SOURCE_DIR}/utils/inc/global/definitions.hh)



## ----------------------------------------------------------------------------
## Setting up the library
## ----------------------------------------------------------------------------

message (STATUS "${Green}===== Setting up the ${PROJECTNAME} library =====${ColourReset}")
if (USE_CUBLAS)
  enable_language(CUDA)
  SET (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} 
       -Xcompiler 
       -fPIC 
       -Xcudafe 
       --diag_suppress=boolean_controlling_expr_is_constant 
       --disable-warnings
       -lineinfo"
  )
  file (GLOB CUDASOURCES ${CMAKE_SOURCE_DIR}/utils/src/CUBLAS.cu)
  list (APPEND CXXHEADERS ${CMAKE_SOURCE_DIR}/utils/inc/CUBLAS.hh ${CMAKE_SOURCE_DIR}/utils/inc/CUBLAS.tpp)
  #
  cuda_add_library(${PROJECTNAME} STATIC ${CXXSOURCES} ${CUDASOURCES})
  set_property(TARGET ${PROJECTNAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(${PROJECTNAME} ${CPU_BLAS_LIBRARY} ${CUDA_LIBRARY} 
                      ${CUDA_cudart_static_LIBRARY} ${CUDA_CUBLAS_LIBRARY} 
                      ${CUDA_cusolver_LIBRARY})
  set_target_properties(${PROJECTNAME} PROPERTIES COMPILE_FLAGS ${CMAKE_CXX_FLAGS})                      
else ()
  add_library(${PROJECTNAME} STATIC ${CXXSOURCES})
  target_link_libraries(${PROJECTNAME} ${CPU_BLAS_LIBRARY}) 
  set_target_properties(${PROJECTNAME} PROPERTIES COMPILE_FLAGS ${CMAKE_CXX_FLAGS})
endif ()

### ALSO for the NICER SOLUTION TO HANDLE DEPENDENCIES (see below)
# target_include_directories(${PROJECTNAME} 
#    # add CPU-BLAS include directory 
#    PUBLIC ${CPU_BLAS_INCLUDE_DIR}
#    # add CUDA include directory if any
#    PUBLIC ${CUDA INCLUDE DIRS}
#    # add the project include directory  
# #    PUBLIC $<INSTALL_INTERFACE:includes> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
#    )
#
message (" ")

## ----------------------------------------------------------------------------
## Install: project headers, library and export target (dependencies)
## ----------------------------------------------------------------------------

install (FILES ${CXXHEADERS} DESTINATION includes)
install (TARGETS ${PROJECTNAME} DESTINATION lib)
###
### A NICER SOLUTION TO HANDLE DEPENDENCIES: Exporting the target
### We will use this in the final project (the CMake/WTFConfig.cmake file must 
### be change as well in order to be able to used) Also, target_include_directories
### must be set to contain the dependency headers!
# install (TARGETS ${PROJECTNAME} DESTINATION lib EXPORT ${PROJECTNAME}Targets)
# install (EXPORT ${PROJECTNAME}Targets DESTINATION lib/cmake/${PROJECTNAME})
# export(PACKAGE ${PROJECTNAME})

configure_file (${CMAKE_SOURCE_DIR}/cmake/${PROJECTNAME}Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/${PROJECTNAME}Config.cmake @ONLY)

install (FILES
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/${PROJECTNAME}Config.cmake
    DESTINATION lib/cmake/${PROJECTNAME}
)


###############################################################################
##  Set some global properties for retrieving proper configuration at testing 
###############################################################################


# local directory in the build for the header files
FILE ( COPY ${CXXHEADERS} DESTINATION ${CMAKE_BINARY_DIR}/includes)
set (leuven_INCLUDE_DIR "${CMAKE_BINARY_DIR}/includes")

# CPU BLAS/LAPACK libraries and possible include dirs
set_property(GLOBAL PROPERTY CPU_BLAS_LIBRARY_PROP      "${CPU_BLAS_LIBRARY}")
set_property(GLOBAL PROPERTY CPU_BLAS_INCLUDE_DIR_PROP  "${CPU_BLAS_INCLUDE_DIR}")

# CUDA libraries, include dirs and nvcc compiler flags
set_property(GLOBAL PROPERTY CUDA_LIBRARIES_PROP   "${CUDA_LIBRARIES};${CUDA_cudart_static_LIBRARY};${CUDA_CUBLAS_LIBRARIES};${CUDA_cusolver_LIBRARY}")
set_property(GLOBAL PROPERTY CUDA_INCLUDE_DIRS_PROP  "${CUDA_INCLUDE_DIRS}")
set_property(GLOBAL PROPERTY CUDA_NVCC_FLAGS_leuven_PROP "${CUDA_NVCC_FLAGS}")

# all (i.e. the leuven, CPU and CUDA BLAS/LAPACK) include directories
set_property(GLOBAL PROPERTY leuven_INCLUDE_DIR_PROP  "${leuven_INCLUDE_DIR};${CPU_BLAS_INCLUDE_DIR};${CUDA_INCLUDE_DIRS}")

