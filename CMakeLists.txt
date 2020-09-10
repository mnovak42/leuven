################################################################################
## Setup the project
################################################################################

## ----------------------------------------------------------------------------
## Minimum required CMake version, project name, CMake text colours, etc...
## ----------------------------------------------------------------------------

# required CMake version
cmake_minimum_required (VERSION 3.3.0 FATAL_ERROR)

# set the project name 
SET (PROJECTNAME "leuven")
SET (LIBNAME "lib${PROJECTNAME}")
project (${PROJECTNAME}) #  LANGUAGES CXX CUDA)

# find available thread libraries and prefer pthreads 
set (CMAKE_THREAD_PREFER_PTHREAD TRUE)
set (THREADS_PREFER_PTHREAD_FLAG TRUE) 
find_package (Threads)

# ensure out of source build 
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
   message (FATAL_ERROR 
     "You don't want to configure in the source directory! CMake don't recommend"
     "to do so. Create a 'build' sub-directory, change to and build from that!"
   )
endif ()

# set some colours for CMake text colouring.
if (NOT WIN32)
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


## ----------------------------------------------------------------------------
## Set build type dependent compiler options and default RELEASE build type
## ----------------------------------------------------------------------------

if (NOT CMAKE_BUILD_TYPE)
  set (CMAKE_BUILD_TYPE Release CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebug RelWithDebInfo MinSizeRel."
    FORCE)
endif (NOT CMAKE_BUILD_TYPE)

if (CMAKE_BUILD_TYPE STREQUAL "" AND NOT CMAKE_CXX_FLAGS MATCHES "-O[123]")
   message (STATUS "WARNING! It seems you are compiling without optimization. Please set CMAKE_BUILD_TYPE.")
endif (CMAKE_BUILD_TYPE STREQUAL "" AND NOT CMAKE_CXX_FLAGS MATCHES "-O[123]")

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall -std=c++11 -pthread")
#set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG -W -Wall -O3 -std=c++11 -pthread -march=native -Rpass=loop-vectorize -Rpass-analysis=loop-vectorize")
set (CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG -W -Wall -O3 -std=c++11 -pthread -march=native")
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

set (CMAKE_CXX_FLAGS "${leuven_CMAKE_CXX_FLAGS}")

# print the compiler flags used
message(STATUS "CXX compiler flags: ${CMAKE_CXX_FLAGS}")

# set the directory where the libraries will be placed 
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)


## ----------------------------------------------------------------------------
## The user needs to chose a CPU BLAS implementation available on their system.
## Optionally, the user can also add the GPU (CUDA) implementation of the BLAS 
## library (on top of the CPU one). These will be done in the 'utils'.
## ----------------------------------------------------------------------------

add_subdirectory (utils)


## ----------------------------------------------------------------------------
## The 'lssvm' directory contains implementations of some Least Square - Support 
## Vector Machine (LS-SVM) algorithms with all their required components. Note,
## these are toolkit-like implementations i.e. there no complete applications 
## (executables) available here. The developer provides such applications as 
## examples in the 'app/examples' directory.
## ----------------------------------------------------------------------------

add_subdirectory (lssvm)


## ----------------------------------------------------------------------------
## Add testing option (OFF by default)
## ----------------------------------------------------------------------------

option (BUILD_TESTS "Build test programs" OFF)
if (BUILD_TESTS)
  message (STATUS "Building test programs is enabled!")
  message (STATUS "${Green}===== Add testing ===== ${ColourReset}")
  enable_testing ()
  add_subdirectory (testing)
endif ()
