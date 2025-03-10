// ############################################################################
// Input configuration file for cmake that will generate the definitions.hh file 
// from it with the actual configuration options and put the generated file into   
// ../inc/global.hh. This file will contain all the selected build configuration
// options and guaranties consistency when the library is used later by an other 
// project. DO NOT MODIFY THIS FILE (IT WILL BE REGENERATED AT EACH BUIL)
//##############################################################################

#ifndef DEFINITIONS_HH
#define DEFINITIONS_HH

// for size_t definition
#include <cstddef>
// for C print(f)outs
#include <cstdio>
// for assert
#include <cassert>

// use std::min and std::max
#include <algorithm>
//#define min(x,y) (((x) < (y)) ? (x) : (y))
//#define max(x,y) (((x) < (y)) ? (y) : (x))

#define _unused(x) ((void)(x))


// cmake configuration options: cmake will set the proper values
#define USE_MKL_BLAS 0
#define USE_OPEN_BLAS 1
#define USE_ATLAS_BLAS 0
#define USE_NETLIB_BLAS 0
#define USE_CBLAS_WRAPPER 0
#define USE_CUBLAS 1
#define CONFIG_VERBOSE 0

#if USE_CBLAS_WRAPPER
  const bool kRowMajorPossible = true;
#else 
  const bool kRowMajorPossible = false;
#endif

const bool kColMajorOrder = true;

const int  kNumThreads = 3;


#endif // DEFINITIONS_HH
