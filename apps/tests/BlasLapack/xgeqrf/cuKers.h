
#ifndef CUKERS_H
#define CUKERS_H


#include <cuda_runtime.h>

// a is an M-by-N matrix with M>=N 
// b is an N-by-N matrix 
// will write the upper triangular of a into the upper triangualr of b
template <class T>
__global__
void GetUpperTriangular_2D(T* a, T* b, int m, int n);

template <class T>
void GetUpperTriangular(T* a_d, T* b_d, int m, int n);

#endif // CUKERS_H