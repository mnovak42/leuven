## Can be used either with the CBLAS or the FBLAS interface that will depend how 
## the WTF toolkit was built 
## In case the WTF toolkit was built with CUDA-support, it can also be used with 
## CUBLAS interface. 
#
# Note, that in case of CBLAS wrapper both col- and row-major order is supported 
# without any run-time performance penalty while FBALS and CUBLAS requires 
# strictly col-major ordering. This is determined by the second template 
# parameter of the 'Matrix' object.
#
# This test will perform the general matrix matrix multiplication:
#  C = alpha AB + beta C  by using the BLAS XBLAS wrapper where X- either 'D' or 
# for double or 'S' for float. This later is automatic since it depends on the 
# first template parameter of the 'Matrix' objects AB and C involved. 
#
#
# default build: i.e. with CPU (C/F-BLAS) support
cmake ../ -DWTF_DIR=/Users/mnovak/opt/xx/lib/cmake/WTF/
#
#
# CPU (C/F-BLAS) and GPU (CUBLAS) support 
cmake ../ -DWTF_DIR=/Users/mnovak/opt/xx/lib/cmake/WTF/  -DON_GPU=ON


