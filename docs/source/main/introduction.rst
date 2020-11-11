Introduction
=============

The :math:`\texttt{leuven}` library provides a very simple **framework** for the 
:math:`\texttt{C/C++}` implementation of 
machine learning, optimisation related and other algorithms, that relies heavily  
on manipulating (multi dimensional) data by using linear algebra. The framework
provides a **lightweight matrix** object for representing and dealing with 
array like (e.g. vector, matrix) data and **generic interfaces to different**
**BLAS/LAPACK linear algebra libraries** operating on such matrix objects. 
Thanks to these, array like data, with generic types and flexible memory 
layouts can be manipulated and transferred easily and efficiently between 
different modules while a high level of robustness can be guaranteed.

Beyond the framework, the library already contains the implementation of some 
machine learning algorithms based on the functionalities provided by the framework. 
This means that the different components of a given algorithm, or required by the 
algorithm, have already been implemented and available when installing the library. 
These components can be used later by a user to put together a related application. 
Therefore, the :math:`\texttt{leuven}` library also serves as a *toolkit* for 
these algorithms.

Example:
  the library contains the implementations of all the functionalities 
  and components that one needs to use in an incomplete Cholesky factorisation 
  based, sparse Kernel Spectral Clustering (KSC) application: implementation of kernels; 
  class to perform the incomplete Cholesky factorisation of a kernel matrix; class  
  for training, hyper parameter tuning and out-of-sample extension of a KSC model; 
  KSC model evaluation criteria; etc.. In other words, all the ingredients that are 
  needed for putting together the corresponding KSC application. All these are 
  available after the installation of the :math:`\texttt{leuven}` library.
  
Moreover, the :math:`app/example` directory contains example implementation 
of applications that can be built and used after the :math:`\texttt{leuven}` 
library has already been installed. These are complete, independent 
applications, provided by the developers as examples.  

Example:
  the :math:`\texttt{app/example/KSC}` directory contains the above mentioned 
  incomplete Cholesky decomposition based, sparse KSC applications for training, 
  hyper parameter tuning and out-of-sample extensions of a related clustering 
  model. After building and installing the :math:`\texttt{leuven}` library, these 
  applications, provided by the developers as example applications, can be built 
  and used directly to solve real clustering problems. Each such example sub-directory 
  is an independent entity i.e. has its own documentation, can be copied, built,
  installed and used separately (as long as the :math:`\texttt{leuven}` library is 
  installed on the system).


Finally, it should be noted, that the :math:`\texttt{leuven}` library/framework/toolkit 
has been developed primarily to provide a flexible but performant environment for 
the implementation of some machine learning algorithms formulated based on the 
LS-SVM framework and I used during my Master of Artificial Intelligence master 
thesis project.

.. _mymatrix_doc:

On the lightweight matrix implementation
-----------------------------------------

A **lightweight matrix** implementation is available in the :math:`\texttt{leuven}` 
library in order to facilitate dealing with array like (i.e. vector, matrix) data. 
The provided matrix object *stores the memory location of the array like data* 
together with its main descriptive fields such as number of rows, columns, etc.. 
The related memory management (i.e. memory allocation and de-allocation) is 
delegated to auxiliary methods such that the memory that stores the data can reside 
both on the CPU or on the GPU memory (see more at the :ref:`leuven_CUDA_support` section).
The matrix implementation also provides the possibility to store **any data types**, 
**both** in **row- and column major memory layouts**.

The implementation hides all the related differences by using template 
metaprogramming (combined with Curiously Recurring Template Pattern (CRTP). This 
means, that **the provided matrix class is templated on the data type** to be stored 
(e.g. :math:`\texttt{double, float, int, etc..}`) **and on the desired memory layout** 
(e.g. row- or columns major order). The main reasons why this implementation 
technique has been used are

 - ``performance``: template metaprogramming provides the possibility of using generic 
   data types while coding an algorithm such that the type(s) determined only at 
   compile time. Moreover, the CRTP makes possible 
   to move from run time (e.g. based on virtual methods) to compile 
   time polymorphism. These gives the maximum flexibility regarding both the data 
   types and the related methods together with **no run time overhead** since 
   everything is determined at compile time (e.g. no run time overhead caused by 
   virtual methods or branching).
       
 - ``compile time guarantee of consistency``: template metaprogramming has an additional 
   advantage over the elimination of several run time overhead. Template 
   specialisation makes possible to control what methods (i.e. with what 
   template argument combinations) are available after the compilation. This 
   provides the possibility to restrict the provided functionalities, available 
   behind a given common interface method, only to those combination of the input 
   (template) arguments of the interface that are allowed. This technique is used 
   in the :math:`\texttt{leuven}` library for example: 
   
    - to prevent the wrong combination of matrix memory layout and BLAS/LAPACK 
      implementation memory layout: depending on the selected BLAS/LAPACK 
      implementation and its ability to deal with both row- and/or column-major memory 
      layouts, methods (behind the common interfaces) capable to operate on row- 
      and/or column-major order data may or may not exists and this is determined 
      automatically at compile time depending on the selected BLAS/LAPACK option.
      
    - to provide data type independent BLAS/LAPACK interface methods: the framework  
      provides BLAS/LAPACK interface methods that are common for both 
      :math:`\texttt{double}` and :math:`\texttt{float}` data types (stored in 
      the provided lightweight matrix object). This is also implemented by using 
      template specialisation to ensures that the appropriate (and only the 
      appropriate) type dependent version of the given BLAS/LAPACK implementation 
      routines are called. Moreover, this appropriate routines are bounded at 
      compile time i.e. without having any type dependent run time conditions to 
      determine which routine should be invoked.

   It should be noted, that the above template specialisations guarantee the 
   exclusive existence of the appropriate methods i.e. those and only those 
   methods are available (regarding data type, memory layout combinations) 
   behind the common interfaces that makes sense in a given build configuration 
   (e.g. with a selected BLAS/LAPACK implementation version).
   This provides a compile time guarantee of excluding any non appropriate 
   combinations data type and/or memory layouts between the user code and that 
   allowed by the given BLAS/LAPACK implementation. Note, that the compile time 
   guarantee actually reveals in the form of compile time error in the user code 
   (that relies on the framework) containing *non-existing method* messages 
   with non-appropriate combinations of template arguments.
   
   
On the BLAS/LAPACK support
---------------------------

The :math:`\texttt{leuven}` library relies heavily on numerical linear algebra 
routines provided by the BLAS/LAPACK specifications. Therefore, an implementation 
of these specifications (preferably an optimised one) needs to be available on
the system. The user can also select and specify the location of the required 
BLAS/LAPACK implementation to be used by the :math:`\texttt{leuven}` library 
as a configuration option (see more at the :ref:`install_doc` section).

.. _leuven_CUDA_support:

On the CUDA support
--------------------

Beyond the CPU implementations of the BLAS/LAPACK specifications, GPU versions 
are also available. The :math:`\texttt{leuven}` library provides the possibility 
of using the NVIDIA CUDA implementations of these specifications 
:cite:`cuda,cuBLAS,cuSOLVER` by the help of  

 - ``CPU/GPU specific memory management``: while the :ref:`lightweight matrix implementation <mymatrix_doc>`,
   provided by the :math:`\texttt{leuven}` library, **stores the location of the memory**
   of the corresponding array like data, the **management of this memory** (i.e. 
   allocation and de-allocation) **is delegated to auxiliary methods**. These auxiliary 
   methods have different implementations in the GPU and CPU BLAS interface objects 
   (see below).  
 
 - the ``BLAS object``: the :math:`\texttt{leuven}` library provides interfaces 
   to the BLAS/LAPACK routines collected in the ``BLAS`` class as public methods.
   While interfaces, to the CPU BLAS/LAPACK implementations, are available through 
   a ``BLAS`` object, the optional GPU implementations can be accessed through 
   a ``BLAS_gpu`` object. **The name as well as the signature of the interface 
   methods provided by the ``BLAS`` and ``BLAS_gpu`` classes are the same**! 
   These classes contain the implementation of the above mentioned auxiliary 
   methods for memory handling (and memory copies for GPU) as well.

The ``CUDA support`` can be enabled at configuration time by using the 
:math:`\texttt{-DUSE}\_\texttt{CUBLAS=ON}` :math:`\texttt{CMake}` configuration 
option. 

Example:
  To perform the :math:`C = AB` matrix multiplication (as a special case of the general 
  :math:`C = \alpha A^T B^T + \beta C` with the optional transposes) on the CPU (using 
  :math:`\texttt{double}` precision data types) 
  
  .. code-block:: c++
  
    // Create matrix objects `A` (m x k), `B` (k x n) and `C` (m x n)
    // Note: the default, i.e. column-major memory layout is used here. One could
    //       create the matrix `A` to store its data in row-major order instead 
    //       as Matrix <double, false> A(m,k)  
    Matrix<double> A(m,k);
    Matrix<double> B(k,n);
    Matrix<double> C(m,n);

    // Allocate the corresponding memory
    // Note: matrix memory is managed by the `BLAS` object on the CPU
    BLAS theBlas;
    theBlas.Malloc(A);
    theBlas.Malloc(B);
    theBlas.Calloc(C);

    // fill in matrices `A` and `B`
    ...
    
    // Invoke the `dgemm/sgemm` general matrix-matrix multiplication implementation 
    // provided by the CPU BLAS option
    // Note: the appropriate method (`dgemm` for `double` and `sgemm` for `float`) 
    //       is automatically selected behind the interface at compile time!
    theBlas.XGEMM(A, B, C);
    
    // The result is available in matrix `C` at this point
    
    // De-allocate all the previously allocated memory
    theBlas.Free(A);
    theBlas.Free(B);
    theBlas.Free(C);

   
  While the same can be done on the GPU with GPU BLAS support as
  
  .. code-block:: c++ 

    // Create matrix objects `A` (m x k), `B` (k x n) and `C` (m x n)
    // Note: data will be stored both on the CPU and GPU memories
    //  - first on the CPU memory: same as before
    Matrix<double> A(m,k);
    Matrix<double> B(k,n);
    Matrix<double> C(m,n);
    //  - then on the GPU memory: same as above! (`_d` stands for `device`)
    Matrix<double> A_d(m,k);
    Matrix<double> B_d(k,n);
    Matrix<double> C_d(m,n);

    // Allocate the corresponding memory (both on CPU and GPU memory)
    //  - first the CPU: matrix memory is managed by the `BLAS` object on the CPU
    BLAS theBlas;
    theBlas.Malloc(A);
    theBlas.Malloc(B);
    theBlas.Calloc(C);

    //  - then on the GPU: matrix memory is managed by the `BLAS_gpu` object on 
    //                     the GPU
    BLAS_gpu  theBlas_gpu; 
    theBlas_gpu.Malloc(A_d);
    theBlas_gpu.Malloc(B_d);
    theBlas_gpu.Calloc(C_d);


    // fill in matrices `A` and `B` on the CPU
    ...
    
    // Copy the data represented by matrix `A` and `B` stored on the CPU memory
    // from the CPU to the GPU memory (note: from-to order!).
    theBlas_gpu.CopyToGPU(A, A_d);
    theBlas_gpu.CopyToGPU(B, B_d);
    
    // Invoke the `dgemm/sgemm` general matrix-matrix multiplication implementation 
    // provided by the cuBLAS GPU BLAS option
    // Note: the appropriate method (`dgemm` for `double` and `sgemm` for `float`) 
    //       is automatically selected behind the interface at compile time!
    theBlas_gpu.XGEMM(A_d, B_d, C_d);    
        
    // Copy the result(data) represented by matrix `C` stored on the GPU memory
    // from the GPU to the CPU memory (note: from-to order!).
    theBlas_gpu.CopyFromGPU(C_d, C);
    
    // The result is available in matrix `C` at this point
    
    // De-allocate all the previously allocated memory (both on the CPU and GPU)
    // - first on the CPU
    theBlas.Free(A);
    theBlas.Free(B);
    theBlas.Free(C);    
    // - then on the GPU
    theBlas_gpu.Free(A_d);
    theBlas_gpu.Free(B_d);
    theBlas_gpu.Free(C_d);


  Note, that the same (name, signature) interface methods needs to be invoked both 
  in case of the CPU and the GPU BLAS/LAPACK implementations with data reside either 
  on the CPU or on the GPU memory (with the difference, that data needs to be copied 
  between the two memories in case of GPU that is simplified by the provided 
  interface method). See more details and the a compete :math:`\texttt{BLAS::D/SGEMM}`
  test example under the :math:`leuven/apps/tests/BlasLapack/xgemm` directory.
  
  
  