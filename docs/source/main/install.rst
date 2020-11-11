.. _install_doc:

Build and install
=================


The :math:`leuven` library provides a very simple **framework** for the 
:math:`\texttt{C/C++}` implementation of
machine learning, optimisation related and other algorithms, that relies heavily  
on manipulating (multi dimensional) data by using linear algebra. This section 
contains the list of dependences and provides detailed instructions on how to 
configure and build the library.


Requirements 
-------------

Building the :math:`\texttt{leuven}` library requires:

 - a :math:`\texttt{c/c++}` compiler, with :math:`\texttt{c++11}` support, to be installed on the system (even a Fortran 
   compiler is needed when other than Intel MKL is used as CPU BLAS/LAPACK option)
 - :math:`\texttt{CMake}` is used for managing (option configuration, locate 
   dependencies etc.) the build process, so it needs to be installed on the system
   (see at http://www.cmake.org/)
 - the :math:`\texttt{leuven}` library heavily relies on BLAS/LAPACK functionalities 
   so BLAS and LAPACK libraries must to be installed on the system. It is strongly 
   recommended to use one of the freely available optimised implementations 
   such as the Intel MKL, OpenBLAS or ATALS (see more below).


Quick (and dirty) start
------------------------

.. caution:: 
    While the following might work well, it gives very little or no control on what 
    BLAS/LAPACK implementation is pick up and will be used by the :math:`\texttt{leuven}` 
    library. Since the performance (as well as the provided flexibility) of the 
    library, strongly depends on the available and selected BLAS/LAPACK options,
    it is **strongly recommended to install and specify explicitly** the BLAS/LAPACK
    implementation related configurations as shown in the :ref:`blas_lapack_options_howto` 
    section.
 
When BLAS/LAPACK libraries are installed at one of the standard location of 
the system (e.g. :math:`\texttt{/usr/local/lib64, /usr/local/lib, /usr/lib64, /usr/lib, etc.}`,
one can skip the explicit specification of the required BLAS/LAPACK implementation 
during the :math:`cmake` configuration of the :math:`\texttt{leuven}` library 
since the required libraries will be searched under these standard locations 
automatically in this case. So one can perform the following steps from the 
:math:`\texttt{leuven}` main directory :: 

    bash-3.2$ mkdir build

to create a :math:`\texttt{build}` directory where all the configuration and 
build related objects, files will be placed ::

    bash-3.2$ cd build/
    bash-3.2$ cmake -DCMAKE_INSTALL_PREFIX=/Users/mnovak/opt/leuven1 ../
    -- The C compiler identification is AppleClang 10.0.0.10001145
    -- The CXX compiler identification is AppleClang 10.0.0.10001145
    -- Check for working C compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc
    -- Check for working C compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc -- works
    -- Detecting C compiler ABI info
    -- Detecting C compiler ABI info - done
    -- Detecting C compile features
    -- Detecting C compile features - done
    -- Check for working CXX compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++
    -- Check for working CXX compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ -- works
    -- Detecting CXX compiler ABI info
    -- Detecting CXX compiler ABI info - done
    -- Detecting CXX compile features
    -- Detecting CXX compile features - done
    -- ===== WRAPPER ===== 
    -- Building with the FBLAS Wrapper

    -- ==== The selected CPU BLAS Option = NETLIB-BLAS  ==== 
    --  
    -- ========  NETLIB BLAS (or any BLAS) ======= 
    --  WAS FOUND = TRUE
    -- NETLIB BLAS LIBRARY = /usr/lib/libblas.dylib;/usr/lib/liblapack.dylib
    --  
    -- ===== CHECKING CPU BLAS WRAPPER AND LIBRARY CONSISTENCY =====
    --  
    -- ===== Setting up the leuven library =====
     
    -- ===== Adding the lssvm part ===== 

    -- Configuring done
    -- Generating done
    
change directory to the previously created build and generate the make files with 
the given configurations (:math:`\texttt{-DCMAKE}\_\texttt{INSTALL}\_\texttt{PREFIX}`
:math:`\texttt{cmake}` configuration option specifies the location where the final 
product will be installed e.g. my :math:`\texttt{/Users/mnovak/opt/leuven1}`
directory in this case) ::

    bash-3.2$ make 
    Scanning dependencies of target leuven
    [ 50%] Building CXX object utils/CMakeFiles/leuven.dir/src/FBLAS.cc.o
    [100%] Linking CXX static library ../lib/libleuven.a
    [100%] Built target leuven

to build the library and :: 

    bash-3.2$ make install
    [100%] Built target leuven
    Install the project...
    -- Install configuration: "Release"
    -- Installing: /Users/mnovak/opt/leuven1/includes/Matrix.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/cxxopts.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/definitions.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/types.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/FBLAS.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/XBLAS.tpp
    -- Installing: /Users/mnovak/opt/leuven1/includes/FBLAS.h
    -- Installing: /Users/mnovak/opt/leuven1/includes/FBLAS.tpp
    -- Up-to-date: /Users/mnovak/opt/leuven1/includes/definitions.hh
    -- Installing: /Users/mnovak/opt/leuven1/lib/libleuven.a
    -- Installing: /Users/mnovak/opt/leuven1/lib/cmake/leuven/leuvenConfig.cmake
    -- Installing: /Users/mnovak/opt/leuven1/includes/IncCholesky.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/Kernels.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/IncCholesky.tpp
    -- Installing: /Users/mnovak/opt/leuven1/includes/KernelChi2.tpp
    -- Installing: /Users/mnovak/opt/leuven1/includes/KernelRBF.tpp
    -- Installing: /Users/mnovak/opt/leuven1/includes/KernelSSK.tpp
    -- Installing: /Users/mnovak/opt/leuven1/includes/KscEncodingAndQM.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/KscEncodingAndQM_AMS.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/KscEncodingAndQM_BAS.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/KscEncodingAndQM_BLF.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/KscWkpcaIChol.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/KscWkpcaIChol.tpp

to install the :math:`\texttt{leuven}` library, headers and configurations to 
the location specified by the :math:`\texttt{-DCMAKE}\_\texttt{INSTALL}\_\texttt{PREFIX}` 
:math:`\texttt{cmake}` configuration option.  
 


On the BLAS/LAPACK dependence 
------------------------------

The :math:`\texttt{leuven}` library :cite:`libleuven` **requires one of the BLAS** 
(Basic Linear Algebra Subprograms) :cite:`BLAS_ref` and **LAPACK** 
(Linear Algebra PACKage) :cite:`LAPACK_ref` implementations available on the system. 
Several implementation versions of these specifications are supported explicitly
while many others can also be used. The :math:`\texttt{leuven}` library 
:cite:`libleuven` hides the dependences behind wrappers that can be configured 
with the appropriate :math:`\texttt{cmake}` options at build time. These 
configuration options as well as the supported BLAS/LAPACK implementation options 
are discussed in the followings.


Row- and column-major memory layout 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to facilitate the work with array like (i.e. vector, matrix) data, the
:math:`\texttt{leuven}` library :cite:`libleuven` contains a very lightweight 
matrix implementation **supporting different data types** stored in 
**both row- and column-major memory layouts**. These options are implemented 
using template arguments combined with Curiously Recurring Template Pattern (CRTP)
for minimising the corresponding run time overheads. 

The BLAS/LAPACK implementation routines manipulate these array like data 
encapsulated in the lightweight matrix objects. The :math:`\texttt{leuven}` library 
contains wrappers that (*i*) hide the complexity of calling these BLAS/LAPACK 
routines, (**ii**) the corresponding methods operate on the lightweight matrix 
objects that greatly simplifies the corresponding calls, (**iii**) hides all the 
differences between BLAS/LAPACK implementations that do or do not provide C 
style interfaces. This later is because while the provided lightweight matrix 
implementation supports both row- and column-major memory layouts, only a sub-set 
of the BLAS/LAPACK implementations contains the :math:`\texttt{cblas/lapacke}` 
C interfaces providing the possibility of using the C style, row-major memory 
layout beyond the Fortran style, column-major one. When the selected  
BLAS/LAPACK implementation supports both C and Fortran style interfaces (i.e. 
both row- and column-major memory layouts) the user can select which of them 
to be used. This can be done at the configuration time of the 
:math:`\texttt{leuven}` library through the 
:math:`\texttt{-DUSE}\_\texttt{CBLAS}\_\texttt{WRAPPER \{ON, OFF(default)\} }`
:math:`\texttt{cmake}` configuration option

 - :math:`\texttt{-DUSE}\_\texttt{CBLAS}\_\texttt{WRAPPER = ON}` : both row- and 
   column-major memory layouts are supported when calling the BLAS/LAPACK routines 
   and the appropriate layout is communicated to the BLAS/LAPACK side 
   automatically by the wrappers that receives the matrix object(s). 

 - :math:`\texttt{-DUSE}\_\texttt{CBLAS}\_\texttt{WRAPPER = OFF (default)}` : 
   only column-major memory layouts is supported when calling the BLAS/LAPACK 
   routines.

Note, that the top level **BLAS/LAPACK interface methods** provided by the 
:math:`\texttt{leuven}` library, **are identical in both cases**!   
See :numref:`table_BLAS` for the list of *explicitly* supported BLAS/LAPACK versions 
and their ability of providing C interfaces.   

There are several mechanisms in the :math:`\texttt{leuven}` library that prevents
the mismatch between the memory layout required by the selected BLAS/LAPACK  
implementation and that used in the given matrix object on which the 
BLAS/LAPACK routines operates on 

 - a check is implemented already at :math:`\texttt{cmake}` configuration of the
   :math:`\texttt{leuven}` library, that will report an appropriate configuration 
   error when the C style BLAS/LAPACK interface is required by the user (i.e. 
   :math:`\texttt{-DUSE}\_\texttt{CBLAS}\_\texttt{WRAPPER = ON}`) with a selected 
   BLAS/LAPACK implementation that do not provide the necessary :math:`\texttt{cblas/lapacke}`
   
 - the wrapper methods, that provides the bridge between :math:`\texttt{leuven}` 
   library and the BLAS/LAPACK implementations, are templated and template 
   specialisation exist only for the appropriate combination of (lower level) 
   wrapper methods and matrix memory layouts. Therefore, interface methods with 
   possible mismatch between the memory layout required by the BLAS/LAPACK routine 
   and that used by the matrix input argument, simply do not exist (resulting 
   compile time errors in the user code in case of wrong combinations).


.. table:: Supported (explicitly) BLAS/LAPACK libraries with the information if 
   it supports row-major memory layout i.e. if it can be used with the 
   :math:`\texttt{-DUSE}\_\texttt{CBLAS}\_\texttt{WRAPPER = ON}` option and if 
   it is multithreaded or sequential. 
   :widths: 15 20 20
   :name: table_BLAS
        
   ==============================  ===================  ===================
     Name                           Row-major support     Multithreaded 
   ==============================  ===================  ===================
    :ref:`Intel MKL <mkl_sc>`             YES                 YES
    :ref:`OpenBLAS <openBLAS_sc>`         YES                 YES
    :ref:`ATLAS <ATLAS_sc>`                NO                 YES (fixed [1]_)
    :ref:`Netlib <Netlib_sc>`              NO [2]_             NO
    :ref:`Any <anyBLAS_sc>`                NO                   ? [3]_
   ==============================  ===================  ===================

.. [1]  The number of threads are determined and fixed at ATALS compile time
        so it cannot be changed dynamically in the dependent applications (contraty 
        to the MKL and OpenBLAS cases).
  
.. [2]  Although more recent versions of the Netlib BLAS/LAPACK implementations 
        supports the :math:`\texttt{cblas/lapacke}` interfaces, the 
        :math:`\texttt{leuven}` library has not been modified yet for following this.

.. [3]  The BLAS/LAPACK implementation found in this case is unknown.
    

.. _blas_lapack_options_howto:

BLAS/LAPACK options 
~~~~~~~~~~~~~~~~~~~~~~

As it has already been mentioned, the :math:`\texttt{leuven}` library requires 
that BLAS and LAPACK libraries are available on the system. Several 
implementations of these libraries are supported explicitly that can be specified 
by the user at configuration time of the :math:`\texttt{leuven}` library through
:math:`\texttt{cmake}` configuration options. These will be listed below with 
examples together with a *wild card* option, supporting any implementations.


.. _mkl_sc:

Intel Math Kernel Library (MKL)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The recommended BLAS/LAPACK implementation is the Intel Math Kernel Library (MKL)
:cite:`mkl` that provides the best performance on the appropriate (i.e. Intel) 
platforms. The MKL library is available at :cite:`mkl` with *Getting Started* 
instruction at :cite:`mkl_developer_guides` for post install setups. 

The Intel MKL BLAS/LAPACK implementation can be selected explicitly at the 
configuration of the :math:`\texttt{leuven}` library by setting the following 
:math:`\texttt{cmake}` configuration options as

 - :math:`\texttt{-DCPU}\_\texttt{BLAS}\_\texttt{OPTION = USE}\_\texttt{MKL}\_\texttt{BLAS}`
   
 - :math:`\texttt{-DMKL}\_\texttt{ROOT}\_\texttt{DIR = where/mkl/is/installed}`
   i.e. where the Intel MKL :math:`\texttt{include}` and :math:`\texttt{lib}` 
   directories (among others) are installed on the system. 

Note, that the :math:`\texttt{leuven cmake}` configuration will look for the 
Intel MKL :math:`\texttt{include}` and :math:`\texttt{lib}` directories in 
the following order: 

 - first the location specified by the :math:`\texttt{-DMKL}\_\texttt{ROOT}\_\texttt{DIR cmake}` 
   configuration option (if any)

 - then the location stored in the :math:`\texttt{MKLROOT}` environment variable 
   (if exist; can be set by an Intel MKL script, see the corresponding 
   *Setting Environment Variables* section of the Intel MKL *Getting Started* 
   instructions at :cite:`mkl_developer_guides`)
   
 - if none of the above were successful, then the :math:`\texttt{/opt/intel/mkl}`
   location is checked  

**Example**

On my (MacOS) system, the Intel MKL library is installed under the 
:math:`\texttt{/Users/mnovak/opt/IntelMKL/}` directory and the math:`\texttt{include}` and 
:math:`\texttt{lib}` directories can be found under the 
:math:`\texttt{compilers}\_\texttt{and}\_\texttt{libraries}\_\texttt{2019.4.233/mac/mkl/}`
sub-directory. 
The :math:`\texttt{leuven}` library can be configured to use the corresponding 
Intel MLK BLAS/LAPACK implementations as ::

    bash-3.2$ cmake -DCMAKE_INSTALL_PREFIX=/Users/mnovak/opt/leuven1 -DUSE_CBLAS_WRAPPER=ON -DCPU_BLAS_OPTION=USE_MKL_BLAS -DMKL_ROOT_DIR=/Users/mnovak/opt/IntelMKL/compilers_and_libraries_2019.4.233/mac/mkl/ ../
    -- The C compiler identification is AppleClang 10.0.0.10001145
    -- The CXX compiler identification is AppleClang 10.0.0.10001145
    -- Check for working C compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc
    -- Check for working C compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc -- works
    -- Detecting C compiler ABI info
    -- Detecting C compiler ABI info - done
    -- Detecting C compile features
    -- Detecting C compile features - done
    -- Check for working CXX compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++
    -- Check for working CXX compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ -- works
    -- Detecting CXX compiler ABI info
    -- Detecting CXX compiler ABI info - done
    -- Detecting CXX compile features
    -- Detecting CXX compile features - done
    -- ===== WRAPPER ===== 
    -- Building with the CBLAS Wrapper

    -- ==== The selected CPU BLAS Option = MKL-BLAS  ==== 
    --  
    -- ========  MKL ======= 
    -- Found MKL: /Users/mnovak/opt/IntelMKL/compilers_and_libraries_2019.4.233/mac/mkl//lib  
    -- MKL INCLUDE DIRECTORY = /Users/mnovak/opt/IntelMKL/compilers_and_libraries_2019.4.233/mac/mkl//include
    -- MKL LIBRARY DIRECTORY = /Users/mnovak/opt/IntelMKL/compilers_and_libraries_2019.4.233/mac/mkl//lib
    -- MKL LIBRARIES = /Users/mnovak/opt/IntelMKL/compilers_and_libraries_2019.4.233/mac/mkl/lib/libmkl_rt.dylib;/usr/lib/libpthread.dylib
    --  
    -- ===== CHECKING CPU BLAS WRAPPER AND LIBRARY CONSISTENCY =====
    --  
    -- ===== Setting up the leuven library =====
     
    -- ===== Adding the lssvm part ===== 

    -- Configuring done
    -- Generating done

Note, that additional :math:`cmake` configuration options were also used explicitly

 - :math:`\texttt{-DCMAKE}\_\texttt{INSTALL}\_\texttt{PREFIX}` : to specify the 
   location where the :math:`\texttt{leuven}` library is required to be installed
 
 - :math:`\texttt{-DUSE}\_\texttt{CBLAS}\_\texttt{WRAPPER=ON}` : specify to use 
   the more flexible, C style BLAS/LAPACK interfaces (i.e. :math:`\texttt{cblas/lapacke}`) 
   provided by the Intel MKL implementations (see above)
   
as well as implicitly (i.e. by taking their default values) such as

 - :math:`\texttt{-DCMAKE}\_\texttt{BUILD}\_\texttt{TYPE=Release}` : for having 
   an optimised build configuration in contrast to e.g. :math:`\texttt{Debug}` 
   build option
  
or the C/C++ compiler options discussed at just to mention some of the most 
important.

After successful configuration one can build the library as ::

  bash-3.2$ make 
  Scanning dependencies of target leuven
  [ 50%] Building CXX object utils/CMakeFiles/leuven.dir/src/CBLAS.cc.o
  [100%] Linking CXX static library ../lib/libleuven.a
  [100%] Built target leuven

and install to the location (specified at configuration by the 
:math:`\texttt{-DCMAKE}\_\texttt{INSTALL}\_\texttt{PREFIX})` as ::

    bash-3.2$ make install
    [100%] Built target leuven
    Install the project...
    -- Install configuration: "Release"
    -- Installing: /Users/mnovak/opt/leuven1/includes/Matrix.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/cxxopts.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/definitions.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/types.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/CBLAS.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/XBLAS.tpp
    -- Installing: /Users/mnovak/opt/leuven1/includes/CBLAS.tpp
    -- Up-to-date: /Users/mnovak/opt/leuven1/includes/definitions.hh
    -- Installing: /Users/mnovak/opt/leuven1/lib/libleuven.a
    -- Installing: /Users/mnovak/opt/leuven1/lib/cmake/leuven/leuvenConfig.cmake
    -- Installing: /Users/mnovak/opt/leuven1/includes/IncCholesky.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/Kernels.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/IncCholesky.tpp
    -- Installing: /Users/mnovak/opt/leuven1/includes/KernelChi2.tpp
    -- Installing: /Users/mnovak/opt/leuven1/includes/KernelRBF.tpp
    -- Installing: /Users/mnovak/opt/leuven1/includes/KernelSSK.tpp
    -- Installing: /Users/mnovak/opt/leuven1/includes/KscEncodingAndQM.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/KscEncodingAndQM_AMS.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/KscEncodingAndQM_BAS.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/KscEncodingAndQM_BLF.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/KscWkpcaIChol.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/KscWkpcaIChol.tpp

Then the :math:`\texttt{leuven}` library/toolkit is ready to be used. See 
example applications at ...


.. _openBLAS_sc:

OpenBLAS 
^^^^^^^^^

A good alternative to MKL is the OpenBLAS :cite:`openBLAS` optimised BLAS 
implementation that also includes a LAPACK version. 

The OpenBLAS BLAS/LAPACK implementation can be selected explicitly at the 
configuration of the :math:`\texttt{leuven}` library by setting the following 
:math:`\texttt{cmake}` configuration options as

 - :math:`\texttt{-DCPU}\_\texttt{BLAS}\_\texttt{OPTION = USE}\_\texttt{OPEN}\_\texttt{BLAS}`
   
 - :math:`\texttt{-DOpenBLAS}\_\texttt{DIR = where/openBLAS/is/installed}`
   i.e. where the OpenBLAS :math:`\texttt{include}` and :math:`\texttt{lib}` 
   directories (among others) are installed on the system. 

Note, that the :math:`\texttt{leuven cmake}` configuration will look for the 
OpenBLAS :math:`\texttt{include}` and :math:`\texttt{lib}` directories in 
the following order: 

 - first the location specified by the :math:`\texttt{-DOpenBLAS}\_\texttt{DIR cmake}` 
   configuration option (if any)

 - then the location stored in the :math:`\texttt{OpenBLASROOT}` environment variable 
   (if exist)
   
 - several other standard locations are searched (e.g. :math:`\texttt{/opt/OpenBLAS, /usr, /usr/opt, /usr/local, etc.}`) 
   for finding the OpenBLAS library and the corresponding headers (but the first two are recommended)

**Example**

On my (MacOS) system, the OpenBLAS library and headers are installed under the 
:math:`\texttt{/Users/mnovak/opt/OpenBLAS/}` directory and the :math:`\texttt{include}`
and :math:`\texttt{lib}` directories can be found directly under this main directory.

The :math:`\texttt{leuven}` library can be configured to use the corresponding 
OpenBLAS BLAS/LAPACK implementations as ::

    bash-3.2$ cmake -DCMAKE_INSTALL_PREFIX=/Users/mnovak/opt/leuven1 -DUSE_CBLAS_WRAPPER=ON -DCPU_BLAS_OPTION=USE_OPEN_BLAS -DOpenBLAS_DIR=/Users/mnovak/opt/OpenBLAS/ ../
    -- The C compiler identification is AppleClang 10.0.0.10001145
    -- The CXX compiler identification is AppleClang 10.0.0.10001145
    -- Check for working C compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc
    -- Check for working C compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc -- works
    -- Detecting C compiler ABI info
    -- Detecting C compiler ABI info - done
    -- Detecting C compile features
    -- Detecting C compile features - done
    -- Check for working CXX compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++
    -- Check for working CXX compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ -- works
    -- Detecting CXX compiler ABI info
    -- Detecting CXX compiler ABI info - done
    -- Detecting CXX compile features
    -- Detecting CXX compile features - done
    -- ===== WRAPPER ===== 
    -- Building with the CBLAS Wrapper

    -- ==== The selected CPU BLAS Option = OpenBLAS  ==== 
    --  
    -- ========  OpenBLAS ======= 
    -- OpenBLAS INCLUDE DIRECTORY = /Users/mnovak/opt/OpenBLAS/include/openblas
    -- OpenBLAS LIBRARY = /Users/mnovak/opt/OpenBLAS/lib/libopenblas.a
    --  
    -- ===== CHECKING CPU BLAS WRAPPER AND LIBRARY CONSISTENCY =====
    --  
    -- ===== Setting up the leuven library =====
     
    -- ===== Adding the lssvm part ===== 

    -- Configuring done
    -- Generating done

Similarly to the previous MKL case, additional :math:`cmake` configuration 
options were also used to specify the install location of the :math:`\texttt{leuven}` 
library or to require the C style BLAS/LAPACK interface to be used.

After successful configuration one can build the library as ::

  bash-3.2$ make 
  Scanning dependencies of target leuven
  [ 50%] Building CXX object utils/CMakeFiles/leuven.dir/src/CBLAS.cc.o
  [100%] Linking CXX static library ../lib/libleuven.a
  [100%] Built target leuven

and install to the location (specified at configuration by the 
:math:`\texttt{-DCMAKE}\_\texttt{INSTALL}\_\texttt{PREFIX})` as ::

    bash-3.2$ make install
    [100%] Built target leuven
    Install the project...
    -- Install configuration: "Release"
    -- Installing: /Users/mnovak/opt/leuven1/includes/Matrix.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/cxxopts.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/definitions.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/types.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/CBLAS.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/XBLAS.tpp
    -- Installing: /Users/mnovak/opt/leuven1/includes/CBLAS.tpp
    -- Up-to-date: /Users/mnovak/opt/leuven1/includes/definitions.hh
    -- Installing: /Users/mnovak/opt/leuven1/lib/libleuven.a
    -- Installing: /Users/mnovak/opt/leuven1/lib/cmake/leuven/leuvenConfig.cmake
    -- Installing: /Users/mnovak/opt/leuven1/includes/IncCholesky.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/Kernels.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/IncCholesky.tpp
    -- Installing: /Users/mnovak/opt/leuven1/includes/KernelChi2.tpp
    -- Installing: /Users/mnovak/opt/leuven1/includes/KernelRBF.tpp
    -- Installing: /Users/mnovak/opt/leuven1/includes/KernelSSK.tpp
    -- Installing: /Users/mnovak/opt/leuven1/includes/KscEncodingAndQM.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/KscEncodingAndQM_AMS.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/KscEncodingAndQM_BAS.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/KscEncodingAndQM_BLF.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/KscWkpcaIChol.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/KscWkpcaIChol.tpp

Then the :math:`\texttt{leuven}` library/toolkit is ready to be used. See 
example applications at ...

.. _ATLAS_sc:

ATLAS (Automatically Tuned Linear Algebra Software)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An other alternative, for having an optimised version of BLAS implementation, is 
to use ATLAS (Automatically Tuned Linear Algebra Software) :cite:`ATLAS,whaley04`. 
ATLAS includes a LAPACK implementation but the C interface, provided to the 
LAPACK part, is different than :math:`\texttt{lapacke}` (although the BLAS part 
supports the :math:`\texttt{cblas}` C interface).
Therefore, only the :math:`\texttt{-DUSE}\_\texttt{CBLAS}\_\texttt{WRAPPER=OFF}` 
(or leave it to the default :math:`\texttt{OFF}` value) is supported by the 
:math:`\texttt{leuven}` library. This means, that only column-major matrix 
memory layouts can be used in the BLAS/LAPACK interfaces when ATLAS is selected 
as a BLAS/LAPACK implementation option.

The ATLAS BLAS/LAPACK implementation can be selected explicitly at the 
configuration of the :math:`\texttt{leuven}` library by setting the following 
:math:`\texttt{cmake}` configuration options as

 - :math:`\texttt{-DCPU}\_\texttt{BLAS}\_\texttt{OPTION = USE}\_\texttt{ATLAS}\_\texttt{BLAS}`
   
 - :math:`\texttt{-DAtlasBLAS}\_\texttt{DIR = where/ATLAS/is/installed}`
   i.e. where the ATLAS BLAS :math:`\texttt{include}` and :math:`\texttt{lib}` 
   directories are installed on the system. 

Note, that the :math:`\texttt{leuven cmake}` configuration will look for the 
ATLAS BLAS :math:`\texttt{lib}` directories in the following order: 

 - first the location specified by the :math:`\texttt{-DAtlasBLAS}\_\texttt{DIR cmake}` 
   configuration option (if any)

 - then the location stored in the :math:`\texttt{AtlasBLASROOT}` environment variable 
   (if exist)
   
 - several other standard locations are searched (e.g. :math:`\texttt{/opt/AtlasBLAS, /usr, /usr/opt, /usr/local, etc.}`) 
   for finding the ATLAS BLAS library (but the first two are recommended)

**Example**

On my (MacOS) system, the ATLAS BLAS libraries and headers are installed under the 
:math:`\texttt{/Users/mnovak/opt/ATLAS/}` directory and the :math:`\texttt{include}`
and :math:`\texttt{lib}` directories can be found directly under this main directory.

The :math:`\texttt{leuven}` library can be configured to use the corresponding 
BLAS/LAPACK implementations provided by ATLAS as ::

    bash-3.2$ cmake -DCMAKE_INSTALL_PREFIX=/Users/mnovak/opt/leuven1 -DCPU_BLAS_OPTION=USE_ATLAS_BLAS -DAtlasBLAS_DIR=/Users/mnovak/opt/ATLAS/ ../
    -- The C compiler identification is AppleClang 10.0.0.10001145
    -- The CXX compiler identification is AppleClang 10.0.0.10001145
    -- Check for working C compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc
    -- Check for working C compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc -- works
    -- Detecting C compiler ABI info
    -- Detecting C compiler ABI info - done
    -- Detecting C compile features
    -- Detecting C compile features - done
    -- Check for working CXX compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++
    -- Check for working CXX compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ -- works
    -- Detecting CXX compiler ABI info
    -- Detecting CXX compiler ABI info - done
    -- Detecting CXX compile features
    -- Detecting CXX compile features - done
    -- ===== WRAPPER ===== 
    -- Building with the FBLAS Wrapper

    -- ==== The selected CPU BLAS Option = ATLAS-BLAS  ==== 
    --  
    -- ========  ATLAS BLAS ======= 
    --  WAS FOUND = TRUE
    -- ATLAS BLAS LIBRARY = /Users/mnovak/opt/ATLAS/lib/libatlas.a;/Users/mnovak/opt/ATLAS/lib/liblapack.a;/Users/mnovak/opt/ATLAS/lib/libcblas.a;/Users/mnovak/opt/ATLAS/lib/libptcblas.a;/Users/mnovak/opt/ATLAS/lib/libf77blas.a;/Users/mnovak/opt/ATLAS/lib/libptf77blas.a
    --  
    -- ===== CHECKING CPU BLAS WRAPPER AND LIBRARY CONSISTENCY =====
    --  
    -- ===== Setting up the leuven library =====
     
    -- ===== Adding the lssvm part ===== 

    -- Configuring done
    -- Generating done


Similarly to the previous cases, additional :math:`cmake` configuration 
option was also used to specify the install location of the :math:`\texttt{leuven}` 
library.

After successful configuration one can build the library as ::

    bash-3.2$ make 
    Scanning dependencies of target leuven
    [ 50%] Building CXX object utils/CMakeFiles/leuven.dir/src/FBLAS.cc.o
    [100%] Linking CXX static library ../lib/libleuven.a
    [100%] Built target leuven

and install to the location (specified at configuration by the 
:math:`\texttt{-DCMAKE}\_\texttt{INSTALL}\_\texttt{PREFIX})` as ::

    bash-3.2$ make install
    [100%] Built target leuven
    Install the project...
    -- Install configuration: "Release"
    -- Installing: /Users/mnovak/opt/leuven1/includes/Matrix.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/cxxopts.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/definitions.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/types.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/FBLAS.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/XBLAS.tpp
    -- Installing: /Users/mnovak/opt/leuven1/includes/FBLAS.h
    -- Installing: /Users/mnovak/opt/leuven1/includes/FBLAS.tpp
    -- Up-to-date: /Users/mnovak/opt/leuven1/includes/definitions.hh
    -- Installing: /Users/mnovak/opt/leuven1/lib/libleuven.a
    -- Installing: /Users/mnovak/opt/leuven1/lib/cmake/leuven/leuvenConfig.cmake
    -- Installing: /Users/mnovak/opt/leuven1/includes/IncCholesky.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/Kernels.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/IncCholesky.tpp
    -- Installing: /Users/mnovak/opt/leuven1/includes/KernelChi2.tpp
    -- Installing: /Users/mnovak/opt/leuven1/includes/KernelRBF.tpp
    -- Installing: /Users/mnovak/opt/leuven1/includes/KernelSSK.tpp
    -- Installing: /Users/mnovak/opt/leuven1/includes/KscEncodingAndQM.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/KscEncodingAndQM_AMS.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/KscEncodingAndQM_BAS.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/KscEncodingAndQM_BLF.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/KscWkpcaIChol.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/KscWkpcaIChol.tpp

Then the :math:`\texttt{leuven}` library/toolkit is ready to be used. See 
example applications at ...

.. _Netlib_sc:

Netlib refrence BLAS/LAPACK
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Netlib reference BLAS :cite:`netlibBLAS` and LAPACK :cite:`netlibLAPACK` 
implementations are always available as a final solution. It should be noted, 
that these are less performant compared to the MKL, OpenBLAS or ATLAS versions. 
Moreover, they do not provide the :math:`\texttt{cblas/lapacke}` interfaces 
(actually the latest version do!), that would be necessary for us to in order to 
be able to use both row- and column-major memory layouts when calling the 
corresponding routines. Therefore,
only the :math:`\texttt{-DUSE}\_\texttt{CBLAS}\_\texttt{WRAPPER=OFF}` 
(or leave it to the default :math:`\texttt{OFF}` value) is supported by the 
:math:`\texttt{leuven}` library. This means, that only column-major matrix 
memory layouts can be used in the BLAS/LAPACK interfaces when the Netlib
reference implementation is selected as a BLAS/LAPACK option.

The Netlib BLAS/LAPACK implementation can be selected explicitly at the 
configuration of the :math:`\texttt{leuven}` library by setting the following 
:math:`\texttt{cmake}` configuration options as

 - :math:`\texttt{-DCPU}\_\texttt{BLAS}\_\texttt{OPTION = USE}\_\texttt{NETLIB}\_\texttt{BLAS}`
   
 - :math:`\texttt{-DNETLIB}\_\texttt{BLAS}\_\texttt{DIR = where/Netlib/is/installed}`
   i.e. where the Netlib :math:`\texttt{lib}` directories is installed on the system. 

Note, that the :math:`\texttt{leuven cmake}` configuration will look for the 
Netlib BALS/LAPACK libraries in the :math:`\texttt{lib}` directories in the 
following order: 

 - first the location specified by the :math:`\texttt{-DNETLIB}\_\texttt{BLAS}\_\texttt{DIR cmake}` 
   configuration option (if any)

 - then the location stored in the :math:`\texttt{NETLIB}\_\texttt{BLASROOT}` environment variable 
   (if exist)
   
 - several other standard locations are searched (e.g. :math:`\texttt{/opt/BLAS, /usr, /usr/opt, /usr/local, etc.}`) 
   for finding the ATLAS BLAS/LAPACK libraries (but the first two are recommended)

**Example**

On my (MacOS) system, the Netlib reference BLAS/LAPACK implementations 
are installed under the :math:`\texttt{/Users/mnovak/opt/Netlib/}` directory and
:math:`\texttt{lib}` directory (containing :math:`\texttt{libblas.a,liblapack.a}`) 
can be found directly under this main directory.

The :math:`\texttt{leuven}` library can be configured to use the corresponding 
BLAS/LAPACK implementations provided by Netlib as ::

    bash-3.2$ cmake -DCMAKE_INSTALL_PREFIX=/Users/mnovak/opt/leuven1 -DCPU_BLAS_OPTION=USE_NETLIB_BLAS -DNETLIB_BLAS_DIR=/Users/mnovak/opt/NETLIB_LAPACK_BLAS/ ../
    -- The C compiler identification is AppleClang 10.0.0.10001145
    -- The CXX compiler identification is AppleClang 10.0.0.10001145
    -- Check for working C compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc
    -- Check for working C compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc -- works
    -- Detecting C compiler ABI info
    -- Detecting C compiler ABI info - done
    -- Detecting C compile features
    -- Detecting C compile features - done
    -- Check for working CXX compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++
    -- Check for working CXX compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ -- works
    -- Detecting CXX compiler ABI info
    -- Detecting CXX compiler ABI info - done
    -- Detecting CXX compile features
    -- Detecting CXX compile features - done
    -- ===== WRAPPER ===== 
    -- Building with the FBLAS Wrapper

    -- ==== The selected CPU BLAS Option = NETLIB-BLAS  ==== 
    --  
    -- ========  NETLIB BLAS (or any BLAS) ======= 
    --  WAS FOUND = TRUE
    -- NETLIB BLAS LIBRARY = /Users/mnovak/opt/NETLIB_LAPACK_BLAS/lib/libblas.a;/Users/mnovak/opt/NETLIB_LAPACK_BLAS/lib/liblapack.a
    --  
    -- ===== CHECKING CPU BLAS WRAPPER AND LIBRARY CONSISTENCY =====
    --  
    -- ===== Setting up the leuven library =====
     
    -- ===== Adding the lssvm part ===== 

    -- Configuring done
    -- Generating done


Similarly to the previous cases, additional :math:`cmake` configuration 
option was also used to specify the install location of the :math:`\texttt{leuven}` 
library.

After successful configuration one can build the library as ::

    bash-3.2$ make 
    Scanning dependencies of target leuven
    [ 50%] Building CXX object utils/CMakeFiles/leuven.dir/src/FBLAS.cc.o
    [100%] Linking CXX static library ../lib/libleuven.a
    [100%] Built target leuven

and install to the location (specified at configuration by the 
:math:`\texttt{-DCMAKE}\_\texttt{INSTALL}\_\texttt{PREFIX})` as ::

    bash-3.2$ make install
    [100%] Built target leuven
    Install the project...
    -- Install configuration: "Release"
    -- Installing: /Users/mnovak/opt/leuven1/includes/Matrix.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/cxxopts.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/definitions.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/types.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/FBLAS.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/XBLAS.tpp
    -- Installing: /Users/mnovak/opt/leuven1/includes/FBLAS.h
    -- Installing: /Users/mnovak/opt/leuven1/includes/FBLAS.tpp
    -- Up-to-date: /Users/mnovak/opt/leuven1/includes/definitions.hh
    -- Installing: /Users/mnovak/opt/leuven1/lib/libleuven.a
    -- Installing: /Users/mnovak/opt/leuven1/lib/cmake/leuven/leuvenConfig.cmake
    -- Installing: /Users/mnovak/opt/leuven1/includes/IncCholesky.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/Kernels.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/IncCholesky.tpp
    -- Installing: /Users/mnovak/opt/leuven1/includes/KernelChi2.tpp
    -- Installing: /Users/mnovak/opt/leuven1/includes/KernelRBF.tpp
    -- Installing: /Users/mnovak/opt/leuven1/includes/KernelSSK.tpp
    -- Installing: /Users/mnovak/opt/leuven1/includes/KscEncodingAndQM.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/KscEncodingAndQM_AMS.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/KscEncodingAndQM_BAS.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/KscEncodingAndQM_BLF.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/KscWkpcaIChol.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/KscWkpcaIChol.tpp

Then the :math:`\texttt{leuven}` library/toolkit is ready to be used. See 
example applications at ...


.. _anyBLAS_sc:

The *wild card* BLAS option
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As it has already been mentioned, while only a sub-set of BLAS/LAPACK implementation 
provides explicit C interfaces, all of them can be accessed from any C/C++ codes
by the appropriate :math:`external` function definitions and linking. The only 
drawback of this case is that the matrices, used in the BLAS/LAPACK routines, 
must follow the Fortran style, column-major memory layout. This is actually 
exactly the case when using the :math:`\texttt{FBLAS}` interface i.e. when 
:math:`\texttt{-DUSE}\_\texttt{CBLAS}\_\texttt{WRAPPER=OFF}` (or left with its 
default :math:`\texttt{OFF}` value).
Moreover, when the required BLAS/LAPACK option location is not specified explicitly, 
several standard locations will be automatically searched by the 
:math:`\texttt{cmake}` configuration for trying to find any BLAS/LAPACK libraries.

Therefore, by not specifying the required BLAS/LAPACK option at the 
:math:`\texttt{cmake}` configuration time (and not requiring the 
:math:`\texttt{CBLAS}` wrapper to be used i.e. setting 
math:`\texttt{-DUSE}\_\texttt{CBLAS}\_\texttt{WRAPPER=OFF}` or leaving on its default 
:math:`\texttt{OFF}`) can be used to try to find any suitable BLAS/LAPACK libraries 
on the system. (Note, that the default BLAS/LAPACK option is set to Netlib and 
this name will be shown independently from what implementation can eventually be 
found on the system.)


**Example**

On my (MacOS) system, the (whatever) BLAS/LAPACK libraries are also installed at 
:math:`\texttt{/usr/lib/}` i.e. one of the standard library locations.

The :math:`\texttt{leuven}` library can be configured to find any suitable 
BLAS/LAPACK libraries on the system in the following way ::

    bash-3.2$ cmake -DCMAKE_INSTALL_PREFIX=/Users/mnovak/opt/leuven1 ../
    -- The C compiler identification is AppleClang 10.0.0.10001145
    -- The CXX compiler identification is AppleClang 10.0.0.10001145
    -- Check for working C compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc
    -- Check for working C compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc -- works
    -- Detecting C compiler ABI info
    -- Detecting C compiler ABI info - done
    -- Detecting C compile features
    -- Detecting C compile features - done
    -- Check for working CXX compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++
    -- Check for working CXX compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ -- works
    -- Detecting CXX compiler ABI info
    -- Detecting CXX compiler ABI info - done
    -- Detecting CXX compile features
    -- Detecting CXX compile features - done
    -- ===== WRAPPER ===== 
    -- Building with the FBLAS Wrapper

    -- ==== The selected CPU BLAS Option = NETLIB-BLAS  ==== 
    --  
    -- ========  NETLIB BLAS (or any BLAS) ======= 
    --  WAS FOUND = TRUE
    -- NETLIB BLAS LIBRARY = /usr/lib/libblas.dylib;/usr/lib/liblapack.dylib
    --  
    -- ===== CHECKING CPU BLAS WRAPPER AND LIBRARY CONSISTENCY =====
    --  
    -- ===== Setting up the leuven library =====
     
    -- ===== Adding the lssvm part ===== 

    -- Configuring done
    -- Generating done

Note, that with this minimal :math:`\texttt{cmake}` configuration (i.e. only 
:math:`\texttt{leuven}` library installation destination is specified) everything 
works fine as well as (long as BLAS/LAPACK libraries can be located at one of the  
standard locations of libraries). However, there is very little control on the 
selected BLAS/LAPACK implementations and the corresponding performance.

After successful configuration I can build the library as ::

    bash-3.2$ make 
    Scanning dependencies of target leuven
    [ 50%] Building CXX object utils/CMakeFiles/leuven.dir/src/FBLAS.cc.o
    [100%] Linking CXX static library ../lib/libleuven.a
    [100%] Built target leuven

and install to the location (specified at configuration by the 
:math:`\texttt{-DCMAKE}\_\texttt{INSTALL}\_\texttt{PREFIX})` as ::

    bash-3.2$ make install
    [100%] Built target leuven
    Install the project...
    -- Install configuration: "Release"
    -- Installing: /Users/mnovak/opt/leuven1/includes/Matrix.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/cxxopts.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/definitions.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/types.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/FBLAS.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/XBLAS.tpp
    -- Installing: /Users/mnovak/opt/leuven1/includes/FBLAS.h
    -- Installing: /Users/mnovak/opt/leuven1/includes/FBLAS.tpp
    -- Up-to-date: /Users/mnovak/opt/leuven1/includes/definitions.hh
    -- Installing: /Users/mnovak/opt/leuven1/lib/libleuven.a
    -- Installing: /Users/mnovak/opt/leuven1/lib/cmake/leuven/leuvenConfig.cmake
    -- Installing: /Users/mnovak/opt/leuven1/includes/IncCholesky.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/Kernels.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/IncCholesky.tpp
    -- Installing: /Users/mnovak/opt/leuven1/includes/KernelChi2.tpp
    -- Installing: /Users/mnovak/opt/leuven1/includes/KernelRBF.tpp
    -- Installing: /Users/mnovak/opt/leuven1/includes/KernelSSK.tpp
    -- Installing: /Users/mnovak/opt/leuven1/includes/KscEncodingAndQM.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/KscEncodingAndQM_AMS.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/KscEncodingAndQM_BAS.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/KscEncodingAndQM_BLF.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/KscWkpcaIChol.hh
    -- Installing: /Users/mnovak/opt/leuven1/includes/KscWkpcaIChol.tpp

Then the :math:`\texttt{leuven}` library/toolkit is ready to be used. See 
example applications at ...



CUDA support
~~~~~~~~~~~~~~~~~~

See :ref:`leuven_CUDA_support` for the description of ``CUDA`` support in the 
:math:`\texttt{leuven}` library.

