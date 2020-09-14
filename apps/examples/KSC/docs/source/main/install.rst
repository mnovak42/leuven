.. _ksc_build_and_install_sec_doc:

Build and install 
==================

This section contains detailed instructions on how to build and install the 
provided KSC applications.


Requirements 
-------------

Building the KSC applications requires:

  - a :math:`\texttt{c/c++}` compiler, with :math:`\texttt{c++11}` support, to be installed 
    on the system 
  - :math:`\texttt{CMake}` is used for managing (option configuration, locate 
    dependencies etc.) the build process, so it needs to be installed on the system
    (see at http://www.cmake.org/)
  - the KSC applications are developed based on the functionalities provided by the 
    :math:`\texttt{leuven}` library and framework :cite:`libleuven` as example 
    applications. Therefore, the :math:`\texttt{leuven}` library needs to be 
    installed on the system before building the KSC applications. The source code of the
    :math:`\texttt{leuven}` library can be obtained from `its git repository <https://github.com/mnovak42/leuven>`_ 
    and detailed instructions on its requirements and installation are provided 
    in the `Build and Install Section <https://leuven.readthedocs.io/en/latest/main/install.html>`_ 
    of its `Documentation <https://leuven.readthedocs.io/en/latest/>`_. 


Quick start
------------

The source codes of the KSC applications are located under the :math:`\texttt{leuven/apps/examples/KSC}`
directory of the :math:`\texttt{leuven}` library :cite:`libleuven`. It can be built and 
installed as ::

  bash-3.2$ cd leuven/apps/examples/KSC
  bash-3.2$ mkdir build
  bash-3.2$ cd build
  bash-3.2$ cmake ../ -DCMAKE_INSTALL_PREFIX=/where/to/install -Dleuven_DIR=/where/leuven/is/installed/lib/cmake/leuven/ 
  bash-3.2$ make install
  ...

Then the KSC application for `training`, `hyper parameter tuning` and `out-of-sample extension`,
located under the :math:`\texttt{/where/to/install/bin}` are ready to be used to solve 
clustering problems. See for example :ref:`Test1 <sec_test1>` for a simple clustering application.


More details
-------------

The source codes of the KSC applications are located under the :math:`\texttt{leuven/apps/examples/KSC}`
directory of the :math:`\texttt{leuven}` library :cite:`libleuven`. This :math:`\texttt{KSC}` 
sub-directory (as all the others under :math:`\texttt{leuven/apps/examples/}`) can be 
freely copied to any other location and the corresponding applications can be built 
and used independently from the others. 

After making sure that all the above requirements are installed on the system, the 
KSC applications can be built and installed as (assuming that the current directory 
is the :math:`\texttt{leuven/apps/examples/KSC}` or wherever it has been copied) ::

  bash-3.2$ mkdir build
  bash-3.2$ cd build
  bash-3.2$ cmake ../ -DCMAKE_INSTALL_PREFIX=/where/to/install/ -Dleuven_DIR=/where/leuven/is/installed/lib/cmake/leuven/
  bash-3.2$ make install
  ...

Note, that the following :math:`\texttt{CMake}` configuration variables are used 
above 

 - ``-DCMAKE_INSTALL_PREFIX`` : the location, where the KSC applications are supposed 
   to be installed after a successful build
 - ``-Dleuven_DIR`` : the location, where the :math:`\texttt{leuven}` library has 
   been installed. Actually, the ``-Dleuven_DIR`` variable should point to the directory
   where the :math:`\texttt{leuvenConfig.cmake}` configuration file is located.

Further :math:`\texttt{CMake}` options might also be used at this stage. For example, 
on my :math:`MacOS`, I have the :math:`\texttt{gcc/g++ gnu C/C++}` compilers installed 
(beyond :math:`\texttt{Clang}`)under the :math:`\texttt{/usr/local/bin/}` directory (version :math:`\texttt{9.2.0}`) 
that were used in all cases of the examples shown in this document. These can be specified as preferred C and C++ compilers by using the 
:math:`\texttt{-DCMAKE}\_\texttt{C}\_\texttt{COMPILER=/usr/local/bin/gcc-9}` and :math:`\texttt{-DCMAKE}\_\texttt{CXX}\_\texttt{COMPILER=/usr/local/bin/g++-9}` 
:math:`\texttt{CMake}` options (also the Fortran compiler as :math:`\texttt{-DCMAKE}\_\texttt{FC}\_\texttt{COMPILER=/usr/local/bin/gfortran}` 
if needed). 
      
      
Usage 
--------   
      
After successful build and install, the KSC applications are available under the 
:math:`\texttt{bin}` sub-directory of the install location (i.e. :math:`/where/to/install/bin`).
These can be used immediately to solve clustering problems. The detailed description of 
the applications can be found in the :ref:`ksc_example_application_sec_doc` Section while several examples are provided 
in the :ref:`ksc_test_and_application_sec_doc` Section. See for example :ref:`Test1 <sec_test1>`
for a simple clustering application (or follow the instructions given in :math:`\texttt{/where/to/install/tests/test1/Readme.md}`).

