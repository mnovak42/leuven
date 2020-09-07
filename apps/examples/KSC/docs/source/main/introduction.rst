
Introduction
=============

This document contains description of some sparse Kernel Spectral Clustering(KSC)
applications based on the :math:`\texttt{leuven}` library/framework. 

The document contains instructions for building and using the applications 
developed for :ref:`training <sec_kscapp_training>` (:numref:`sec_kscapp_training`), 
:ref:`hyper parameter tuning <sec_kscapp_tuning>` (:numref:`sec_kscapp_tuning`) and 
:ref:`out of sample extension or testing <sec_kscapp_testing>` (:numref:`sec_kscapp_testing`) 
of the **sparse KSC model**. The sparsity is achieved by the combination 
of **incomplete Cholesky decomposition (ICD)** based low rank approximation of the 
(training data) kernel matrix with the so-called reduced set method. Before 
describing the applications, the ICD is discussed briefly in :numref:`sec_icd`
(:ref:`sec_icd`) taking into 
account its strong influence to the final accuracy of the KSC model and the 
clustering results itself. 

Description of the applications is followed by a number of test problems that 
gives the possibility to illustrate the usage of the applications on some special 
toy problems. Auxiliary scripts, provided for visualising or evaluating the 
output of the applications e.g. results of the hyper parameter tuning or the 
clustering itself, are also demonstrated.

.. table:: Description of the provided tests.
    :widths: 10 35 48 42
    
    +--------------------+-----------------------------+------------------------------------------------------------+-----------------------------------+
    | Test               |    Problem                  |    Description                                             |       Purpose                     |
    +====================+=============================+============================================================+===================================+
    | | :ref:`sec_test1` | Pre-generated, N = 10 000,  |  *Simple clustering problem*. Not only the data but the    | Running the applications with the | 
    |                    | d = 2 dimensional samples   |  scripts, for running the applications with the appropriate| provided shell scripts on the     | 
    |                    | from 10, (linearly) well    |  input parameter values, are also provided.                | provided data to verify if all    |
    |                    | separated clusters.         |                                                            | the application works correctly.  |
    +--------------------+-----------------------------+------------------------------------------------------------+-----------------------------------+ 
    | | :ref:`sec_test2` |  More flexible              |   Balskdn aldlasl aldsk                                    |                                   |
    +--------------------+-----------------------------+------------------------------------------------------------+-----------------------------------+

