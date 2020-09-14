
Introduction
=============

This is the documentation and user guide of some **sparse Kernel Spectral Clustering(KSC)**
applications developed based on the :math:`\texttt{leuven}` library and framework 
:cite:`libleuven` (:math:`\texttt{leuven}` library `git repository <https://github.com/mnovak42/leuven>`_ 
and `Documentation <https://leuven.readthedocs.io/en/latest/>`_). 

The document contains instructions for building and using the applications 
developed for :ref:`training <sec_kscapp_training>` (see the :ref:`sec_kscapp_training` Section), 
:ref:`hyper parameter tuning <sec_kscapp_tuning>` (see the :ref:`sec_kscapp_tuning` Section) and 
:ref:`out of sample extension or testing <sec_kscapp_testing>` (see the :ref:`sec_kscapp_testing` Section) 
the **sparse KSC model**. The sparsity is achieved by the combination 
of **incomplete Cholesky decomposition (ICD)** based low rank approximation of the 
(training data) kernel matrix with the so-called reduced set method. The KSC model,
trained on a training set, can be used to cluster any unseen data thanks its *out-of-sample extension* 
capability. Before describing the applications, the ICD is discussed briefly in the 
:ref:`sec_icd` Section due to its strong influence to the final accuracy of the 
KSC model and the clustering result itself. 

The description of the applications is followed by a number of test problems that 
gives insight into the usage of the provided applications (using special toy problems). 
Auxiliary scripts, provided for visualising or evaluating the output of the 
applications e.g. results of the hyper parameter tuning or the clustering itself, 
are also demonstrated.

.. table:: Description of the provided tests.
    :widths: 10 35 48 42
    
    +--------------------+-----------------------------+------------------------------------------------------------+------------------------------------------+ 
    | Test               |    Problem                  | Description                                                |       Purpose                            |
    +====================+=============================+============================================================+==========================================+
    | | :ref:`sec_test1` | Pre-generated, N = 10 000,  | **Simple clustering problem**. Not only the data but the   | Verify if all applications works         |
    |                    | d = 2 dimensional samples   | scripts, for executing the KSC applications with the       | correctly.                               | 
    |                    | from 10,  well  separated   | appropriate input parameter values, are also provided.     |                                          |
    |                    | clusters.                   |                                                            |                                          |
    +--------------------+-----------------------------+------------------------------------------------------------+------------------------------------------+ 
    | | :ref:`sec_test2` | More flexible, dynamic data | **Problems with variable cluster overlaps**. The data      | Give a deeper insight into the nature    |
    |                    | generation (including number| sets are generated dynamically as part of the test with the| of the:                                  |
    |                    | of cluster centers, data set| possibility of configurations (number of cluster centers,  |                                          |
    |                    | size, dimension, etc.).     | separation of clusters, dimension, etc.). Shell scripts,   | - KSC algorithm trough the applications  | 
    |                    |                             | for the KSC applications, are also generated at the same   | - effects of using the ICD as kernel     |
    |                    |                             | time, with *initial* parameter values according to the     |   matrix approximation                   |                                                                                                       
    |                    |                             | the given data generation settings.                        | - behaviour and capability of the KSC    |                                                                                                              
    |                    |                             |                                                            |   algorithm and applications in case of  |
    |                    |                             |                                                            |   more and more  overlapping clusters    |                         
    +--------------------+-----------------------------+------------------------------------------------------------+------------------------------------------+
    | | :ref:`sec_test3` | Four different, pre-defined | Four **clustering problems, with variable complexity**,    | Demonstrate the capability of the        |  
    |                    | 2D clustering problems with | including **even very non-linear** problems that are known | underlying sparse KSC algorithm and the  |
    |                    | 4 cluster centers each.     | to be challenging such as the *intertwined spiral* problem.| related applications to solve even the   |
    |                    |                             | The complete set is automatic: data generation, KSC shell  | most challenging clustering problems in a|
    |                    |                             | script generation, clustering and result visualisation.    | very efficient way (both in terms of     |
    |                    |                             |                                                            | memory usage and computing speed).       |
    +--------------------+-----------------------------+------------------------------------------------------------+------------------------------------------+
