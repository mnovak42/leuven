"""
.. _evaluateClusteringRes_doc:

Evaluate clustering result
===========================

Auxiliary script for evaluating the results of the clustering. 

After executing the KSC application for testing or training, the assigned  
cluster lables are available in a file (specified as input argument of the KSC
applications). This script can be used to evaluate this result using either 
the true labels (if available) or the data that have been clustered. The 
**Adjusted Rand Index (ARI)** can be computed in the first while the 
**Silhouette Score (SC)** in the second case.


Example
-------

If the clustering result is located in the in the :math:`\\texttt{out/CRes.dat}`
file and the true cluster labels are available and located in the 
:math:`\\texttt{data/data}\_\\texttt{Labels.dat}` file, then the ARI can be 
computed as ::

    python ../utils/evaluate.py -c out/CRes.dat -t data/data_Labels.dat 

In case the true cluster labels are not available and the data used to cluster 
is located in the :math:`\\texttt{out/data.dat}` file, one can use the script to 
compute the SC as ::

    python ../utils/evaluate.py -c out/CRes.dat -d data/data.dat -s 

.. note:: Computing the **Silhouette Score** might take a long time in case of 
   lage data sets.
  
"""

import os
import sys
import getopt

import numpy as np

from sklearn.metrics    import silhouette_score
from sklearn.metrics    import adjusted_rand_score



## cdata - data that was clustered
## cres  - results of clustering
## truel - the corresponding true lables
def EvaluateClustering(cResFile, trueLFile, dataFile, compShil):
    if cResFile != '':
        cres  = ((np.loadtxt(cResFile))[:,0]).astype(int).flatten()
    if trueLFile != '':
        truel = np.loadtxt(trueLFile).astype(int).flatten()
    if dataFile != '':
        cdata = np.loadtxt(dataFile)
    ##
    ##  compute the silhouette_score:requires the data and the lables from clustering
    if compShil:
        print("   ---- (Python) --- : Computing Silhouette-Score ... (can take long...)")
        silhouetteScore = silhouette_score(cdata, cres, metric='euclidean')
        print ("    ===> The Silhouette-Score    =  {0:5.3f}".format(silhouetteScore))
    ## compute the adjusted rand index: requires true and clustering results labels
    if trueLFile != '':             
        print("   ---- (Python) --- : Computing Adjusted-Rand-Score ...")
        adjustedRandScore = adjusted_rand_score(truel, cres)
        print ("    ===> The Adjusted Rand-Score =  {0:5.3f}".format(adjustedRandScore))        
################################################################################



def main(argv):
    cResFile   = ''
    trueLFile  = ''     # optional
    dataFile   = ''     # optional
    compShil   = False  # requires data and cres    
    try:
        opts, args = getopt.getopt(argv,"hc:t:d:s")
    except getopt.GetoptError:
        print ('evaluateClusteringRes.py -c <cluster label file: result> [-t cluster label file: true] [-d data file] [-s compute silhouette score]')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('evaluateClusteringRes.py -c <cluster label file: result> [-t cluster label file: true] [-d data file] [-s compute silhouette score] [-x compute complete table]')
            sys.exit()
        elif opt in ("-c"):
            cResFile  = arg
        elif opt in ("-t"):
            trueLFile = arg
        elif opt in ("-d"):
            dataFile  = arg
        elif opt in ("-s"):
            compShil = True
    print(" ==== (Python) === : Evaluating clustering result ...")
    if cResFile == '':
        print(" -c <cluster label file: result> is a required argument")
        exit()
    if compShil and dataFile == '':
        print(" [-d data file] (clustered data) is required for the silhouette score")
        exit()
    if not compShil and trueLFile == '':
        print(" [-t cluster label file: true] is required for the adjusted rand score")
        exit()        
    
    EvaluateClustering(cResFile, trueLFile, dataFile,  compShil)
################################################################################



if __name__ == "__main__":
    main(sys.argv[1:])

