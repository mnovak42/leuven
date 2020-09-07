"""
.. plotClusteringRes_doc:

Plot clustering result
======================

Auxiliary script for plotting the results of the clustering. Only for 2D and 3D
data. 

Example
-------

Plotting the result (located in the :math:`\\texttt{out/CRes.dat}`) of 
clustering the data in the :math:`\\texttt{out/data.dat}` file ::

  python ../utils/plotClusteringRes.py -d data/data.dat -l out/CRes.dat 

The same, but instead of plotting the cluster labels, the soft cluster 
membership indicator value (available only in case of AMS and BAS) is plotted ::

  python ../utils/plotClusteringRes.py -d data/data.dat -l out/CRes.dat -t 1
  
The same as above, but a figure file is generated as :math:`\\texttt{out/f}\_\\texttt{res.eps}` 
instead of the plot with the title of "the title" ::

  python ../utils/plotClusteringRes.py -d data/data.dat -l out/CRes.dat --title "The title" --saveTo out/f_res
  
"""

import os
import sys
import getopt

import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



def Plot(fdata, flabel, type, title, fsave): 
    print(" ==== (Python) === : visualising the result of the clustering...") 
    xyzData = np.loadtxt(fdata)
    numDim  = xyzData.shape[1]
    if not (numDim == 2 or numDim == 3):
        print("   ---- (Python) --- : Only 2D and 3D data can be visualised!")
        exit()
    labels  = np.loadtxt(flabel)
    clabels = labels[:,min(type,labels.shape[1])]
    fig = plt.figure()
    if numDim == 3:
        ax = fig.add_subplot(111, projection='3d')    
        sc = ax.scatter(xyzData[:,0], xyzData[:,1], xyzData[:,2], c=clabels, 
                        marker='.', s=0.2, cmap='jet')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Feature 3')        
    else :
        ax = fig.add_subplot(111)    
        sc = ax.scatter(xyzData[:,0], xyzData[:,1], c=clabels,
                        marker='.',  s=0.2, cmap='jet')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
    ## 
    if title != ' ':
        ax.set_title(title)
    #    
    plt.colorbar(sc)
    if fsave != '':
        plt.savefig(fsave+'.eps', format='eps')
    else:           
        plt.show()
################################################################################


def Help():
    print ('plotClusetringRes.py -d <data file> -l <cluster label file>', 
           '[-t col. of cluster label file {0}] [--title plot title]',
           '[--saveTo save as]')
################################################################################


def main(argv):
    fdata  = ''
    flabel = ''
    title  = ' '
    fsave  = ''
    type   = 0
    try:
        opts, args = getopt.getopt(argv,"hd:l:t:",["title=","saveTo="])
    except getopt.GetoptError:
        Help()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            Help()
            sys.exit()
        elif opt in ("-d"):
            fdata  = arg
        elif opt in ("-l"):
            flabel = arg
        elif opt in ("-t"):
            type   = arg
        elif opt in ("--title"):
            title  = arg
        elif opt in ("--saveTo"):
            fsave  = arg
    if fdata == '' or flabel == '':
        Help()
        print (" -d <data file> -l <cluster label file> are required arguments")
        sys.exit()
    Plot(fdata, flabel, int(type), title, fsave)   
 ################################################################################



if __name__ == "__main__":
    main(sys.argv[1:])

   
                                  