"""
.. _generateData_test3:

Generate data for clustering
=============================

Auxiliary fuctions for generating data, distributed in different shapes, for 
clustering. The geneated clusters are tipically hard to detect and separate 
(e.g. 4 intertwined spirals or concentric rings) for classical clustering 
algorithms. The provided functionalities are used by :ref:`Test3 <sec_test3>`.


Description
-----------

Data are generated in 2 dimensions, clustered around 4 centers with different 
`shapes`: 

 - ``4Clusters`` : distributed **normally around** the 4, random **point** centers
 - ``4Moons``    : distributed around 4 **moon-shaped** centers 
 - ``4Circles``  : distributed around 4 **concentric circles** as centers
 - ``4Spirals``  : distributed around 4 **intertwined spirals** as centers

These can be selected ny providing one of the above strings as input ragument.
The generated data will be **shuffled, standardised**. Sub-sets for `training` 
and `validation` will aslo be selected accoridng to the sizes given as input 
arguments. The generated `data set`, with the corresponding `labels` as well as 
the `training` and `validation` sets will be saved into files at the location 
specified by the corresponding input argument (see :ref:`Example <ex_generateData>`).


.. _ex_generateData:

Example
-------

The following example generates 100 000 data in 2 dimensions as 4 concentric 
rings. The data will be shuffled, standardised and saved to the 
:math:`\\texttt{output/data}\_\\texttt{4Circles.dat}` file together with the 
:math:`\\texttt{output/data}\_\\texttt{4Circles}\_\\texttt{Train}\_\\texttt{N20000.dat}` and 
:math:`\\texttt{output/data}\_\\texttt{4Circles}\_\\texttt{Valid}\_\\texttt{N20000.dat}` files containing the 
20 000 and 80 0000 sub-sampled data for training and validation ::

   GenData('4Circles', 'output', nSamples=100000, nTrain=20000, nValid=80000)
  
Plot of the generated data is not required. 

-----

"""


import numpy as np
import sklearn.datasets
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from itertools import cycle, islice



## Generate a random n-class classification problem (with n=4)
def genKClusters (nSamples, nFeatures=2, nClusters=4, levSeparation=2, rndseed=0):
    """
    Generates data for clustering that are normally distributed around the centers.

    Args:     
        nSamples  (int): number of sample points to generate
        nFeatures (int): dimensions of the data points
        nClusters (int): number of clusters to generate
        levSeparation (float): level of cluster separation
        rndseed   (int): state of the random number generator

    Returns:
        (:obj:`numpy::array`, :obj:`numpy::array`) : tuple containing the generated data and their labels
    """
    return sklearn.datasets.make_classification(
              n_samples            = nSamples, 
              n_features           = nFeatures, 
              n_classes            = nClusters, 
              n_clusters_per_class = 1,
              class_sep            = levSeparation,
              flip_y               = 0,
              n_informative        = nFeatures, 
              n_redundant          = 0,
              n_repeated           = 0,
              random_state         = rndseed)

## Generate 4 moons 
def genKMoons (nSamples, levNoise=0.1, rndseed=0):
    """
    Generates data for clustering: 4, `moon-shape` clusters in 2 dimension.

    Args:     
        nSamples  (int)  : number of sample points to generate
        levNoise (float) : determines the `thickness` of the `moons`
        rndseed   (int)  : state of the random number generator

    Returns:
        (:obj:`numpy::array`, :obj:`numpy::array`) : tuple containing the generated data and their labels
    """    
    nHalf = int(nSamples*0.5)
    xData, yData = sklearn.datasets.make_moons (
      n_samples            = nHalf,
      noise                = levNoise, 
      random_state         = rndseed)
    xData[np.where(yData[:]==0),1] *= 2.5
    xData[np.where(yData[:]==1),1] *= 2.5
    #
    d1 = sklearn.datasets.make_moons (
      n_samples            = nSamples-nHalf, 
      noise                = levNoise,
      random_state         = rndseed+12345678)
    # rotate, translate scale 
    tt = 90.0/180.0*3.1415  
    rot = np.array([[np.cos(tt),-np.sin(tt)],[np.sin(tt),np.cos(tt)]])  
    d2 = np.matmul(d1[0],rot)
    d2[np.where(d1[1]==0),0] += 2.0   # push blue to +x
    d2[np.where(d1[1]==0),1] += 0.25  # 
    d2[np.where(d1[1]==0),1] *= 3 
    d2[np.where(d1[1]==1),1] += 1.25  
    d2[np.where(d1[1]==1),0] -= 1.5   # push red to -x
    d2[np.where(d1[1]==1),1] *= 3  
    # concatenate
    xData = np.concatenate((xData,d2))
    yData = np.concatenate((yData, d1[1]+2))
    return xData, yData

def genKCircles(nSamples, levNoise=0.075, rndseed=0):
    """
    Generates data for clustering: 4, concentric rings in 2 dimension.

    Args:     
        nSamples  (int)  : number of sample points to generate
        levNoise (float) : determines the `thickness` of the rings
        rndseed   (int)  : state of the random number generator

    Returns:
        (:obj:`numpy::array`, :obj:`numpy::array`) : tuple containing the generated data and their labels
    """        
    nHalf = int(nSamples*0.5)
    xData, yData = sklearn.datasets.make_circles(
      n_samples    = nHalf, 
      factor       = 0.5,
      noise        = levNoise,
      random_state = rndseed)     
    xData[np.where(yData[:]==0)] *= 6
    xData[np.where(yData[:]==1)] *= 4 
    xd, yd = sklearn.datasets.make_circles(
      n_samples    = nSamples-nHalf, 
      factor       = 0.2,
      noise        = levNoise*0.25,
      random_state = rndseed+12345678)
    xd[np.where(yd[:]==0)] *= 8
    xd[np.where(yd[:]==1)] *= 2
    #
    xData = np.concatenate((xData, xd))
    yData = np.concatenate((yData, yd+2))
    return xData, yData


def genK4Spirals(nSamples, levNoise=0.1, rndseed=0):
    """
    Generates data for clustering: 4, intertwined spirals in 2 dimension.

    Args:     
        nSamples  (int)  : number of sample points to generate
        levNoise (float) : determines the `thickness` of the spirals
        rndseed   (int)  : state of the random number generator

    Returns:
        (:obj:`numpy::array`, :obj:`numpy::array`) : tuple containing the generated data and their labels
    """    
    clusterSize = (int)(nSamples/4);
    cl    = np.array([0,1,2,3]) 
    xData = np.zeros((nSamples,2))
    yData = np.zeros(nSamples, dtype=np.int32)
    ## r
    r     = np.linspace(0.05, 1.0, clusterSize)
    cc    = 3.1415/2
    for c in cl:
        # theta
        t = np.linspace(c*cc, (c+5.0)*cc, clusterSize)
        np.add(t, np.random.normal(0, levNoise, clusterSize), out=t)
        #
        st = c*clusterSize
        ed = (c+1)*clusterSize
        rr = np.arange(st,ed)
        xData[rr,0] = r*np.sin(t)
        xData[rr,1] = r*np.cos(t)
        yData[rr]   = c
    return xData, yData



def GenData(name, outpath, nSamples=100000, nTrain=0, nValid=0, doPlot=False):
    """
    Generates any of the 4 (blobs, moons, rings, spirals) data sets for clustering.

    Generates 2D data with the required size in 4 clusters according to the 
    required shape of clusters. The data are shuffled and standardised. Data
    sets for training and validation, with the required sizes, are sub-
    sampled. The corresponding files are saved under the required location.

    Args:
        name      (str)  : one of {'4Clusters','4Moons','4Circles','4Spirals'}  
        outpath   (str)  : location where the geneated data files will be saved
        nSamples  (int)  : number of sample points to generate
        nTrain    (int)  : number of sample points for training
        nValid    (int)  : number of sample points for validation
        doPlot   (bool)  : flag to indicate if data should be plotted (visualised)

    Yields:        
        Files, saved under the specified location, containing
          
          - :math:`\\texttt{data}\_\\texttt{x.dat}`            : the complete data set (data points as rows)
          - :math:`\\texttt{data}\_\\texttt{x}\_\\texttt{Labels.dat}`    : the corresponding cluster labels (for each row)
          - :math:`\\texttt{data}\_\\texttt{x}\_\\texttt{Train}\_\\texttt{Ny.dat}` : the `y` sub-sampled data for training
          - :math:`\\texttt{data}\_\\texttt{x}\_\\texttt{Valid}\_\\texttt{Nz.dat}` : the `z` sub-sampled data for validation
        
        where `x` is one of the available data set names.  
        The generated data are also plotted in case it was required. 
    """        
    if nTrain+nValid > nSamples:
        print("****  Error: nTrain + nValid = ", nTrain, " + ", nValid, " = ", nTrain+nValid, " > nSample = ", nSample)
        return 
    if   name == '4Clusters':
         xData, yData = genKClusters(nSamples)   
    elif name == '4Moons':
         xData, yData = genKMoons(nSamples)
    elif name == '4Circles':
         xData, yData = genKCircles(nSamples)
    elif name == '4Spirals':
         xData, yData  = genK4Spirals(nSamples)
    else :
        print("**** Error: unknown data set name ' ", name, " ' in generateData::GenData")
        return
    ## shuffle to get random examples (store the state and set back)
    np.random.seed(12345678)
    st0 = np.random.get_state()
    np.random.shuffle(xData)
    np.random.set_state(st0)
    np.random.shuffle(yData)
    ## sandardise
    xData = StandardScaler().fit_transform(xData)
    ## ouput name 
    fNameDat      = ''.join([outpath,'/data_',name,'.dat'])
    fNameLab      = ''.join([outpath,'/data_',name,'_Labels.dat'])
    fNameDatTrain = ''.join([outpath,'/data_',name,'_Train_N',str(nTrain),'.dat'])
    fNameDatValid = ''.join([outpath,'/data_',name,'_Valid_N',str(nValid),'.dat'])
    np.savetxt(fNameDat, xData, fmt='%14.4e')
    np.savetxt(fNameLab, yData, fmt='%d')
    if nTrain > 0:
        np.savetxt(fNameDatTrain, xData[0:nTrain,]     , fmt='%14.4e')
    if nValid > 0:    
        np.savetxt(fNameDatValid, xData[nTrain:nTrain+nValid,], fmt='%14.4e')
    ##
    ## if plotting was required
    if doPlot:
        nClusters = 4
        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                              int(nClusters + 1))))
#        print (colors)                                      
        # add black color for outliers (if any)
        colors = np.append(colors, ["#000000"])
        plt.scatter(xData[:, 0], xData[:, 1], s=1, color=colors[yData])
        plt.show()


#GenData('4Circles', 'output', nSamples=100000, nTrain=20000, nValid=80000)