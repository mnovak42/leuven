"""
Plotting results of the KSC model tuning
========================================

Functionalities for visualising the results of KSC hyper parameter tuning. 


Description
-----------

The ``KscIchol_Tune`` application 

 - **trains** a sparse **KSC model** at each points of a 2D grid of candidate 
   `cluster numbers` and RBF `kernel parameter values` **on the training set**
 - each of the traind KSC models are **evaluated on validation set** and the 
   model selection criterion value is reported
 - the 2D grid data as well as the corresponding model selection criterion values 
   are written into files as final results of the hyper parameter tuning

The output of the hyper parameter tuning application can be visualised by this 
application. The application reads the output files of the tuning application 
and generates 

 - **a 2D image plot** of the model selection criterion values(its maximum 
   scalled to 1) over the `cluster number` and `RBF kernel parameter` value grid  
 - **a projection** of this 2D image to the `cluster number` values containing 
   `the maximum model selection criterion value` at each cluster number values 
   (i.e. the maximum values over the kernel parameters at each cluster number)
 - see more at the :ref:`Example <example_plotResTuning>` below 


.. _example_plotResTuning:

Example:
--------

Example for calling the ``plotResTuning`` function for visualising the results 
of the ``KscIchol_Tune`` application written to the 
:math:`\\texttt{TuningRes}` files ::


   $ python plotResTuning.py -f TuningRes


Note, that the output file names of the ``KscIchol_Tune`` 
application can be set by its :math:`\\texttt{--resFile}` input argument e.g. 
:math:`\\texttt{--resFile TuningRes}` that will generate 3 files containing

 - :math:`\\texttt{TuningRes\_clusterNumbers}` : the cluster number values
 - :math:`\\texttt{TuningRes\_kernelPars}`     : the RBF kernel parameter values
 - :math:`\\texttt{TuningRes}`                 : model selection criteria over this 2D grid

The title of the generated plots can be set by providing an additional argument 
with :\math:`\\texttt{-title}` option. The plots can be saved to a location 
specified by an additional :\math:`\\texttt{-saveTo}` argument. The 2D image 
plot is smoothed by default to get nicer images. The level of this smoothing can 
be set by the :\math:`\\texttt{-s}` argument and can be turned off by setting 
it to 1. The following example is used in case of :ref:`Test2 <sec_test2>` ::

   python ../utils/plotResTuning.py -f out/TuningRes.dat -s 1 --title "Test2 tuning (stdv = 2.0)" --saveTo "out/fig" 


-----

"""

import os
import sys
import getopt


import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage


def PlotResTuning2D(clusterNumbers, kernelParams, resTuneMatrix, smooth, title, saveTo):
    fig = plt.figure(figsize=(6,5))
    left, bottom, width, height = 0.12, 0.1, 0.9, 0.8
    ax = fig.add_axes([left, bottom, width, height]) 
    #
    xValues0 = clusterNumbers
    yValues0 = kernelParams 
    zValues0 = resTuneMatrix
    # apply c.spline interpolation on the input data to smooth
    zoomVal = smooth
    xValues = scipy.ndimage.zoom(xValues0, zoomVal)
    yValues = scipy.ndimage.zoom(yValues0, zoomVal)
    zValues = scipy.ndimage.zoom(zValues0, zoomVal)
    # reset: can be lost due to smoothing 
    yValues[-1] = yValues0[-1]
    # create the X,Y grid which the z-matrix is plotted
    xGrid, yGrid = np.meshgrid(xValues, yValues)
    # scale the z-matrix into [0,1]
    zMax = zValues.max()
    zMin = zValues.min()
    zValues  -= zMin
    zValues  /= (zMax-zMin) 
    #
    # 1. plot the 2D contour
    cp_filled = plt.contourf(xGrid, yGrid, zValues, levels=50, cmap='jet')
    cp_lines  = plt.contour (xGrid, yGrid, zValues, levels=50, colors='k', linewidths=0.2)
    plt.colorbar(cp_filled, ticks=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    # set logscale on kernel parameters, labels and title axes
    plt.grid(which='major',axis='both',color='grey', linestyle='--', linewidth=0.2)
    #
    plt.xlim( clusterNumbers.min(), clusterNumbers.max() )
#    plt.xlim(15,25)
    #
    labelFontSize = 12
    ax.set_yscale('log') 
    if smooth > 1:
        if title != ' ':
            ax.set_title(title+"\n(scaled to [0,1] and smoothed)")
        else :
            ax.set_title("(scaled to [0,1] and smoothed)")
    else :
        if title != ' ':
            ax.set_title(title+"\n(scaled to [0,1])")    
        else :
            ax.set_title("(scaled to [0,1])")    
    ax.set_xlabel('number of clusters'    , fontsize=labelFontSize)
    ax.set_ylabel('kernel parameter value', fontsize=labelFontSize)
    ##
    kerParAtMaxVals = np.zeros((clusterNumbers.shape[0],)) 
    for ic in range(zValues0.shape[1]):
        kerParAtMaxVals[ic] = kernelParams[np.argmax(zValues0[:,ic])]
    plt.plot(clusterNumbers, kerParAtMaxVals, 'ko')
    ##
    # save (if file name was given) or show the plot
    if saveTo!=' ':
        saveTo = saveTo+'_2D.eps'
        plt.savefig(saveTo, format='eps')
    else:
        plt.show()

# which = 0 : maximum values of the QM at each number of kernels
# which = 1 : maximum values of the QM at each kernel maparameters
def PlotResTuningMaxAt(clusterNumbers, kernelParams, resTuneMatrix, which, title, saveTo=' '):
    fig = plt.figure(figsize=(6,5))
    left, bottom, width, height = 0.12, 0.1, 0.8, 0.8
    ax = fig.add_axes([left, bottom, width, height]) 
    #
    zValues = resTuneMatrix
    if which == 0:
        xValues = clusterNumbers 
        maxProj = np.amax(zValues, axis=0)
        xLabel  = 'number of clusters'
    else :
        xValues = kernelParams 
        maxProj = np.amax(zValues, axis=1)
        xLabel  = 'kernel parameters'
    #
#    print(xValues)
#    print(np.diff(xValues))
    plt.plot(xValues, maxProj, 'bo')#, clusterNumbers, maxProj, 'k--')
    if which==0:
        plt.bar(xValues, maxProj, color=(0.955, 0.955, 0.955),  edgecolor='blue')
    else :
        widths = np.zeros(xValues.size, dtype=float)
        widths = np.diff(xValues)
        widths = np.append(widths, 2*widths[-1]-widths[-2])
#        print(widths.size," ",xValues.size, " ", maxProj.size)
        plt.bar(xValues, maxProj, width=widths, align="edge", color=(0.955, 0.955, 0.955),  edgecolor='blue')
    plt.ylim( max(0,(maxProj.min()*0.97)), min(1.01,(maxProj.max()*1.05)) )
    #plt.bar(clusterNumbers, maxProj, align='center')
    labelFontSize = 12
    if which!=0:
        ax.set_xscale('log') 
    if title != ' ':
        ax.set_title(title)
    ax.set_xlabel(xLabel                              , fontsize=labelFontSize)
    ax.set_ylabel('maximum KSC model evaluation value', fontsize=labelFontSize)
    # save (if file name was given) or show the plot
    if saveTo!=' ':
        saveTo = saveTo+'_1D.eps'
        plt.savefig(saveTo, format='eps')
    else:
        plt.show()


def Help():
    print ('plotResTuning.py -f <tuning results files> [s smoothing(>= 1)] [-title title of the plots]', 
           '[-saveTo save plots where]')


def main(argv):
    fname  = ' '
    title  = ' '
    saveTo = ' '
    smooth = 5
    try:
        opts, args = getopt.getopt(argv,"hf:s:",["smooth=","title=","saveTo="])
    except getopt.GetoptError:
        Help()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            Help()
            sys.exit()
        elif opt in ("-f"):
            fname  = os.path.splitext(arg)[0] # file name without extension
        elif opt in ("-s,--smooth"):
            smooth = max(1,int(arg))
        elif opt in ("--title"):
            title  = arg
        elif opt in ("--saveTo"):
            saveTo = arg
    #            
    if fname == ' ':
        Help()
        print(" Results of the tuning needs to be provided with the -f argument.")
        sys.exit()                
    dataZ  = np.loadtxt(fname+'.dat', dtype=float)
    dataX  = np.loadtxt(fname+'_clusterNumbers.dat', dtype=float)
    dataY  = np.loadtxt(fname+'_kernelPars.dat', dtype=float)
    # replace NaN-s with he mean
    theMean = np.nanmean(dataZ)
    dataZ[np.isnan(dataZ)] = theMean
    #
    PlotResTuning2D(dataX, dataY, dataZ, smooth, title, saveTo=saveTo)
    PlotResTuningMaxAt(dataX, dataY, dataZ, which=0, title=title, saveTo=saveTo)
  


    
if __name__ == "__main__":
    main(sys.argv[1:])


