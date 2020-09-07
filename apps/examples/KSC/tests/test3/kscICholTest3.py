"""
Test3: script
==============

This script generates data for four different clustering problems, 
invokes the KSC application for clustering each and generates plots to visualise
the results (located under the `:math:\texttt{res}` directory at the end). This 
is the only script for :ref:`Test3 <sec_test3>`.

See :ref:`generateData_test3` for more details on the data generation. 


Example
-------

:: 

    bash-3.2$ python kscICholTest3.py
    ==== (Python) === : clustering the  4Circles  data set ...
    ==== (Python) === : generating decision boundary for the  4Circles  data set ...
    ==== (Python) === : clustering the  4Clusters  data set ...
    ==== (Python) === : generating decision boundary for the  4Clusters  data set ...
    ==== (Python) === : clustering the  4Moons  data set ...
    ==== (Python) === : generating decision boundary for the  4Moons  data set ...
    ==== (Python) === : clustering the  4Spirals  data set ...
    ==== (Python) === : generating decision boundary for the  4Spirals  data set ...
  

"""


import os
import sys
import getopt


import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.append('../utils/')
import generateData


def Help():
    print ('kscICholTest3.py [-v print KSC application output] [-h help]')


def main(argv):
    ## get possible input args 
    verbose = False
    try:
        opts, args = getopt.getopt(argv,"hv")
    except getopt.GetoptError:
        Help()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            Help()
            sys.exit()
        elif opt in ("-v"):
            verbose = True
    ##
    ## clean /output /res
    os.system('rm -f -r output')
    os.system('mkdir output')
    os.system('rm -f -r res')
    os.system('mkdir res')

    ## Path to the KscIchol_Test application 
    pathTo_KscIchol_Test = '../../bin/KscIchol_Test'

    ##
    ## The dictionary for generating the KscIchol_Test input parameters
    scrip_kscIchol_Test = {}
    # the main program:
    scrip_kscIchol_Test['main'] = pathTo_KscIchol_Test
    # the incomplete Cholesky decomposition related:
    scrip_kscIchol_Test['--icholTolError']     = 0.6
    scrip_kscIchol_Test['--icholMaxRank']      = 800
    scrip_kscIchol_Test['--icholRBFKernelPar'] = 0.005
    scrip_kscIchol_Test['--icholRedSetFile']   = 'output/ReducedSetData.dat'
    # the training data set related:
    scrip_kscIchol_Test['--trDataNumber']      = 20000
    scrip_kscIchol_Test['--trDataDimension']   = 2
    scrip_kscIchol_Test['--trDataFile']        = 'output/data_4Circles_Train_N200000.dat'
    # the test data set related:
    scrip_kscIchol_Test['--tstDataNumber']     = 100000
    scrip_kscIchol_Test['--tstDataFile']       = 'output/data_4Circles.dat'
    # the clustering related: 
    scrip_kscIchol_Test['--clNumber']          = 4
    scrip_kscIchol_Test['--clRBFKernelPar']    = 0.0021
    scrip_kscIchol_Test['--clEncodingScheme']  = 'BAS'
    scrip_kscIchol_Test['--clEvalWBalance']    = 0.1
    scrip_kscIchol_Test['--clLevel']           = 1
    scrip_kscIchol_Test['--clResFile']         = 'output/CRes.dat'
    
    scrip_kscIchol_Test["--verbosityLevel"]    = 1

    ##
    ## replace relative with absolute paths
    scrip_kscIchol_Test['main']        = os.path.abspath(scrip_kscIchol_Test['main'])
    scrip_kscIchol_Test['--clResFile'] = os.path.abspath(scrip_kscIchol_Test['--clResFile'])


    ##
    dataSetSize  = 100000
    dataSetTrain = 20000
    scrip_kscIchol_Test['--trDataNumber']      = dataSetTrain
    scrip_kscIchol_Test['--tstDataNumber']     = dataSetSize
    #
    theDataSets = ['4Circles', '4Clusters', '4Moons', '4Spirals']
    #theDataSets = ['4Spirals']
    theScripts  = {}
    ## 4Circles 
    dataSetName  = '4Circles'
    scrip_kscIchol_Test['--icholRBFKernelPar'] = 0.007
    scrip_kscIchol_Test['--trDataFile']        = os.path.abspath(''.join(['output/data_',dataSetName,'_Train_N',str(dataSetTrain),'.dat']))
    scrip_kscIchol_Test['--tstDataFile']       = os.path.abspath(''.join(['output/data_',dataSetName,'.dat']))
    scrip_kscIchol_Test['--clRBFKernelPar']    = 0.0007
    theScripts['4Circles'] = scrip_kscIchol_Test.copy();
    ## 4Clusters
    dataSetName  = '4Clusters'
    scrip_kscIchol_Test['--icholRBFKernelPar'] = 0.032
    scrip_kscIchol_Test['--trDataFile']        = os.path.abspath(''.join(['output/data_',dataSetName,'_Train_N',str(dataSetTrain),'.dat']))
    scrip_kscIchol_Test['--tstDataFile']       = os.path.abspath(''.join(['output/data_',dataSetName,'.dat']))
    scrip_kscIchol_Test['--clRBFKernelPar']    = 0.0021
    theScripts['4Clusters'] = scrip_kscIchol_Test.copy();
    ## 4Moons
    dataSetName  = '4Moons'
    scrip_kscIchol_Test['--icholRBFKernelPar'] = 0.016
    scrip_kscIchol_Test['--trDataFile']        = os.path.abspath(''.join(['output/data_',dataSetName,'_Train_N',str(dataSetTrain),'.dat']))
    scrip_kscIchol_Test['--tstDataFile']       = os.path.abspath(''.join(['output/data_',dataSetName,'.dat']))
    scrip_kscIchol_Test['--clRBFKernelPar']    = 0.021
    theScripts['4Moons'] = scrip_kscIchol_Test.copy();
    ## 4Spirals
    dataSetName  = '4Spirals'
    scrip_kscIchol_Test['--icholRBFKernelPar'] = 0.0006  # 0.0006 ==> 0.0009
    scrip_kscIchol_Test['--icholTolError']     = 0.6     # 0.6    ==> 0.9
    scrip_kscIchol_Test['--trDataFile']        = os.path.abspath(''.join(['output/data_',dataSetName,'_Train_N',str(dataSetTrain),'.dat']))
    scrip_kscIchol_Test['--tstDataFile']       = os.path.abspath(''.join(['output/data_',dataSetName,'.dat']))
    scrip_kscIchol_Test['--clRBFKernelPar']    = 0.005   # 0.005  ==> 0.0016
    theScripts['4Spirals'] = scrip_kscIchol_Test.copy();



    ## ========================================================================== ##
    ## Run 
    for dataSetName in theDataSets:
        generateData.GenData(dataSetName, 'output', dataSetSize, dataSetTrain)
        # get script parameters
        #
        # generate the script and execute the test 
        sc_kscIchol_Test = theScripts[dataSetName]
        theScript  = sc_kscIchol_Test['main']+' '
        theScript += ' '.join((str(name)+' '+str(value)) for name,value in sc_kscIchol_Test.items())
        print (" ==== (Python) === : clustering the ", dataSetName, " data set ...")
#        os.system(theScript)
        resOut     = os.popen(theScript).read()
        if verbose:
            print(resOut)
        # find the reduced set size 
        part        = resOut.partition('---> Rank of the aprx :')[2]
        redSetSize  = part.split()[0]
        # find the training and test times
        part        = part.partition('---> Duration         :')[2]
        timeTr      = part.split()[0]
        timeTst     = (part.partition('---> Duration         :')[2]).split()[0]
        #print(" R        = ", redSetSize)
        #print(" Time-tr  = ", timeTr, " [s]")
        #print(" time-tst = ", timeTst, " [s]")
        ##
        #
        # load the data 
        data4Circles   = np.loadtxt(sc_kscIchol_Test['--tstDataFile'])
        data4Train     = np.loadtxt(sc_kscIchol_Test['--trDataFile'])
        dataReducedSet = np.loadtxt('output/ReducedSetData.dat')
        res4Circles    = np.loadtxt(sc_kscIchol_Test['--clResFile'])
        # rerun the test to generate the decision boundary plot
        print (" ==== (Python) === : generating decision boundary for the ", dataSetName, " data set ...")
        sc_kscIchol_Test["--tstDataFile"]    = os.path.abspath("dataSurf2D/xydata.dat")
        sc_kscIchol_Test["--tstDataNumber"]  = 62500
        sc_kscIchol_Test["--verbosityLevel"] = 0
        theScript = sc_kscIchol_Test['main']+' '
        theScript += ' '.join((str(name)+' '+str(value)) for name,value in sc_kscIchol_Test.items())
        os.system(theScript)
        dataDecSurfData = np.loadtxt(sc_kscIchol_Test['--tstDataFile'])
        dataDecSurfRes  = np.loadtxt(sc_kscIchol_Test['--clResFile'])        
        #
        #
        nClusters = 4
        listColors = list(['b', 'r', 'g', 'k', 'y', 'm'])
        cmp  = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', listColors[0:nClusters], nClusters)
        norm = mpl.colors.BoundaryNorm(np.linspace(0, nClusters, nClusters+1), nClusters)

        fxSize  = 6
        fySize  = 6
        if dataSetName == '4Moons':
            fySize = 5
        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(fxSize,fySize))
        # Remove horizontal space between axes
        #fig.subplots_adjust(hspace=0, wspace=0)
        for r,c in np.ndindex(axs.shape):
            axi = axs[r,c]
            axi.tick_params(axis = "x", which = "both", bottom = False, top   = False, labelbottom = False)
            axi.tick_params(axis = "y", which = "both", left   = False, right = False, labelleft   = False)    
            axi.set_xlim(-2.5,2.5)
            axi.set_ylim(-2.5,2.5)
            #axi.set_aspect('equal')

        ## plot the training data over a decision surface
        xValues = np.loadtxt('dataSurf2D/xdata.dat')
        yValues = np.loadtxt('dataSurf2D/ydata.dat')
        xGrid, yGrid = np.meshgrid(xValues, yValues)
    #    cp_filled = plt.contourf(xGrid, yGrid, np.transpose((dataDecSurfRes[:,1]).reshape(1000,1000)), levels=50, cmap='gray')
        axs[1,1].contourf(xGrid, yGrid, np.transpose((dataDecSurfRes[:,0]).reshape(250,250)), levels=nClusters, cmap=cmp)#, norm=norm)
        axs[1,1].contour(xGrid, yGrid, np.transpose((dataDecSurfRes[:,0]).reshape(250,250)), levels=nClusters, linewidths=1.0, colors='gray')
    #    axs[1,1].scatter(data4Circles[:,0], data4Circles[:,1], edgecolors='none', s=1.0, c=res4Circles[:,0], cmap=cmp, norm=norm)
            
        ## plot the clustering results on the whole test data set
        axs[1,0].scatter(data4Circles[:,0], data4Circles[:,1], edgecolors='none', s=1.0, c=res4Circles[:,0], cmap=cmp, norm=norm)
        axs[1,0].text(-2.4, +2.2, r'test', fontsize=8)
        axs[1,0].text(-2.4, -2.2, r'$N = 10^5$', fontsize=8)
        tx = 'time: ' + '{:.2f}'.format(float(timeTst)) + ' [s]'
        axs[1,0].text(-2.4, -2.4, tx, fontsize=8)
        
        ## plot the training data + the reduced set
        axs[0,1].scatter(data4Train[:,0], data4Train[:,1], edgecolors='none', s=0.5, c='gray')
        axs[0,1].scatter(dataReducedSet[:,0], dataReducedSet[:,1], edgecolors='none', s=6, c='r')
        axs[0,1].text(-2.4, +2.2, 'training', fontsize=8)
        axs[0,1].text(-2.4, -2.0, r'$N_{tr} = 2\times 10^4$', fontsize=8)
        tx = 'R = ' + redSetSize
        axs[0,1].text(-2.4, -2.2, tx,  fontsize=8)
        tx = 'time: ' + '{:.2f}'.format(float(timeTr)) + ' [s]'
        axs[0,1].text(-2.4, -2.4, tx,  fontsize=8)
        
        ## plot the test data set
        axs[0,0].scatter(data4Circles[:,0], data4Circles[:,1], edgecolors='none', s=0.5, c='gray')
        axs[0,0].text(-2.4, +2.2, r'data set: $N = 10^5$', fontsize=8)
        
        ####  
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)    
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        figName = 'res/fig_'+dataSetName+'.eps'
        plt.savefig(figName, bbox_inches = 'tight', pad_inches = 0, format='eps', quality=30)
    #    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
