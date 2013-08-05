'''
Created on May 15, 2012

@author: agross
'''
import numpy as np
from AutoLuc import io
import pandas as pandas
import matplotlib.pylab as plt

from Vis.CyclePlots2 import cyclePlot

def quadrantFilter(plate):
    '''
    Here I'm assuming that the movement of the plate isn't so drastic that 
    a plant gets moved across a quadrant.  I'm splitting each plate into a
    top half and a bottom half and checking if the ratio of pixel intensity
    for either is off compared to the other time points.  I'm doing the same 
    for the left and right portions of the plate.
    '''
    xDim, yDim = plate.shape[1:]
    topBottom = (1.*(plate[:,:xDim/2].sum(axis=1).sum(axis=1)) / 
                 plate[:,xDim/2:].sum(axis=1).sum(axis=1))
    leftRight = (1.*(plate[:,:,:yDim/2].sum(axis=1).sum(axis=1)) / 
                 plate[:,:,yDim/2:].sum(axis=1).sum(axis=1))
    leftRight = leftRight / np.median(leftRight)
    topBottom = topBottom / np.median(topBottom)
    goodSlices = ((leftRight > .2) * (leftRight < 5.) * (topBottom > .2) * 
                  (topBottom < 5.))
    return goodSlices

def flashFilter(plate):
    '''
    Check for large flash effect in first time point.
    Also check for one bad image killing the range for the rest.
    '''
    goodSlices = np.array([True for i in plate])
    totalLuc = plate.sum(axis=1).sum(axis=1).astype(np.float)
    ratio = totalLuc[0] / np.max(totalLuc[1:])
    if ratio > 2:
        goodSlices[0] = False
        
    maxPlate = np.argmax(totalLuc[1:])+1   
    if (totalLuc[maxPlate] / np.mean(sorted(totalLuc[1:])[-5:-1])) > 5:
        goodSlices[maxPlate] = False
    
    return goodSlices


def levelFilter(plate):
    '''
    A lot of bad images have much less signal that their neighbors.  Here I'm 
    tracking the total intensity over the whole plate.  If the intensity drops 
    10 fold from its last value, I throw out that plate.
    '''
    totalLuc = plate.sum(axis=1).sum(axis=1).astype(np.float)
    goodSlices = np.array([True for i in totalLuc])
    goodSlices[1:] = np.diff(totalLuc) / totalLuc[1:] > -3
    return goodSlices



def processImageSet(inPath, fList, runNum, plotMe=True):
    '''
    Pre-processes an image set by a set of automated filters to get rid of
    bad or redundant images. 
    
    returns a dictonary mapping (plate, qx, qy) to another dictionary containing the 
    plate data as well as the times-tamps for each plate, where qx, and qy define the
    quadrant.
    '''
    goodPlates = {}    
    filesForRun = io.getFilesForRun(runNum, fList);
    plates = io.getPlates(filesForRun, inPath)
    
    if plotMe:
        fig, axs = plt.subplots(2,2,figsize=(10,6))
    for i,j in [(0,0),(0,1),(1,0),(1,1)]:
        plate = plates[i,j]
        times = io.getTimesFromStack(inPath, filesForRun)
        uniqueImages = np.array(times.diff() != 0)
        goodSlices = (quadrantFilter(plate) * flashFilter(plate) * uniqueImages)
        plate = plate[goodSlices]
        times = times[goodSlices]
        
        goodSlices = levelFilter(plate)
        plate = plate[goodSlices]
        times = times[goodSlices]
        
        if(float(sum(goodSlices)) / len(filesForRun)) > .5:
            if plotMe:
                totalLuc = plate.sum(axis=1).sum(axis=1)
                totalLuc = pandas.Series(totalLuc, index=times)
                cyclePlot(totalLuc, ax=axs[i][j])
                axs[i][j].set_ylabel('Total Intensity')
                axs[i][j].set_yticks([])
            goodPlates[(runNum,i,j)] = dict(plate=plate, times=times)
        elif plotMe:
            totalLuc = plate.sum(axis=1).sum(axis=1)
            totalLuc = pandas.Series(totalLuc, index=times)
            cyclePlot(totalLuc, ax=axs[i][j])
            axs[i][j].annotate('X', (.5,.5), xycoords='axes fraction', 
                        size=50, ha='center', color='red')
    return goodPlates