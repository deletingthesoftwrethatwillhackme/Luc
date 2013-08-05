import numpy as np
from skimage import io, filter #@UnresolvedImport
import pandas as pandas
import os as os
from skimage.filter import canny #@UnresolvedImport
from scipy import ndimage
import matplotlib.pyplot as plt


def findCenters(labeled_plants, stackSum):
    center = {}
    for i in set(labeled_plants.flat):
        blob = np.where(labeled_plants==i)
        center[i] = [blob[0][np.argmax(stackSum[blob])], blob[1][np.argmax(stackSum[blob])]]
    center = pandas.Series(center)
    return center

def getCircle(center, radius=5):
    points = []
    for i in range(center[0]-radius, center[0]+radius+1):
        for j in range(center[1]-radius, center[1]+radius+1):
            if (i < 0) or (j < 0) or (i >= 240) or (j >=360): continue
            if np.sqrt(sum(pow(np.array(center[1]) - (i,j),2))):
                points.append([i,j])
    return np.array(points)

def detect_peaks(image):
    """
    http://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array

    Takes an image and detect the peaks using the local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """
    from scipy.ndimage.filters import maximum_filter
    from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2,2)

    #apply the local maximum filter; all pixel of maximal value 
    #in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood)==image
    background = (image==0)

    #a little technicality: we must erode the background in order to 
    #successfully subtract it form local_max, otherwise a line will 
    #appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, 
                                       border_value=1)
    detected_peaks = local_max - eroded_background
    peaks = np.array(np.where(detected_peaks)).T

    return peaks

def findPlantsCanny(stackVar, stackSum, showImages=True):
    edges = canny(stackVar)
    fill_stack = ndimage.binary_fill_holes(edges)
    label_objects, nb_labels = ndimage.label(fill_stack)
    sizes = np.bincount(label_objects.ravel())
    mask_sizes = sizes > 25
    
    for label in range(len(mask_sizes)):
        '''
        Get rid of lines in addition to the straight size threshold.
        '''
        pts = np.where(label_objects == label)
        xRange = (max(pts[0]) - min(pts[0]))
        yRange = (max(pts[1]) - min(pts[1]))
        areaCovered = float(len(pts[0])) / (xRange*yRange)
        if (areaCovered < .33) or (xRange < 3) or (yRange < 3):
            mask_sizes[label] = False

    mask_sizes[0] = 0
    plants_cleaned = mask_sizes[label_objects]
    labeled_plants, numPlants = ndimage.label(plants_cleaned)
    center = findCenters(labeled_plants, stackSum)
    
    if showImages:
        fig, axs = plt.subplots(1,3, figsize=(14,4), sharey=True)
        axs[0].imshow(stackVar)
        axs[1].imshow(stackVar, cmap=plt.cm.jet, interpolation='nearest') #@UndefinedVariable
        axs[1].contour(plants_cleaned, [0.5], linewidths=1.2, colors='y')
        axs[2].imshow(labeled_plants, cmap=plt.cm.spectral, interpolation='nearest') #@UndefinedVariable
        axs[2].scatter(np.array(center.tolist())[:,1], np.array(center.tolist())[:,0], 
                       color='grey')
        for ax in axs: ax.axis('off')
        fig.subplots_adjust(wspace=.01)
       
    return labeled_plants, center
    
def getFeatures(outDir, sizeThreshold=25, showImages=True):
    fList = os.listdir(outDir)
    fList = np.array(fList).take(np.argsort(map(lambda s: int(s.split('.')[0]), 
                                                fList)))
    s = np.array([io.imread(outDir+f, as_grey=False, plugin=None, flatten=None) 
                 for f in fList])
    stackSum = np.sum(s, axis=0)
    stackVar = np.var(s, axis=0)
    stackVar = filter.tv_denoise(stackVar, weight=300, eps=1e-5)
    stackSum = filter.tv_denoise(stackSum, weight=300, eps=1e-5)

    labeled_plants, center = findPlantsCanny(stackVar, stackSum)
    circles = center.map(getCircle)
    maxTrace = pandas.DataFrame([np.max(s[:,circles[i][:,0],circles[i][:,1]],axis=1) 
                               for i in set(labeled_plants.flat)]).T
    wholeRegion = pandas.DataFrame([np.max(s[:,np.where(labeled_plants==i)[0],
                                             np.where(labeled_plants==i)[1]],axis=1) 
                               for i in set(labeled_plants.flat)]).T
    traces = pandas.DataFrame([np.mean(s[:,circles[i][:,0],circles[i][:,1]],axis=1) 
                               for i in set(labeled_plants.flat)]).T
                               
    return labeled_plants, center, traces, maxTrace, wholeRegion

