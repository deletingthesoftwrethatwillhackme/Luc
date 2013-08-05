'''
Created on May 15, 2012

@author: agross
'''
import numpy as np
import pandas as pandas
from skimage import io, filter


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
    y,x = np.where(detected_peaks > 0)
    peaks = map(tuple, np.array([x,y]).T)

    return peaks

def detect_peaks_local_max(image, radius=10, threshold=25, ax=None):
    '''
    http://stackoverflow.com/questions/9111711/get-coordinates-of-local-maxima-in-2d-array-
    above-certain-value
    '''
    import scipy.ndimage as ndimage
    import scipy.ndimage.filters as filters
    
    neighborhood_size = radius
    foldchange = 1.5
    
    data_max = filters.maximum_filter(image, neighborhood_size)
    maxima = (image == data_max)
    data_min = filters.minimum_filter(image, neighborhood_size)
    fc = ((data_max - data_min)/data_min > foldchange)
    maxima[fc == 0] = 0
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0
    
    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    
    y,x = np.where(labeled > 0 )
    peaks = map(tuple, np.array([x,y]).T)
    
    if ax is not None:     
        ax.imshow(image)    
        ax.autoscale(False)
        ax.plot(x,y, 'ro')
        
    return peaks

def getCircle(center, radius=5, img_width=320, img_height=240):
    '''
    Takes a point and returns all valid pixels in the image within a 
    given radius.
    '''
    x,y = center
    span = np.array([-radius, radius+1])
    points = [(i,j) for i in range(*(x + span))
                    for j in range(*(y + span))
                    if (i >= 0) and (j >= 0) 
                    if (i < img_width) and (j < img_height)
                    if (x-i)**2 + (y-j)**2 < radius**2]
    return np.array(points)

def getCenters(stack):
    '''
    Takes in a stack and finds the centers of all of the features.
    '''
    stackSum = 1.*np.sum(stack, axis=0)
    stackSum = filter.denoise_tv_chambolle(stackSum, weight=300, eps=1e-5)
    centers = detect_peaks_local_max(stackSum, radius=30)
    return centers

def getMovingTrace(stack, center, radius=5, r_big=10):
    x,y = max(center[0],r_big), max(center[1], r_big)
    movingTrace = []
    values = []
    for i,s in enumerate(stack):
        section_big = s[y-r_big:y+r_big+1, x-r_big: x+r_big+1]
        xsums = section_big.sum(axis=1)**2
        xpos = x + (xsums.dot(range(len(xsums))) / sum(xsums)) - r_big
        ysums = section_big.sum(axis=0)**2
        ypos = y + (ysums.dot(range(len(ysums))) / sum(ysums)) - r_big
        adj_center = (int(xpos), int(ypos))
        centers.append(adj_center)
        circle = getCircle(adj_center, radius)
        movingTrace.append(s[circle[:,1], circle[:,0]].sum())
    return movingTrace

getMaxTrace = lambda stack, circle: stack[:, circle[:,1], 
                                          circle[:,0]].max(axis=1)
getTrace = lambda stack, circle: stack[:, circle[:,1], 
                                       circle[:,0]].sum(axis=1)