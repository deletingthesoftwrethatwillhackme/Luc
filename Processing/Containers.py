'''
Created on Jul 20, 2012

@author: agross
'''
import sys
from numpy import array
from pandas import Series

import matplotlib.pyplot as plt
import numpy as np

from AutoLuc.Blob import getCenters, getCircle, getTrace, getMaxTrace
from Processing.RBFRegression import plotFit, crossValidation, getModel
from Processing.RBFRegression import getErrorAcrossDays
from Vis.CyclePlots2 import cyclePlot

from Normalization import findPeriod, rollingMeanScale, amplitudeAdjust
from Vis.Descriptive import annotated_scatter

QUADRANTS = [['top_left', 'top_right'], ['bottom_left', 'bottom_right']]
CIRCLE_SIZE = 69.
MAX_VALUE = 252

quadrant_pos = {'top_left': (0,0), 'top_right': (0,1), 
                'bottom_left': (1,0), 'bottom_right': (1,1), 
                'bottom-right': (1,1)}

stack_id = lambda plate: (plate.set,) + quadrant_pos[plate.quadrant]

class Plate(object):
    '''
    Plate class for storing and manipulating plate level data.  
    '''
    def __init__(self, stack, folder, plate, times):
        self.folder = folder
        self.set = plate[0]
        self.quadrant = QUADRANTS[plate[1]][plate[2]]
        self.times = times
        self.stack = stack
        self.status = 'Sucess'
        
        'Now find features and then their time series in the stack'
        self.centers = getCenters(stack)
        self.circles = map(getCircle, self.centers)
        self.maxTraces = [getMaxTrace(stack, circle) for circle in 
                          self.circles]
        #self.movingTraces = [getMovingTrace(stack, center) for center
        #                     in self.centers]
        self.traces = [getTrace(stack, circle) for circle in self.circles]     
    
    def showPlate(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1,1, figsize=(15,10))
        ax.imshow(self.stack.sum(axis=0), interpolation='nearest');
        for c in self.circles:
            ax.scatter(*zip(*c), color='black', alpha=.1)
        ax.scatter(*zip(*self.centers), color = 'orange');
        for k,c in enumerate(self.centers):
            ax.annotate(k, c, color='white', size=20, weight='extra bold')
        ax.set_title('Folder %s   Set %s   %s Plate'%(self.folder, self.set, 
                                                      self.quadrant), size=30)
    
    def plotTraces(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1,1, figsize=(8,6))
        for trace in self.traces: 
            ax.plot(self.times, trace)
        ax.set_xlabel('Time Past ZT')
        ax.set_ylabel('Signal Intensity')
        
    def runTraceFits(self, job_server=None):
        if job_server is None:
            self.fits = [getTraceFittedObject(trace, self.times) for trace
                         in self.traces]
        else:
            traces = []
            jobs = [job_server.submit(fitTrace, args=(trace, self.times), 
                                      modules=('sklearn.svm', 'RBFRegression')) 
                    for trace in self.traces]
            for job in jobs:
                results = job()
                trace = Trace(series=results['trace'])
                trace.recordFit(results)
                traces.append(trace)
            self.fits = traces
            
    def use_refits(self):
        '''
        The refits get rid of the first and/or last day if the data is bad.
        This method replaces the fit objects with their refitted counterparts
        if they exist.
        '''
        self.fits = [fit.refit if 'refit' in fit.__dict__
                        and fit.refit != False 
                        and sum(map(len, fit.refit.normedDays)) > 15
                        else fit for fit in self.fits]
    def set_features(self):
        from pandas import DataFrame
        '''
        Finds the features of each fit and creates a DataFrame for easy access.
        '''
        features = array([[f.period, f.phase, f.error, f.trend, np.mean(f.values), 
                           f.amp[-2]] for f in self.fits])
        features = DataFrame(features, columns=['period', 'phase', 'error', 
                                                'trend', 'baseline', 'amplitude'])
        if features['phase'].mean() < 12:  #make visualization look better
            features['phase'] = features.apply(adj_phase,1)
        self.features = features
        return features
    
    def feature_scatters(self):
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        
        fig, axs = plt.subplots(1,2, figsize=(9,4))
        features = self.features
        annotated_scatter(features['trend'], features['amplitude'], 
                          features.index, ax=axs[0],  c=features['error'], 
                          cmap=plt.cm.jet, s=200, alpha=.5)
        sc = annotated_scatter(features['period'], features['phase'], features.index, 
                          ax=axs[1], c=features['error'], cmap=plt.cm.jet, s=200, 
                          alpha=.5)
        
        divider = make_axes_locatable(axs[1])
        cax = divider.append_axes("right", "5%", pad="3%")
        cax.set_xlabel('error')
        fig.colorbar(sc, cax=cax)
        fig.tight_layout()
        
def adj_phase(s):
   if s['phase'] > s['period']/2:
       return s['phase'] - s['period'] 
   else:
       return s['phase']
                    

        
class Trace(object):
    '''
    Object to store information for a trace.
    I'm trying to keep all of the internal data as standard data-types to 
    make it easier to pickle.
    '''
    def __init__(self, values=None, times=None, series=None):
        self.period = None
        self.normedDays = None
        
        if series is not None:
            self.values = array(series)
            self.times = array(series.index, dtype=float)
        else:
            self.values = values
            self.times = times
            
    def findPeriod(self):
        self.period = findPeriod(self.times, self.values)
        return self.period
        
    def normalizeDays(self, force=False):
        if (self.normedDays is not None) or force: 
            return self.normedDays
        if self.period is None: self.findPeriod()
        series = Series(self.values, index=self.times)
        scaled, self.trend = rollingMeanScale(series, self.period, 
                                              getTrend=True)
        r = amplitudeAdjust(scaled, self.period, getAmp=True)
        self.normedDays, self.phase, self.amp = r
        return self.normedDays
            
    def fit(self):
        
        if self.normedDays is None:
            self.normalizeDays()
        trace = appendUnique(self.normedDays)
        fits = crossValidation(trace, self.period, self.phase)
        self.error, self.model = getModel(fits, self.normedDays, self.period, 
                                          self.phase)
        return self.model
    
    def adjustStartAndEnd(self):
        self.normedDays = [day for day in self.normedDays if len(day)>0]
        dailyErrors = getErrorAcrossDays(self.normedDays, self.period, 
                                         self.phase, self.model.gamma)
        start, end = 0,len(self.values)
        if (dailyErrors[0] / np.mean(dailyErrors)) > 1.5:
            start = len(self.normedDays[0])
        if (dailyErrors[-1] / np.mean(dailyErrors)) > 1.5:
            end = len(self.values) - len(self.normedDays[-1])
        if ((start, end) != (0,len(self.values))):
            self.refit = Trace(self.values[start:end], self.times[start:end])
            try:
                self.refit.fit()
            except Exception:  #usually because not enough points left
                self.refit = False
        else:
            self.refit = False
        self.dailyErrors = dailyErrors
        return self.dailyErrors, self.refit
    
    def plotFit(self, axs=None):
        if axs is None:
            fig,axs = plt.subplots(1,2,figsize=(12,4))
        cyclePlot(Series(self.values, index=self.times), ax=axs[0])
        plotFit(self.model, self.period, self.phase, self.normedDays, 
                plotAxis=axs[1])
        
def compare_fits(fit1, fit2):
    fig,axs = plt.subplots(1,2, figsize=(12,4))
    fit1.plotFit(axs)
    for l in axs[0].get_lines() + axs[1].get_lines():
        if l.get_marker() != 'None':
            l.set_marker('D')
    fit2.plotFit(axs)
    axs[1].legend().set_visible(False)    
        
def appendUnique(normedDays):
    data = Series([])
    for day in normedDays:
        for t,d in day.iteritems():
            if t not in data:
                data = data.set_value(t,d)
            else:
                data[t] = (data[t]+day[t]) / 2
    return data
    
def fitTrace(values, times):
    from Processing.RBFRegression import crossValidation, getModel 
    from pandas import Series
    trace = Series(values, index=times)
    results = crossValidation(trace, method='')
    error, model = getModel(fits, normedDays, period, phase)
    return dict(trace=trace, period=period, phase=phase, error=error, 
                model=model, normedDays=normedDays)

def getTraceFittedObject(values, times):
    trace = Trace(values, times)
    try:
        trace.fit()
        trace.adjustStartAndEnd()
    except Exception:
        do_nothing = True
        #print "Unexpected error:", sys.exc_info()[0] 
    return trace

def get_saturated_pixels(stack, circle): 
    return (stack[:, circle[:,1], circle[:,0]] == MAX_VALUE).sum(axis=1)

scale = lambda s: (s-min(s)) / float((max(s)-min(s))+.01)

def adjust_for_saturation(plate, plot=False):
    saturated_pixels = [get_saturated_pixels(plate.stack, circle) for circle in 
                    plate.circles]
    combine = lambda i: scale(plate.traces[i]) + scale(saturated_pixels[i]/CIRCLE_SIZE)
    
    plate_adj = Plate(plate.stack, plate.folder, stack_id(plate), plate.times)
    plate_adj.traces = map(combine, range(len(plate.traces)))
    #plate_adj.runTraceFits()
    if plot:
        fig, axs = subplots(2,1, figsize=(8,5))
        plate.plotTraces(ax=axs[0])
        axs[1].plot(plate.times, array(saturated_pixels).T);
        axs[1].set_xlabel('Time Past ZT')
        axs[1].set_ylabel('# Saturated Pixels')
        fig.tight_layout()
    return plate_adj
