'''
Created on Feb 9, 2012

@author: agross
'''

from Expression.Arser import Arser
from sklearn import linear_model #@UnresolvedImport
from sklearn.svm import SVR  #@UnresolvedImport
from sklearn.preprocessing import normalize #@UnresolvedImport
from pylab import detrend_linear #@UnresolvedImport
from pylab import arange, subplots, array
import numpy as numpy
import pandas as pandas

def scaleMe(matrix):
    _min = numpy.min(matrix, axis=0)
    _max = numpy.max(matrix, axis=0)
    return (matrix-_min)/(_max-_min)

def cycle_adjust(series):
    slope = (series[-1] - series[0])/len(series)
    rescaled = series - [i*slope + series[0] for i in range(len(series))]
    normed = normalize([rescaled], norm='l1')[0]
    #return normed
    _min, _max = numpy.min(normed), numpy.max(normed)
    standardized = numpy.array([-1+2*(item - _min)/(_max-_min) for item in normed])
    return standardized

def rescaleMe(series, phase, period, numDays=4, n_neighbors=2, axs=[]):
    dayBreaks = [phase+period*day for day in range(numDays+1)]
    timePoints = []
    axs[0].plot(series.index, numpy.array(series), label='normal')
    axs[0].plot(series.index, detrend_linear(numpy.array(series)), label='detrend')
    #axs[0].plot(numpy.array(times_per_day).flatten(), 10000*valuesFlat, label='detrend each')
    
    for i in range(numDays):
        values = map(float, list(series.index[(series.index > dayBreaks[i]) & 
                                              (series.index < dayBreaks[i+1])]))
        timePoints.append(values)
    for i in range(numDays):
        axs[1].plot(numpy.array(timePoints[i]-phase) % period, 
                    cycle_adjust(series.ix[timePoints[i]]), 'o', label=str(i))
        axs[1].set_title('Raw Data Phase Shifted')
        axs[1].legend(loc='best')
        
def plotScaledPoints(data, phase, period, numDays=4, ax='None'):
    numDays = 4
    dayBreaks = [phase+period*day for day in range(numDays+1)]
    timePoints = []
    for i in range(numDays):
        values = map(float, list(data.index[(data.index > dayBreaks[i]) & (data.index < dayBreaks[i+1])]))
        timePoints.append(values)
    for i in range(numDays):
        ax.plot(numpy.array(timePoints[i]-phase) % period, 
                cycle_adjust(data.ix[timePoints[i]]), 'o', label=str(i))
        ax.set_title('Raw Data Phase Shifted')
        ax.legend(loc='best')
        
def plotScaledPointsKNN(valuesFlat, times_per_day, phase, period, timeIndex, numDays=4, ax='None'):
    from sklearn import neighbors #@UnresolvedImport
    knn = neighbors.KNeighborsRegressor(2, weights='distance') 
    knn.fit([[t] for t in numpy.array(times_per_day).flatten()], valuesFlat)
    
    dayBreaks = [phase+period*day for day in range(numDays+1)]
    timePoints = []
    for i in range(numDays):
        values = map(float, list(timeIndex[(timeIndex > dayBreaks[i]) & (timeIndex < dayBreaks[i+1])]))
        timePoints.append(values)
    for i in range(numDays):
        ax.plot(numpy.array(timePoints[i]-phase) % period, 
                knn.predict([[t] for t in timePoints[i]]), 'o', label=str(i))
        ax.set_title('KNN Normalized')
        ax.legend(loc='best')

def plotSeriesOld(series):
    times = arange(5,106,2.5)
    numDays=4
    colors = ['b','g','r','c']
    
    fig, axs = subplots(1,2,figsize=(12,4))
    arser = Arser(list(series.index), series)
    stats = arser.evaluate()
    period, phase = stats['period'][0], stats['phase'][0]
    dayBreaks = [phase+period*day for day in range(numDays+1)]
    timePoints = []
    for i in range(numDays):
        values = map(float, list(series.index[  (series.index > (dayBreaks[i]-.5)) 
                                              & (series.index < (dayBreaks[i+1]+.5))]))
        timePoints.append(values)
    
    axs[0].plot(times, array(series), '--',label='normal',color='black')
    axs[0].plot(times, array(series), 'o',label='normal',color='black', alpha=.3)
    axs[0].set_title('Raw Data')
    
    normedDays = [cycle_adjust(array(series.ix[timePoints[i]])) for i in range(numDays)]
    t0 = array([[t] for day in timePoints for t in day])
    t0 = (t0 - phase) % period
    t0 = [[t] for t in array([t0-period,t0,t0+period]).flatten()]
    normedSeries = [e for day in normedDays for e in day]
    normedSeries = array([normedSeries,normedSeries,normedSeries]).flatten()

    for i in range(numDays):
        axs[1].plot(array(timePoints[i]-phase) % period, normedDays[i], 'o', 
                    label=str(i),color=colors[i])
        axs[1].set_title('Raw Data Phase Shifted')
        axs[1].legend(loc='best')
    
    svr_rbf = SVR(kernel='rbf', C=1e4, gamma=.03, epsilon=.01)
    y_rbf = svr_rbf.fit(t0, list(normedSeries))
    t1 = [[t] for t in arange(0,period, period/100.)]
    axs[1].plot(t1, y_rbf.predict(t1));
    axs[1].set_xbound(0,period)
    
def plotLucData(series, ax=''):
    if ax=='': fig,ax=subplots(1,1)
    ax.plot(series.index, series, 'o')
    ax.plot(series.index, series, '--')
    ax.set_ylabel('Relative Luminescence')
    ax.set_title('Raw Data')
    
def rollingMeanScale(series, period, plotAxis=False):
    svr_rbf = SVR(kernel='rbf', C=1e4, gamma=.01, epsilon=.01)
    '''Fit Model to Data Series'''
    tS= numpy.array([series.index]).T
    y_rbf = svr_rbf.fit(tS, list(series))
    '''Up-sample to get rid of bias'''
    fFit = arange(series.index[0],series.index[-1]+.1,.25)
    trend = y_rbf.predict(numpy.array([fFit]).T)
    
    '''Take rolling mean over 1-day window'''
    shift = int(round(period/.5))
    rMean = pandas.rolling_mean(trend, shift*2)
    rMean = numpy.roll(rMean, -shift)
    rMean[:shift]=rMean[shift]
    rMean[-(shift+1):]=rMean[-(shift+1)]
    rMean = pandas.Series(rMean, index=fFit)
    
    '''Adjust Data Series by subtracting out trend'''
    series = series - array(rMean[array(series.index, dtype=float)])
    series = scaleMe(series)-.5
    
    if plotAxis:
        plotAxis.plot(fFit, trend, label='Series Trend')
        plotAxis.plot(fFit, rMean, label='Rolling Mean')
        plotAxis.set_title('Detrend the Data')
        plotAxis.legend(loc='lower left')

    return series

def amplitudeAdjust(series, phase, period, numDays=4, plotAxis=False):
    colors = ['b','g','r','c']
    dayBreaks = [phase+period*day for day in range(numDays+1)]
    timePoints = []
    for i in range(numDays):
        values = map(float, list(series.index[  (series.index > (dayBreaks[i]-.5)) 
                                              & (series.index < (dayBreaks[i+1]+.5))]))
        timePoints.append(values)
    normedDays = [cycle_adjust(array(series.ix[timePoints[i]])) 
                  for i in range(numDays)]
    if plotAxis:
        for i,day in enumerate(normedDays):
            plotAxis.plot(timePoints[i], day, 'o', color=colors[i])
            plotAxis.plot(timePoints[i], day, '--', color='black')
            plotAxis.set_title('Adjust Daily Amplitude')
    
    normedDays = [pandas.Series(day,index=array(timePoints[i])+.01*i) 
                  for i,day in enumerate(normedDays)]
    return normedDays

def plotDailyPoints(series, timePoints, ax):
    colors = ['b','g','r','c']
    ax.plot(series.index, series, 'o', color='black', alpha=.3)
    ax.plot(series.index, series, '--', color='black')
    for i,day in enumerate(timePoints):
        ax.plot(numpy.array(day), series.ix[day], 
                'o', label=str(i), color=colors[i], alpha=.5)
        
def getCharacteristicSignal(normedDays, phase, period, plotAxis=False):
    series = pandas.Series()
    for day in normedDays:
        series = series.append(day)
    '''Shift the times to give relative time of day'''
    t0 = array(series.index, dtype=float)
    t0 = (t0 - phase) % period
    t0 = array([t0]).T
    
    '''Shift the array to fit the edges'''
    tExt = array([array([t0-period,t0,t0+period]).flatten()]).T
    seriesExt = numpy.array([array(series),array(series),
                             array(series)]).flatten()
    
    '''Fit the model'''
    svr_rbf = SVR(kernel='rbf', C=1e4, gamma=.03, epsilon=.01)
    y_rbf = svr_rbf.fit(tExt, seriesExt)
    
    '''Predict a new characteristic signal'''
    t1 = array([arange(0,period, period/100.)]).T
    signal = y_rbf.predict(t1)
    
    if plotAxis:
        plotAxis.plot(t1, signal)
        colors = ['b','g','r','c']
        for i,day in enumerate(normedDays):
            timesAdjusted = array(normedDays[i].index,dtype=float)
            timesAdjusted = (timesAdjusted - phase) % period
            plotAxis.plot(timesAdjusted, day, 'o', label=str(i), 
                          color=colors[i])
        plotAxis.set_title('Characteristic Signal')
        plotAxis.legend(loc='best')
        plotAxis.set_xbound(0,period)
        plotAxis.set_ybound(-1.1,1.1)
    return signal

def plotSeries(series):
    fig, axs = subplots(2,2, figsize=(12,9))
    arser = Arser(list(series.index), series)
    stats = arser.evaluate()
    period, phase = stats['period'][0], stats['phase'][0]
    
    plotLucData(series, ax=axs[0,0])
    series = rollingMeanScale(series, period, plotAxis=axs[0,1])
    normedDays = amplitudeAdjust(series, phase, period, plotAxis=axs[1,0])
    signal = getCharacteristicSignal(normedDays, phase, period, plotAxis=axs[1,1])