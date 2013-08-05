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
    #_max = normed[0]
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
    
    svr_rbf = SVR(kernel='rbf', C=1e4, gamma=.03, epsilon=.01, scale_C=True)
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
    
def rollingMeanScale(series, period, gamma=.01, plotAxis=False):
    from sklearn import neighbors #@UnresolvedImport
    svr_rbf = SVR(kernel='rbf', C=1e5, gamma=.02, 
              epsilon=numpy.abs(series).median()/20, scale_C=True)
    tS= numpy.array([series.index]).T
    y_rbf = svr_rbf.fit(tS, list(series))
    fFit = arange(series.index[0],series.index[-1]+.1,.25)
    fFitT = numpy.array([fFit]).T
    #fFit = sorted(array(list(set(fFit).union(set(series.index)))))
    trend = y_rbf.predict(fFitT)
    
    '''Take rolling mean over 1-day window'''
    rMeanFitter = neighbors.KNeighborsRegressor(2, weights='uniform')
    shift = int(round(period/.5))
    rMean = pandas.rolling_mean(trend, shift*2)
    rMean = numpy.roll(rMean, -shift)
    rMean[:shift]=rMean[shift]
    rMean[-(shift+1):]=rMean[-(shift+1)]
    rMeanFitter = rMeanFitter.fit(fFitT, rMean)
    
    '''Remove bad points from series'''
    goodPoints = numpy.where(abs((series - y_rbf.predict(tS)) 
                                 / abs(series)) < 1.5)[0]
    badPoints = series.ix[list(set(range(len(series))).difference(goodPoints))]

    '''Adjust Data Series by subtracting out trend'''
    series = series - rMeanFitter.predict(tS)
    series = series.ix[series.index[goodPoints]]
    series = scaleMe(series)-.5
    
    if plotAxis:
        if len(badPoints) > 0:  
            plotAxis.scatter(badPoints.index, badPoints, marker='x', color='red')
        plotAxis.plot(fFit, trend, label='Series Trend')
        plotAxis.plot(fFit, rMeanFitter.predict(fFitT), label='Rolling Mean')
        plotAxis.set_title('Detrend the Data')
        #plotAxis.legend(loc='best')
        
    return series

def amplitudeAdjust(series, period, plotAxis=False):
    adj = pandas.Series(list(series), array(series.index,dtype=float) % period)
    t0 = array(adj.index, dtype=float)
    tExt = array([array([t0-period,t0,t0+period]).flatten()]).T
    seriesExt = numpy.array([array(adj),array(adj),array(adj)]).flatten()
    '''Fit the model'''
    svr_rbf = SVR(kernel='rbf', C=1e4, gamma=.02, epsilon=.01, scale_C=True)
    y_rbf = svr_rbf.fit(tExt, seriesExt)
    phase = numpy.argmax(y_rbf.predict(array([arange(0,period,.1)]).T))/10.
    
    colors = ['b','g','r','c','y','m','b']
    numPeriods = int(numpy.ceil((max(series.index) - min(series.index)) / period))
    startTime = min(series.index) - (min(series.index) % period)
    if numPeriods > 4:
        days = range(1,numPeriods)
        dayBreaks = [startTime + phase+period*day for day in days]
    else:
        days = range(0, numPeriods+1)
        dayBreaks = [startTime + phase+period*day for day in days]
    timePoints = []
    for i in range(len(dayBreaks)-1):
        values = map(float, list(series.index[  (series.index > (dayBreaks[i]-.5)) 
                                              & (series.index < (dayBreaks[i+1]+.5))]))
        if (len(values)>7 and (dayBreaks[i] > min(series.index)) and 
            (dayBreaks[i+1] < max(series.index))):
            timePoints.append(values)
    normedDays = [cycle_adjust(array(series.ix[points])) 
                  for points in timePoints]
    if plotAxis:
        for i,day in enumerate(normedDays):
            plotAxis.plot(timePoints[i], day, 'o', color=colors[i])
            plotAxis.plot(timePoints[i], day, '--', color='black')
            plotAxis.set_title('Adjust Daily Amplitude')
    
    normedDays = [pandas.Series(day,index=array(timePoints[i])+.01*i, 
                                dtype=numpy.float) 
                  for i,day in enumerate(normedDays)]
    return normedDays, phase

def plotDailyPoints(series, timePoints, ax):
    colors = ['b','g','r','c','y','m']
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
    svr_rbf = SVR(kernel='rbf', C=1e4, gamma=.02, epsilon=.01, scale_C=True)
    y_rbf = svr_rbf.fit(tExt, seriesExt)
    
    if plotAxis:
        '''Predict a new characteristic signal'''
        t1 = array([arange(0,period, period/100.)]).T
        signal = y_rbf.predict(t1)
        
        plotAxis.plot(t1, signal)
        colors = ['b','g','r','c','y','m','b']
        for i,day in enumerate(normedDays):
            timesAdjusted = array(normedDays[i].index,dtype=float)
            timesAdjusted = (timesAdjusted - phase) % period
            plotAxis.plot(timesAdjusted, day, 'o', label=str(i), 
                          color=colors[i])
        plotAxis.set_title('Characteristic Signal')
        #plotAxis.legend(loc='lower right', title='Days', frameon=False)
        plotAxis.set_xbound(0,period)
        plotAxis.set_ybound(-1.1,1.1)
    return y_rbf
    #return pandas.Series(signal,index=t1.flatten())

def getTimePoints(series):
    numNum = int(series.name[1].split('#')[1])
    start  = 3 + ((numNum-1)/2.)
    timePoints = numpy.arange(start, start+102.1, 2.5)
    return timePoints

def plotSeries(series, method='Arser', period=False, figsize=(12,9)):
    fig, axs = subplots(2,2, figsize=figsize)
    if series.index.dtype ==  numpy.dtype('int64'): 
        series = pandas.Series(array(series), index=getTimePoints(series))
    #series = rollingMeanScale(series, 24, plotAxis=axs[0,1])
    arser = Arser(list(series.index), series)
    if period:
        stats = arser.evaluate(T_start =period, T_end=period)
        period, phase = stats['period'][0], stats['phase'][0]
    else:
        if method=='Arser':
            stats = arser.evaluate()
            period, phase = stats['period'][0], stats['phase'][0]
            if int(period)==24:
                stats = arser.evaluateNew()
                period, phase = stats['period'], stats['phase']
        else:
            stats = arser.evaluateNew()
            period, phase = stats['period'], stats['phase']
    
    plotLucData(series, ax=axs[0,0])
    series = rollingMeanScale(series, period, plotAxis=axs[0,1])
    normedDays,phase = amplitudeAdjust(series, period, plotAxis=axs[1,0])
    axs[1,0].set_ylabel('Normalized Luminescence')
    signal = getCharacteristicSignal(normedDays, phase, 
                                     period, plotAxis=axs[1,1])
    return signal, normedDays, period, phase

def plotSeries2(series, method='Arser', period=False, figsize=(12,9)):
    fig, axs = subplots(1,4, figsize=figsize)
    if series.index.dtype ==  numpy.dtype('int64'): 
        series = pandas.Series(array(series), index=getTimePoints(series))
    #series = rollingMeanScale(series, 24, plotAxis=axs[0,1])
    arser = Arser(list(series.index), series)
    if period:
        stats = arser.evaluate(T_start =period, T_end=period)
        period, phase = stats['period'][0], stats['phase'][0]
    else:
        if method=='Arser':
            stats = arser.evaluate()
            period, phase = stats['period'][0], stats['phase'][0]
            if int(period)==24:
                stats = arser.evaluateNew()
                period, phase = stats['period'], stats['phase']
        else:
            stats = arser.evaluateNew()
            period, phase = stats['period'], stats['phase']
    
    plotLucData(series, ax=axs[0])
    series = rollingMeanScale(series, period, plotAxis=axs[1])
    normedDays = amplitudeAdjust(series, phase, period, plotAxis=axs[2])
    #axs[1,0].set_ylabel('Normalized Luminescence')
    signal = getCharacteristicSignal(normedDays, phase, 
                                     period, plotAxis=axs[3])
    for ax in axs[:3]: ax.set_xlabel('Hours in LL')
    axs[3].set_xlabel('Hours Past Peak Expression')
    for ax in axs[1:]: ax.set_yticks([])
    axs[0].set_xbound(5,105)
    axs[1].set_xbound(5,105)
    axs[2].set_xbound(5,105)
    fig.subplots_adjust(wspace=.05, top=.9, bottom=.2)
    return signal, normedDays, period, phase


def plotSeriesFake(series, period=False, method='Arser', cheap=False):
    if series.index.dtype ==  numpy.dtype('int64'): 
        series = pandas.Series(array(series), index=getTimePoints(series))
    #series = rollingMeanScale(series, 24, plotAxis=axs[0,1])
    arser = Arser(list(series.index), series)
    if period:
        stats = arser.evaluate(T_start =period, T_end=period)
        period, phase = stats['period'][0], stats['phase'][0]
    else:
        if method=='Arser':
            stats = arser.evaluate()
            period, phase = stats['period'][0], stats['phase'][0]
            if int(period)==24:
                stats = arser.evaluateNew()
                period, phase = stats['period'], stats['phase']
        else:
            stats = arser.evaluateNew()
            period, phase = stats['period'], stats['phase']

    if not cheap:
        series = rollingMeanScale(series, period)
    else:
        series = rollingMeanScale(series, period, gamma=.001)
    normedDays = amplitudeAdjust(series, phase, period)
    signal = getCharacteristicSignal(normedDays, phase, period)
    return signal, normedDays, period, phase

def getNormedDays(series, period=False, method='Arser', cheap=False):
    if series.index.dtype ==  numpy.dtype('int64'): 
        series = pandas.Series(array(series), index=getTimePoints(series))
    #series = rollingMeanScale(series, 24, plotAxis=axs[0,1])
    arser = Arser(list(series.index), series)
    if period:
        stats = arser.evaluate(T_start =period, T_end=period)
        period, phase = stats['period'][0], stats['phase'][0]
    else:
        if method=='Arser':
            stats = arser.evaluate()
            period, phase = stats['period'][0], stats['phase'][0]
            if int(period)==24:
                stats = arser.evaluateNew()
                period, phase = stats['period'], stats['phase']
        else:
            stats = arser.evaluateNew()
            period, phase = stats['period'], stats['phase']

    if not cheap:
        series = rollingMeanScale(series, period)
    else:
        series = rollingMeanScale(series, period, gamma=.001)
    normedDays, phase = amplitudeAdjust(series, period, plotAxis=False)
    return normedDays, period, phase