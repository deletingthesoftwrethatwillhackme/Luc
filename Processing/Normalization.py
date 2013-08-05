'''
Created on Mar 14, 2012

@author: agross
'''
import numpy as numpy
from numpy import array, arange, dtype, mean, cumsum, split
from pandas import rolling_mean, Series #@UnresolvedImport
from Arser import Arser
from sklearn import neighbors #@UnresolvedImport
from sklearn.svm import SVR  #@UnresolvedImport
from sklearn.preprocessing import normalize #@UnresolvedImport
#import pandas as pandas

        
def findPeriod(times, values):
    arser = Arser(times, values)
    stats = arser.evaluateNew()
    return stats['period']
    
def scaleMe(matrix):
    _min = numpy.min(matrix, axis=0)
    _max = numpy.max(matrix, axis=0)
    return (matrix-_min)/(_max-_min)

def cycle_adjust(series, getAmp=False):
    slope = (series[-1] - series[0])/len(series)
    rescaled = series - [i*slope + series[0] for i in range(len(series))]
    normed = normalize([rescaled], norm='l1')[0]
    #return normed
    _min, _max = mean(sorted(normed)[:3]), mean(sorted(normed)[-3:])
    #_max = normed[0]
    standardized = numpy.array([-1+2*(item - _min)/(_max-_min) for item in normed])
    if getAmp:
        amp = _max - _min
        return standardized, amp
    else:
        return standardized

def adjustDays(normedDays, period):
    '''Make indicies unique'''
    adj  = split(arange(sum(map(len, normedDays)))*10e-5, 
                 cumsum(map(len, normedDays)))
    normedDaysU = [Series(day, index=array(day.index, dtype=float) + adj[i]) 
                   for (i,day) in enumerate(normedDays)]
    for i,day in enumerate(normedDays):
        normedDaysU[i] = Series(day, index=array(day.index, dtype=float) + 
                                adj[i])
    n = reduce(Series.append, normedDaysU)
    idx = array(n.index.copy(),dtype=float)
    n = Series(array(n), idx % period)
    idx_rel = n.index.copy()
    n = n.sort_index()
    n = n.apply(cycle_adjust)
    n = Series(array(n.ix[idx_rel]), index=idx)
    normedDaysAdj = [n.ix[d.index] for d in normedDaysU]
    for i,day in enumerate(normedDaysAdj):
        normedDaysAdj[i] = Series(day, index=array(day.index, dtype=float) - 
                                  adj[i])
    return normedDaysAdj

def getTimePoints(series):
    '''
    A lot of my data does not have times associated but rather just a list
    of integers.  This is just a place-holder that fills in a standard set 
    of times but needs to be updated to account for plate number and day-
    light savings time.
    '''
    numNum = int(series.name[1].split('#')[1])
    start  = 3 + ((numNum-1)/2.)
    timePoints = arange(start, start+102.1, 2.5)
    return timePoints

def rollingMeanScale(series, period, getTrend=False, plotAxis=False):
    knnModel = neighbors.KNeighborsRegressor(2, weights='distance')
    if numpy.rank(series.index[0]) > 0:
        tS= array([series.index.levels[0]]).T
    else:
        tS= array([series.index]).T
    y_knn = knnModel.fit(tS, list(series))
    fFit = arange(tS[0][0]+.001,tS[-1][0]+.1,.25)
    fFitT = array([fFit]).T
    trend = y_knn.predict(fFitT)
    
    '''Take rolling mean over 1-day window'''
    rMeanFitter = neighbors.KNeighborsRegressor(2, weights='distance')
    shift = int(round(period/.5))
    rMean = rolling_mean(trend, shift*2)
    rMean = numpy.roll(rMean, -shift)
    rMean[:shift]=rMean[shift]
    rMean[-(shift+1):]=rMean[-(shift+1)]
    rMeanFitter = rMeanFitter.fit(fFitT, rMean)
  
    '''Adjust Data Series by subtracting out trend'''
    series = series - rMeanFitter.predict(tS)
    series = scaleMe(series)-.5
    
    if plotAxis:
        plotAxis.plot(fFit, trend, label='Series Trend')
        plotAxis.plot(fFit, rMeanFitter.predict(fFitT+.01), label='Rolling Mean')
        plotAxis.set_title('Detrend the Data')
        #plotAxis.legend(loc='best')
        
    if getTrend:
        '''
        The trend is the difference of the one day average of the interpolated
        data from the beginning to the end of the measurement.
        '''
        trend = (rMean[-1] - rMean[0]) / (rMean[-1] + rMean[0])
        return series, trend
    else:
        return series

inBetween = lambda start, end: lambda t: (t >= start) & (t <= end)
floatArray = lambda arr: array(arr, dtype=float) 

def amplitudeAdjust(series, period, getAmp=False, plotAxis=False):
    '''
    Adjust the series to sqeeze each period to a uniform amplitude.
    '''
    adj = Series(list(series), floatArray(series.index) % period).sort_index()
    
    #create a long series for prediction
    t0 = floatArray(adj.index)
    tExt = array([list(t0 - period) + list(t0) + list(t0 + period)]).T
    seriesExt = array(list(adj)*3)
    svr_rbf = SVR(kernel='rbf', C=1e2, gamma=.05, epsilon=.01)
    y_rbf = svr_rbf.fit(tExt, seriesExt)
    phase = numpy.argmax(y_rbf.predict(array([arange(0,period,.1)]).T))/10.
    
    #find some information about the length of the series and where the day breaks lie
    numPeriods = int(numpy.ceil((max(series.index) - min(series.index)) / period))
    startTime = min(series.index) - (min(series.index) % period)
    days = range(0, numPeriods)
    dayBreaks = [startTime + phase+period*day for day in days]
    
    normedDays = []
    amplitudes = []
    
    #build up the list of normalized days
    points = floatArray(series.select(inBetween(0, series.index[0]+period)).index)
    norm, amp = cycle_adjust(array(series.ix[points]), getAmp=True)
    firstDay = Series(norm, index=points)
    firstDay = firstDay[firstDay.index < dayBreaks[0]]
    normedDays.append(firstDay)
    amplitudes.append(amp)
    
    for i in range(0,len(dayBreaks)-1):
        values = floatArray(series.select(inBetween(dayBreaks[i]-.5, 
                                                    dayBreaks[i+1]+.5)).index)                                             
        if (len(values) > 7 and (dayBreaks[i] > min(series.index)) and 
            (dayBreaks[i+1] < max(series.index))):
            norm, amp = cycle_adjust(array(series.ix[values]), getAmp=True)
            normedDay = Series(norm, index=values)
            normedDay = normedDay.select(inBetween(*dayBreaks[i:i+2]))
            normedDays.append(normedDay)
            amplitudes.append(amp)
    
    points = floatArray(series.select(lambda t: t > (series.index[-1]-period)).index)
    norm, amp = cycle_adjust(array(series.ix[points]), getAmp=True)
    amplitudes.append(amp)
    lastDay = Series(norm, index=points)
    normedDays.append(lastDay)
    
    if plotAxis:
        colors = ['b','g','r','c','y','m']
        for i,day in enumerate(normedDays):
            plotAxis.plot(day.index, day, 'o', color=colors[i%6])
            plotAxis.plot(day.index, day, '--', color='black')
            plotAxis.set_title('Adjust Daily Amplitude')
    if getAmp:
        return normedDays, phase, amplitudes
    else:
        return normedDays, phase
'''
def getNormedDays(series, period=False, method='Arser', getFeatures=False):
    if series.index.dtype ==  dtype('int64'): 
        series = Series(array(series), index=getTimePoints(series))
    arser = Arser(list(series.index), series)
    if period is not False:
        doNothing = False
        #stats = arser.evaluate(T_start =period, T_end=period)
        #period, phase = stats['period'][0], stats['phase'][0]
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
    if not getFeatures:
        series = rollingMeanScale(series, period)
        normedDays, phase = amplitudeAdjust(series, period, plotAxis=False)
        return dict(normedDays=normedDays, period=period, phase=phase)
    else:
        series,trend = rollingMeanScale(series, period, getTrend=True)
        normedDays, phase, amp = amplitudeAdjust(series, period, getAmp=True)
        return dict(normedDays=normedDays, period=period, phase=phase,
                    trend=trend, amp=amp)
'''
        