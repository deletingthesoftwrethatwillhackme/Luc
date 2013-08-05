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

def scaleMe(matrix):
    _min = numpy.min(matrix, axis=0)
    _max = numpy.max(matrix, axis=0)
    return (matrix-_min)/(_max-_min)

def cycle_adjust1(series):
    slope = (series[-1] - series[0])/len(series)
    rescaled = series - [i*slope + series[0] for i in range(len(series))]
    normed = normalize([rescaled], norm='l1')[0]
    #return normed
    _min, _max = numpy.min(normed), numpy.max(normed)
    #_max = normed[0]
    standardized = numpy.array([-1+2*(item - _min)/(_max-_min) for item in normed])
    return standardized

def cycle_adjust(series):
    slope = (series[-1] - series[0])/len(series)
    rescaled = series - [i*slope + series[0] for i in range(len(series))]
    normed = normalize([rescaled], norm='l1')[0]
    #return normed
    _min, _max = mean(sorted(normed)[:3]), mean(sorted(normed)[-3:])
    #_max = normed[0]
    standardized = numpy.array([-1+2*(item - _min)/(_max-_min) for item in normed])
    return standardized

#from pandas import Series
#from Processing.Normalization import cycle_adjust
def adjustDays(normedDays, period):
    for i,day in enumerate(normedDaysAdj):
        normedDaysAdj[i] = Series(day, index=array(day.index, dtype=float) - 
                                  adj[i])
    n = reduce(Series.append, normedDays)
    idx = array(n.index.copy(),dtype=float) + arange(len(n))*10e-5
    n = Series(array(n), idx % period)
    idx_rel = array(n.index.copy(), dtype=float)
    n = n.sort_index()
    n = n.apply(cycle_adjust)
    n = Series(array(n[idx_rel]), index = idx)
    adj  = split(arange(len(idx))*10e-5, cumsum(map(len, normedDays)))
    normedDaysAdj = [n[array(d.index, dtype=float) + adj[i]] for (i,d) in 
                     enumerate(normedDays)]
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
        trend = (rMean[-1] - rMean[0])
        return series, trend
    else:
        return series

def amplitudeAdjust(series, period, plotAxis=False):
    adj = Series(list(series), array(series.index,dtype=float) % period).sort_index()
    t0 = array(adj.index, dtype=float)
    tExt = array([array([t0-period,t0,t0+period]).flatten()]).T
    seriesExt = numpy.array([array(adj),array(adj),array(adj)]).flatten()
    svr_rbf = SVR(kernel='rbf', C=1e4, gamma=.1, epsilon=.01)
    y_rbf = svr_rbf.fit(tExt, seriesExt)
    phase = numpy.argmax(y_rbf.predict(array([arange(0,period,.1)]).T))/10.
    
    numPeriods = int(numpy.ceil((max(series.index) - min(series.index)) / period))
    startTime = min(series.index) - (min(series.index) % period)
    days = range(0, numPeriods)
    dayBreaks = [startTime + phase+period*day for day in days]
    
    normedDays = []
    
    points = array(series.index[series.index < (series.index[0]+period)], dtype=float)
    firstDay = Series(cycle_adjust(array(series.ix[points])), index=points)
    firstDay = firstDay[firstDay.index < dayBreaks[0]]
    normedDays.append(firstDay)
    
    for i in range(0,len(dayBreaks)-1):
        values = array(series.index[(series.index > (dayBreaks[i]-.5)) & 
                                    (series.index < (dayBreaks[i+1]+.5))], 
                       dtype=float)
                                             
        if (len(values)>7 and (dayBreaks[i] > min(series.index)) and 
            (dayBreaks[i+1] < max(series.index))):
            normedDay = Series(cycle_adjust(array(series.ix[values])), index=values)
            normedDay = normedDay[(normedDay.index >= dayBreaks[i])  &
                                  (normedDay.index < dayBreaks[i+1])]
            normedDays.append(normedDay)
    
    points = array(series.index[series.index > (series.index[-1]-period)], dtype=float)
    lastDay = Series(cycle_adjust(array(series.ix[points])), index=points)
    #lastDay = lastDay[lastDay.index > dayBreaks[-1]]
    normedDays.append(lastDay)
    
    if plotAxis:
        colors = ['b','g','r','c','y','m']
        for i,day in enumerate(normedDays):
            plotAxis.plot(day.index, day, 'o', color=colors[i%6])
            plotAxis.plot(day.index, day, '--', color='black')
            plotAxis.set_title('Adjust Daily Amplitude')

    return normedDays, phase

def getNormedDays(series, period=False, method='Arser', cheap=False):
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

    series = rollingMeanScale(series, period)
    normedDays, phase = amplitudeAdjust(series, period, plotAxis=False)
    return normedDays, period, phase