'''
Created on Jul 24, 2012

@author: agross
'''
from Arser import Arser

class NormalizedSeries(object):
    def __init__(self, series):
        self.series = series
        self.values = array(series)
        self.times = array(series.index, dtype=float)
        self.period = None
        
    def findPeriod(self):
        arser = Arser(list(series.index), series)
        stats = arser.evaluateNew()
        self.period = stats['period']
        return self.period
        
    def normalizeDays(self):
        if self.period is None: self.findPeriod()
        series, self.trend = rollingMeanScale(self.series, self.period, 
                                              getTrend=True)
        r = amplitudeAdjust(series, self.period, getAmp=True)
        self.normedDays, self.phase, self.amp = r
        
        
def getNormedDays(series, period=False, getFeatures=False):
    arser = Arser(list(series.index), series)
    stats = arser.evaluateNew()
    period, phase = stats['period'], stats['phase']
    series,trend = rollingMeanScale(series, period, getTrend=getFeatures)
    normedDays, phase, amp = amplitudeAdjust(series, period, 
                                             getAmp=getFeatures)
    return dict(normedDays=normedDays, period=period, phase=phase,
                trend=trend, amp=amp)
    
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
        plotAxis.plot(fFit, rMeanFitter.predict(fFitT+.01), 
                      label='Rolling Mean')
        plotAxis.set_title('Detrend the Data')
        #plotAxis.legend(loc='best')
        
    if getTrend:
        '''
        The trend is the difference of the one day average of the 
        interpolated data from the beginning to the end of the measurement.
        '''
        trend = (rMean[-1] - rMean[0]) / (rMean[-1] + rMean[0])
        return series, trend
    else:
        return series