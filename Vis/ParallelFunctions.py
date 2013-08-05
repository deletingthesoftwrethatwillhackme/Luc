'''
Created on Feb 27, 2012

@author: agross
'''
from numpy import array, arange, round
import pandas as pandas
from sklearn.svm import SVR #@UnresolvedImport

def getParams(series, phase, period):
    '''Shift the times to give relative time of day'''
    t0 = array(series.index, dtype=float)
    t0 = (t0 - phase) % period
    t0 = array([t0]).T
    
    '''Shift the array to fit the edges'''
    tExt = array([array([t0-period,t0,t0+period]).flatten()]).T
    seriesExt = array([array(series),array(series),
                       array(series)]).flatten()
    return tExt, seriesExt

def getError(signal, normedDay, period, phase):
    tUp = array([arange(0,period+.1,.01)]).T
    upSample = pandas.Series(signal.predict(tUp), index=round(arange(0,period+.1,.01),2))

    t0 = round((array(normedDay.index,dtype=float) - phase)%period,2)
    nD = pandas.Series(normedDay, index=t0)

    xDiff = (nD - upSample)
    xDiff = xDiff[xDiff.notnull()]

    t0 = array([t0]).T
    yDiff = nD - signal.predict(t0)

    error = sum(pow(xDiff/period,2) + pow(yDiff/2,2))
    return error

def fitModel(series, tExt, seriesExt, period, phase, test, C=1e4, gamma=.02, epsilon=.01):
    '''Fit the model'''
    svr_rbf = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
    y_rbf = svr_rbf.fit(tExt, seriesExt)
    error  = getError(y_rbf, test, period, phase)
    return error, y_rbf



def fitGammas(train, test, period, phase, tExt, seriesExt, gammas):
    tExt, seriesExt = getParams(train, phase, period)
    
    return array([fitModel(train, tExt, seriesExt, period, phase, test, gamma=gamma) 
                       for gamma in gammas])
    
def printMe(aa):
    return aa
    