'''
Created on Mar 14, 2012

@author: agross
'''
from numpy import array, arange, round, rank, sqrt, mean
from pandas import Series, DataFrame, Panel
from sklearn.svm import SVR #@UnresolvedImport
from sklearn.cross_validation import LeaveOneOut #@UnresolvedImport
import matplotlib.pylab as plt
import numpy as np

import Normalization as Normalization

def appendUnique(normedDays):
    data = Series([])
    for day in normedDays:
        for t,d in day.iteritems():
            if t not in data:
                data = data.set_value(t,d)
            else:
                data[t] = (data[t]+day[t]) / 2
    return data
                
def getParams(series, phase, period):
    '''
    Takes a series and shifts the times to give relative time of day
    verses ZT time.  Then adds additional points on the day's perimeter
    to allow the regression to fit the edges.
    
    Returns a list of times in the form [[t1],[t2],...] and a series of 
    measurements in the form suited for sklearn's model.fit methods.
    '''
    from numpy import array
    
    t0 = array(series.index, dtype=float)
    t0 = (t0 - phase) % period
    t0 = array([t0]).T
    
    tExt = array([array([t0-period,t0,t0+period]).flatten()]).T
    seriesExt = array(list(series)*3)
    return tExt, seriesExt

def fitModel(tExt, seriesExt, gammas=[.02], C=1e2, epsilon=.05):
    '''
    Fits a series of models using RBF Support Vector Regression.
    Imports the sklearn package locally to make parallelization 
    a little easier. 
    
    Returns an array of sklean fitted models.
    '''
    fits = {}
    for gamma in gammas:
        svr_rbf = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
        fits[gamma] = dict(model=svr_rbf.fit(tExt, seriesExt))
    return fits

def getError(signal, normedDay, period, phase):
    '''
    Gets the error for a list of points across a normed day given a sklearn 
    model, the period, and the phase of the fitted signal.
    
    Returns the Euclidean error.
    '''
   
    if rank(normedDay.index[0]) > 0:
        t0= round((array(normedDay.index.get_level_values(0))- phase)%period,3)
    else:
        t0 = round((array(normedDay.index,dtype=float) - phase)%period,3)
    nD = Series(array(normedDay), index=t0)
    tUp = array([arange(-5,period+5.,.1)]).T
    sampled = Series(signal.predict(tUp), index=tUp.flatten())

    diff = [(((sampled - val)/2.)**2 + 
            (((array(sampled.index, dtype=float) - t)/period))**2).min()
            for t,val in nD.iteritems()]

    error = mean(sqrt(diff))
    return error

def getErrorAcrossDays(normedDays, period, phase, gamma):
    days = array(normedDays)    
    dailyErrors = []
    for (train, test) in LeaveOneOut(len(days)):
        training = appendUnique(days[train])
        testing = days[test][0]
        tExt, seriesExt = getParams(training, phase, period)
        fit = fitModel(tExt, seriesExt, [gamma])[gamma]['model']
        dailyErrors.append(getError(fit, testing, period, phase))
    return dailyErrors

def getError1(signal, normedDay, period, phase):
    '''
    Gets the error for a list of points across a normed day given a sklean 
    model, the period, and the phase of the fitted signal.
    
    Here I'm using the Euclidean distance as the error measurement.  This 
    requires a little more computation due to the need to fit an inverse
    model, but provides better fits.
    
    Returns the squared Euclidean error.
    '''
    
    if rank(normedDay.index[0]) > 0:
        t0= round((array(normedDay.index.get_level_values(0))- phase)%period,3)
    else:
        t0 = round((array(normedDay.index,dtype=float) - phase)%period,3)
    nD = Series(normedDay, index=t0)
    
    tUp = array([arange(0,period+.1,.1)]).T
    invSignal = SVR(kernel='rbf', C=signal.C, gamma=signal.gamma, 
                    epsilon=signal.epsilon)
    
    invSignal.fit(array([signal.predict(tUp)]).T, tUp.flatten())
    
    xDiff = nD - signal.predict(array([array(nD)]).T)
    yDiff = nD - signal.predict(array([nD.index]).T)
    
    error = sum(pow(xDiff/period,2) + pow(yDiff/2,2))
    return error

def getModel(fits, normedDays, period, phase, plotAxis=None):
    '''
    Takes a Panel of fits and their errors, finds the best fit, and
    creates a new (this time witout missing data) model using that fit.
    
    Returns the new model and its error.  
    '''
    meanErrors = fits.major_xs('error').mean(axis=1)
    gamma = meanErrors.index[meanErrors.argmin()]
    #data = reduce(Series.append, normedDays)
    data = appendUnique(normedDays)
    tExt, seriesExt = getParams(data, phase, period)
    model = fitModel(tExt, seriesExt, gammas=[gamma])
    model = model[gamma]['model']
    error = getError(model, data, period, phase)
    if plotAxis is not None: 
        plotAxis.plot(meanErrors)
        plotAxis.set_title(str(gamma)+', '+str(error))
    return error, model

def get_trasform(matrix):
    _min = np.min(matrix, axis=0)
    _max = np.max(matrix, axis=0)
    transform = lambda l: 2*(l -_min)/(_max-_min) - 1
    return transform

def plotFit(model, period, phase, normedDays, plotAxis=None, label=''):
    if plotAxis is None:
        fig, plotAxis = plt.subplots(1,1)
    '''Predict a new characteristic signal'''
    t1 = array([arange(0,period, period/100.)]).T
    
    signal = model.predict(t1)
    transform = get_trasform(signal)
    signal = transform(signal)
    
    
    plotAxis.plot(t1, signal, lw=6, color='black', alpha=.6)
    colors = plt.rcParams['axes.color_cycle']
    for i,day in enumerate(normedDays):
        timesAdjusted = array(normedDays[i].index,dtype=float)
        timesAdjusted = (timesAdjusted - phase) % period
        plotAxis.plot(timesAdjusted, transform(day), 'o', label=str(i), 
                      color=colors[i % len(colors)], ms=10, alpha=.8)
    plotAxis.set_title(label, size=17)
    plotAxis.legend(loc='best', title='Day')
    plotAxis.set_xbound(0,period)
    plotAxis.set_xlabel('Time Past Peak Expression', size=14)
    plotAxis.tick_params(length=6,width=2, labelsize=12)
    
def crossValidationOld(series, job_server=None, period=False, method='Arser', gammas=[]):
    if len(gammas) == 0:
        gammas = list(2**arange(-9.,-3))
    
    normedDays, period, phase = Normalization.getNormedDays(series, method=method, period=period)
    normedDays = Normalization.adjustDays(normedDays, period)
    #data = reduce(Series.append, normedDays)
    '''Gets Unique indicies for all days'''
    data = appendUnique(normedDays)
    loo = LeaveOneOut(len(data))
    cvData = []
    for (train, test) in list(loo):
        train = data.ix[data.index[train]]
        test = data.ix[data.index[test]]
        tExt, seriesExt = getParams(train, phase, period)
        cvData.append(dict(train=train, test=test, tExt=tExt, seriesExt=seriesExt))
        
    fits = [fitModel(dS['tExt'],dS['seriesExt'], gammas) for dS in cvData]
    for i, cvSet in enumerate(fits):
            for fit in cvSet:
                cvSet[fit]['error'] = getError(cvSet[fit]['model'], cvData[i]['test'], 
                                      period, phase)
        
    fits = Panel(dict([(i, DataFrame(cvSet)) for (i, cvSet) in enumerate(fits)]))
    return fits, normedDays, period, phase

def crossValidation(series, period, phase, gammas=[]):
    if len(gammas) == 0:
        gammas = list(2**arange(-9.,-3))

    loo = LeaveOneOut(len(series))
    cvData = []
    for (train, test) in list(loo):
        train = series.ix[series.index[train]]
        test = series.ix[series.index[test]]
        tExt, seriesExt = getParams(train, phase, period)
        cvData.append(dict(train=train, test=test, tExt=tExt, 
                           seriesExt=seriesExt))
        
    fits = [fitModel(dS['tExt'],dS['seriesExt'], gammas) for dS in cvData]
    for i, cvSet in enumerate(fits):
            for fit in cvSet:
                cvSet[fit]['error'] = getError(cvSet[fit]['model'], 
                                               cvData[i]['test'], 
                                               period, phase)
        
    fits = Panel(dict([(i, DataFrame(cvSet)) for (i, cvSet) in enumerate(fits)]))
    return fits
