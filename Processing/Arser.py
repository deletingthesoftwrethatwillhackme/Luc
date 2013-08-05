'''
ARSER: Analysis circadian expression data by harmonic regression 
based on autoregressive spectral estimation
Usage: pyhton arser.py inputfile outputfile
Author: Rendong Yang
Email: cauyrd@gmail.com

Note that I tweaked the original code to use rpy2 instead of rpy (--Andy)
'''

import ArserUtility as HR
import rpy2.robjects as robjects
from pylab import detrend_linear
import numpy as numpy
import pandas as pandas
import sys

class Arser:
    '''
    Doc: Class for harmonic regression with MESE period identification
    Input: 
          x values -> time points
          y values -> expression value
    Output:
          self.x -> raw x values
          self.y -> raw y values
          self.mean -> mean value for raw y values
          self.dt_y -> y values detrended
          self.delta -> sampling interval
          self.estimate_period -> period identified by MESE
          self.amplitude -> amplitude for single cosine model
          self.phase -> phase for single cosine model
          self.R2 -> R square of regression curve
          self.R2adj -> adjusted R square of regression curve
          self.coef_var -> (standard deviation)/mean
          self.pvalue -> F test for testing significant regression model
    '''
    
    def __init__(self, x, y):
        '''
            initialized Arser instance object, detrendedy values, mean, period are calculated
        '''
        self.x = numpy.array(x)
        self.y = numpy.array(y)
        self.mean = numpy.mean(self.y)
        self.dt_y = detrend_linear(self.y)
        self.delta = self.x[1] - self.x[0]
        self.R2 = -1
        self.R2adj = -1
        self.coef_var = -1
        self.pvalue = -1
        self.phase = []
        self.estimate_period = []
        self.amplitude = []

    def get_period(self, is_filter=True, ar_method='mle'):
        '''
        estimate possiable cycling period of time series
        '''
        num_freq_mese = 500
        set_order = 24/self.delta
        if (set_order == len(self.x)):
            set_order = len(self.x)/2
        try:
            filter_y = HR.savitzky_golay(self.dt_y)
        except:
            filter_y = HR.savitzky_golay(self.dt_y, kernel=5, order=2)
            
        spec_ar = robjects.r['spec.ar']
        if is_filter:
            try:
                mese = spec_ar(robjects.FloatVector(filter_y), n_freq=num_freq_mese, 
                               plot=False, method=ar_method, order=set_order)
            except:
                return []
        else:
            try:
                mese = spec_ar(robjects.FloatVector(self.dt_y), n_freq=num_freq_mese, 
                               plot=False, method=ar_method, order=set_order)
            except:
                return []
        mese = dict(freq=numpy.array(mese.rx('freq'))[0], spec=numpy.array(mese.rx('spec')[0]))
        # search all the locial peaks of maximum entropy spectrum
        peaks_loc = []      # the locition for each peak in mese spectrum
        for i in range(1, num_freq_mese-1):
            if mese['spec'][i] > mese['spec'][i+1] and mese['spec'][i] > mese['spec'][i-1]:
                peaks_loc.append((mese['spec'][i], i))
        peaks_loc.sort(reverse=True)    # sort frequency by spectrum value
        try:
            periods = [1/mese['freq'][item[1]]*self.delta for item in peaks_loc]
        except:
            periods = []
        return periods

    def harmonic_regression(self, period):    
        '''
        general harmonic regression
        dt_y = mesor + sigma( A*cos(2*pi/T*x) + B*sin(2*pi/T*x) ) + error
        '''
        x = numpy.array([])
        x = x.reshape(len(self.x), 0)
        x_varnm_names = []
        for T in period:
            cosx = numpy.cos(2*numpy.pi/T*self.x)
            sinx = numpy.sin(2*numpy.pi/T*self.x)
            x = numpy.c_[x, cosx, sinx]
            x_varnm_names += ['cos%.1f' % T,'sin%.1f' % T]
        model = HR.ols(self.dt_y, x, y_varnm = 'y', x_varnm = x_varnm_names)
        return model

    def evaluateNew(self, T_start=15, T_end=30):
        '''
        Pseudo brute force but seems to work better for me
        '''
        'Find AIC scores for all candidate periods'
        candidates = pandas.Series(dict([(p, self.harmonic_regression([p]).ll()[1]) 
                                for p in numpy.arange(T_start,T_end,.01)]))
        self.estimate_period = candidates.index[candidates.argmin()]
        model = self.harmonic_regression([self.estimate_period])
        
        m = model
        #return m
        phi = numpy.angle(complex(m.b[1], -m.b[2]))  
        if phi<=1e-6:
            self.phase = numpy.fabs(phi)/(2*numpy.pi)*self.estimate_period
        else:
            self.phase = self.estimate_period - phi/(2*numpy.pi)*self.estimate_period
            
        self.amplitude = numpy.sqrt(m.b[1]**2 + m.b[2]**2)
        self.R2 = m.R2
        self.R2adj = m.R2adj
        self.pvalue = m.Fpv
        self.coef_var = numpy.std(self.y)/self.mean
        
        return {'period':self.estimate_period, 'amplitude':self.amplitude, 
                'phase':self.phase, 'R2':self.R2, 
                'R2adj':self.R2adj, 'coef_var':self.coef_var, 
                'pvalue':self.pvalue
                }
        
    def evaluate(self, T_start=15, T_end=30, T_default=24):    
        '''
        evaluate the best model for each time series
        '''
        is_filter = [True, False]
        ar_method = ['yule-walker', 'mle', 'burg']
        best_model = {'AIC':1e6, 'ols_obj':None, 'period':None, 'filter':None, 'ar_method':''}
        for p1 in is_filter:
            for p2 in ar_method:             
                # choose best model's period from 'mle','yw','burg'
                period = filter((lambda x:x>=T_start and x<=T_end), self.get_period(is_filter=p1, ar_method=p2))
                if not period:
                    p2 = 'default'
                    period = [T_default]
                m = self.harmonic_regression(period)

                # model selection by aic
                aic = m.ll()[1]
                if aic <= best_model['AIC']:
                    best_model['AIC'] = aic
                    best_model['ols_obj'] = m
                    best_model['period'] = period
                    best_model['filter'] = p1
                    best_model['ar_method'] = p2

        # record the best model parameters
        self.estimate_period = best_model['period']
        self.amplitude = []
        self.phase = []
        m = best_model['ols_obj']
        for i in range(len(self.estimate_period)):
            phi = numpy.angle(complex(m.b[2*i+1], -m.b[2*i+2]))            

            # for float point number can not compare with 0, so use <1e-6 as nonpositive real number
            self.phase.append(numpy.fabs(phi)/(2*numpy.pi)*self.estimate_period[i] if phi<=1e-6 else self.estimate_period[i] - phi/(2*numpy.pi)*self.estimate_period[i]) 
            self.amplitude.append(numpy.sqrt(m.b[2*i+1]**2 + m.b[2*i+2]**2))
        self.R2 = m.R2
        self.R2adj = m.R2adj
        self.pvalue = m.Fpv
        self.coef_var = numpy.std(self.y)/self.mean
        return {'period':self.estimate_period, 'amplitude':self.amplitude, 
                'phase':self.phase, 'R2':self.R2, 
                'R2adj':self.R2adj, 'coef_var':self.coef_var, 
                'pvalue':self.pvalue, 'filter':best_model['filter'], 
                'ar_method':best_model['ar_method']}