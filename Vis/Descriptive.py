'''
Created on Nov 3, 2011

@author: agross
'''
import numpy as np
import matplotlib.pyplot as plt

def clipMe(vector, p=.01):
    '''
    Function to clip the extreme values of a vector to make better histograms
    input:
      vector: vector to clip
      p:      quantile to drop from 
              (ie. p=.01 returns values between 1st and 99th percentiles)
    output:
      clipped vector 
    '''
    from scipy.stats.mstats import mquantiles #@UnresolvedImport
    vec = np.array(vector)
    [min,max]= mquantiles(vec, prob=[p,1-p])
    return vec.take(np.where((vec > min) & (vec < max))[0])

def statsBarCharts(dataMatrix, **subplotArgs):
    '''
    Draw up some histograms describing a matrix of time-series data
    Right now hard coded for the 6X12XGenes set of time-series
    '''
    fig, axs = plt.subplots(2, 3, **subplotArgs)
    def annotateMe(ax, string):
        ax.annotate(string, (.95,.9), xycoords='axes fraction', horizontalalignment='right', 
                    verticalalignment='top', fontsize=15)
    axs[0,0].hist(clipMe(dataMatrix.flatten()), bins=20);
    annotateMe(axs[0,0], 'Data Distribution')    
    axs[0,1].hist(clipMe(np.mean(dataMatrix, axis=1)), bins=20);
    annotateMe(axs[0,1], 'Time-course Mean')
    axs[0,2].hist(clipMe(np.var(dataMatrix, axis=1)), bins=20);
    annotateMe(axs[0,2], 'Time-course Variance')
    axs[1,0].hist(clipMe(np.max(dataMatrix, axis=1) - np.min(dataMatrix, axis=1)), bins=20);
    annotateMe(axs[1,0], 'Time-course Range')
    dayIndicies = [i for i in range(len(dataMatrix[1])) if i%12<6]
    nightIndicies = [i for i in range(len(dataMatrix[1])) if i%12>=6]
    crossDayDiff = dataMatrix[:,0:6] - dataMatrix[:,6:12]
    axs[1,1].hist(clipMe(crossDayDiff.flatten()), bins=20);
    annotateMe(axs[1,1], 'Cross Day Diffence')
    axs[1,2].hist(clipMe(np.mean(crossDayDiff, axis=1)), bins=20);
    annotateMe(axs[1,2], 'Cross Day Mean')
    
def bubblePlots(series, labels, title='', xlabel='', ax1='', ylabel='Relative Expression'):
    '''
    Plots out a set of series using large bubbles
    '''
    if ax1=='':
        fig  = plt.figure()
        ax1 = fig.add_subplot(111)
    ax1.plot(series, 'o', alpha=.5, markersize=20);
    ax1.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax1.set_xticklabels(labels, rotation=40, horizontalalignment='right')
    ax1.set_xticks(range(len(labels)))
    ax1.set_axisbelow(True)
    ax1.set_xlim(-.5,len(labels)-.5)
    (min_r, max_r) = (np.min(series), np.max(series))
    range_r = max_r - min_r
    ax1.set_ylim(min_r - .05*range_r, max_r + .05*range_r)
    ax1.axhline(y=0, linewidth=4, color='black', alpha=.3, linestyle='--',zorder=-1)
    if title != '' : ax1.set_title(title)
    if xlabel != '': ax1.set_xlabel(xlabel)
    if ylabel != '': ax1.set_ylabel(ylabel);
    
def stemEigenvalues(S, eigs=10, ax1='', secondAxisStart=1):
    if type(eigs)==type(0):
        useEigs = range(min(eigs,len(S)))
    else:
        useEigs = eigs
    S_norm = np.array(pow(S,2)/sum(pow(S,2)))
    #fig = plt.figure()
    if ax1=='': ax1 = plt.axes([.1,.1,.8,.8])
    ax1.yaxis.tick_left()
    ax1.stem(np.array(useEigs)+1, S_norm.take(useEigs), 
             markerfmt='bo');
    if secondAxisStart > 0:
        ax2 = plt.axes([.1,.1,.8,.8], frameon=False)
        ax2.yaxis.tick_right()
        ax2.stem(np.array(useEigs[secondAxisStart:])+1, S_norm.take(useEigs[secondAxisStart:]), 
                 markerfmt='ro');
        #ax2.set_xbound(min(useEigs)-.5,max(useEigs)+.5)
        ax2.set_xticks([])
        ax2.set_xbound(.5,len(useEigs)+.5)
        ax2.set_ybound(0,S_norm[secondAxisStart]*1.25)
    
    ax1.set_xbound(.5,len(useEigs)+.5)    
    ax1.set_ybound(0,S_norm[0]*1.05)  
    ax1.set_title('Percentage of Variation Captured By Eigenvalues');
    
def getEntropy(S, ax=None):
    p = S**2/sum(S**2)
    entropy = -(np.log(len(S))**-1)*(sum([pK*np.log(pK) for pK in p]))
    if ax is not None:
        ax.bar(range(len(S)), p);
    return entropy

def barEigenvalues(S, eigs=10, colorEigs=None, showEntropy=True, ax=''):
    if type(eigs)==type(0):
        useEigs = np.array(range(min(eigs,len(S))))
    else:
        useEigs = eigs
    S_norm = S**2/sum(S**2)
    #fig = plt.figure()
    if ax=='': 
        ax = plt.gca()
    ax.yaxis.tick_left()
    
    if colorEigs is not None:
        colorCycle = plt.rcParams['axes.color_cycle']
        for i in range(colorEigs):
            ax.bar(useEigs[i]+1, S_norm[useEigs[i]], align='center', 
                   color=colorCycle[i%len(colorCycle)]);
        ax.bar(useEigs[colorEigs:]+1, S_norm.take(useEigs[colorEigs:]), align='center',
               color='grey');
    else:
        ax.bar(useEigs+1, S_norm.take(useEigs), align='center');
        
    if showEntropy:
        entropy = -(np.log(len(S))**-1)*(sum([pK*np.log(pK) for pK in S_norm]))
        ax.annotate('Shannon Entropy = %.2f' % entropy, (-10,-10), xycoords='axes points',
                    ha='right', va='top', size=12)
    
    ax.set_xbound(.5,len(useEigs)+.5)    
    ax.set_ybound(0,S_norm[0]*1.05)  
    ax.set_xticks(useEigs+1)
    ax.set_ylabel('Percentage of Variation');
    ax.set_xlabel('Eigenvalue');
    
def annotated_scatter(x, y, annotations, ax=None, **scatter_args):
    import pandas as pandas
    if ax is None: ax = plt.gca()
    sc = ax.scatter(x, y, **scatter_args)
    for px, py, ann in zip(x, y, annotations):
        ax.annotate(ann, xy=(px, py), ha='center', va='center')   
        
    if type(x) == pandas.Series:
        ax.set_xlabel(x.name)
    if type(y) == pandas.Series:
        ax.set_ylabel(y.name)
    return sc

    
