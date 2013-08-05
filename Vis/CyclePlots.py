'''
Created on Nov 3, 2011

@author: agross
'''
import matplotlib.patches as patches #@UnresolvedImport
import matplotlib.pyplot as plt #@UnresolvedImport
import numpy as numpy
import scipy.linalg as linalg

def centered(matrix):
    mean = numpy.mean(matrix, axis=0)
    return matrix-mean

def plotTimecourse(lines, conditions, ax='', **cyclePlotArgs):
    '''
    wrapper for cyclePlot
    '''
    if ax == '': fig, ax = plt.subplots(1,1)
    cyclePlot(ax, lines, conditions, **cyclePlotArgs)
    
def plotExpressionData(expressionData, centerData=False, ax='Default', **cyclePlotArgs):
    '''
    wrapper for cyclePlot
    '''
    if ax=='Default': 
        fig, ax = plt.subplots(1,1)
    if numpy.rank(expressionData)==1:
        'check if its a Data-Frame or Series-Frame'
        conditions = numpy.array(map(list, expressionData.index.values))
    else:
        conditions = numpy.array(map(list, expressionData.columns.values))
    if centerData:
        cyclePlot(ax, centered(expressionData.T), conditions, **cyclePlotArgs)
    else:
        cyclePlot(ax, expressionData.T, conditions, **cyclePlotArgs)
    
    
def cyclePlot(ax, lines, conditions, times=[], title='', xlabel='Time (h past zt0)', ylabel='Relative Expression', bars=True, plots=True, barHeight=-1, alpha=.7, **plotArgs):
    colorDict = {'Light':'yellow', 'Dark':'.2', 'Hot':'red', 'Cold':'blue'}
    if plots:
        
        if len(times) != 0: 
            ax.plot(times, numpy.array(lines).T, 'o', alpha=alpha, markersize=5, linestyle='-', **plotArgs)
        else: 
            ax.plot(lines, 'o', alpha=alpha, markersize=5, linestyle='-', **plotArgs)
        #ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
        ax.set_xticks(range(len(lines)))
        if len(times) == 0: 
            ax.set_xticklabels(conditions[:,0])
        ax.set_axisbelow(True)
        ax.set_xlim(-.5,len(lines)-.5)
        if len(times) != 0: 
            ax.set_xlim(-.5,max(times)+.5)
        ax.axhline(y=0, linewidth=4, color='black', alpha=.3, linestyle='--',zorder=-1)
        if title != '': ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
            
    if bars:
        #Here I add the color bars
        if barHeight < 0: barHeight = min(map(abs, ax.get_ybound()))*.5 
        i = 0
        if len(times) != 0:
            dX = 4
        else: 
            dX = 1
        for window in conditions:
            light = patches.Rectangle((i+.1,0), width=.8*dX,
                                      height=barHeight,
                                      color=colorDict[window[1]],
                                      alpha=0.2)
            ax.add_patch(light)
            temp = patches.Rectangle((i+.1,0), width=.8*dX,
                                     height=-barHeight,
                                     color=colorDict[window[2]],
                                     alpha=0.2)
            ax.add_patch(temp);
            i+=dX
            
def cyclePlot2(ax, lines, conditions, title='', xlabel='Time (h past zt0)', ylabel='Relative Expression', bars=True, plots=True, barHeight=-1):
    colorDict = {'Light':'yellow', 'Dark':'.2', 'Hot':'red', 'Cold':'blue'}
    if plots:
        ax.plot(lines, 'o', alpha=.7, markersize=5, linestyle='-')
        #ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
        ax.set_xticks(range(len(lines)))
        ax.set_xticklabels(conditions['light'])
        ax.set_axisbelow(True)
        ax.set_xlim(-.5,len(lines)-.5)
        ax.axhline(y=0, linewidth=4, color='black', alpha=.3, linestyle='--',zorder=-1)
        if title != '': ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
            
    if bars:
        #Here I add the color bars
        if barHeight < 0: barHeight = min(map(abs, ax.get_ybound()))*.5 
        i = 0
        for window in conditions:
            light = patches.Rectangle((i+.1,0), width=.8,
                                      height=barHeight,
                                      color=colorDict[window[1]],
                                      alpha=0.2)
            ax.add_patch(light)
            temp = patches.Rectangle((i+.1,0), width=.8,
                                     height=-barHeight,
                                     color=colorDict[window[2]],
                                     alpha=0.2)
            ax.add_patch(temp);
            i+=1
            
def horizontalPlots(lines, conditions, labels, **subplotsArgs):
    '''
    Plots a set of lines as a row of cyclePlots
    inputs:
      lines:        list of series to plot
      conditions:   list (or single set) of conditions, see cyclePlot for description
    '''
    fig, axs = plt.subplots(1,len(lines), **subplotsArgs)
    if len(numpy.array(conditions).shape) ==2: 
        conditionList = [conditions for i in range(len(lines))]
    else: conditionList = conditions
    if type(labels)==type('string'): 
        labelList = [labels for i in range(len(lines))]
    else: labelList = labels
    for i in range(len(lines)):
        cyclePlot(axs[i], lines[i], conditionList[i], title=labelList[i], ylabel='', bars=False, plots=True);
    for i in range(len(lines)):
        cyclePlot(axs[i], lines[i], conditionList[i], title='', ylabel='', bars=True, plots=False);
    #fig.subplots_adjust(hspace=.2);
    plt.setp([ax.get_yticklabels() for ax in fig.axes[1:]], visible=False);
    axs[0].set_ylabel('Relative Expression');
    
def horizontalPlotsFrame(expressionData, centerData=False, **subplotsArgs):
    '''
    Plots a set of lines as a row of cyclePlots
    inputs:
      lines:        list of series to plot
      conditions:   list (or single set) of conditions, see cyclePlot for description
    '''
    if numpy.rank(expressionData)==1:
        'check if its a Data-Frame or Series-Frame'
        experiments = list(expressionData.index.levels[0])
    else:
        experiments = list(expressionData.columns.levels[0])
    fig, axs = plt.subplots(1,len(experiments), **subplotsArgs)

    for i,exp in enumerate(experiments):
        plotExpressionData(expressionData[exp], centerData=centerData, ax=axs[i], title=experiments[i], xlabel='', ylabel='', bars=False, plots=True)
    for i,exp in enumerate(experiments):
        plotExpressionData(expressionData[exp], centerData=centerData, ax=axs[i], title='', ylabel='', bars=True, plots=False)
    plt.setp([ax.get_yticklabels() for ax in fig.axes[1:]], visible=False);
    axs[0].set_ylabel('Relative Expression');

def verticalPlots(lines, conditions, **subplotsArgs):
    fig, axs = plt.subplots(len(lines),1, **subplotsArgs)
    if len(numpy.array(conditions).shape) ==2: 
        conditionList = [conditions for i in range(len(lines))]
    else: conditionList = conditions
    for i in range(len(lines)):
        cyclePlot(axs[i], lines[i], conditions[i], title='', xlabel='', ylabel='', bars=False);
    for i in range(len(lines)):
        cyclePlot(axs[i], lines[i], conditions[i], title='', plots=False);
        fig.subplots_adjust(hspace=.2);
        plt.setp([ax.get_xticklabels() for ax in fig.axes[:-1]], visible=False);
        axs[-1].set_xlabel('Time (h past zt0)');
        axs[3].set_ylabel('Relative Expression'); 
        
def verticalPlotsFrame(expressionData, **subplotsArgs):
    experiments = list(expressionData.index.levels[0])
    fig, axs = plt.subplots(len(experiments),1, **subplotsArgs)
    for i,exp in enumerate(experiments):
        plotExpressionData(expressionData[exp], centerData=False, ax=axs[i], title='', xlabel='', ylabel='', bars=False)
    for i,exp in enumerate(experiments):
        plotExpressionData(expressionData[exp], centerData=False, ax=axs[i], title='', plots=False)
        fig.subplots_adjust(hspace=.2);
        plt.setp([ax.get_xticklabels() for ax in fig.axes[:-1]], visible=False);
        axs[-1].set_xlabel('Time (h past zt0)');
        axs[3].set_ylabel('Relative Expression'); 
