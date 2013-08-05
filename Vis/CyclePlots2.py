'''
Created on Apr 11, 2012

@author: agross
'''
import numpy as np
import matplotlib.pyplot as plt

def cyclePlot(series, conditions={}, xlabel='Time (h past zt0)', 
              ylabel='Relative Expression', bars='both', title='', ax=None, 
              color=None, **plot_args):
    '''
    Fixed line color issue 8/14
    '''
    from pandas.core.index import MultiIndex
    
    if ax is None:
        ax = plt.gca()
    times = list(series.index if series.index != MultiIndex else 
                 series.index.get_level_values(-1))
        
    plot_range = (min(times) - (max(times)-min(times))*.05, 
                 max(times) + (max(times)-min(times))*.05)
    
    color_cylce = plt.rcParams['axes.color_cycle']
    ax.plot(times, np.array(series), '--', linewidth=4, alpha=.3, 
            dash_capstyle='round', **plot_args)
    ax.plot(times, np.array(series), 'o', alpha=.7, color='black')
    
    for i,line in enumerate(ax.get_lines()):
        line.set_color(color_cylce[(i/2)%len(color_cylce)])
        
    if (min(series) < 0) and (max(series) > 0):
        ax.axhline(y=0, linewidth=4, color='black', alpha=.3, linestyle='--',zorder=-1)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    
    if (bars is not None) and (bars != 'None') and len(conditions) > 0:
        '''Add the color bars'''
        light_displays = dict(on=dict(color='yellow'), off=dict(color='black'), 
                             off_entrained=dict(color='black', hatch='//'))
        color_displays = dict(hot=dict(color='red'), cold=dict(color='blue'), 
                             cold_entrained=dict(color='blue', hatch='//'))
        if np.rank(series) == 2:
            minH = series.min().min()
            maxH = series.max().max()
        else:
            minH = min(series)
            maxH = max(series)
        ax.set_ylim(bottom=minH - (maxH-minH)*.2)
        
        if bars == 'both':
            bar_height = .05
            bar_start_light, bar_start_temp = (0.05, 0.)
        else:
            bar_height = .1
            bar_start_light, bar_start_temp = (0, 0)
        num_days = int(np.ceil(max(times)/24.))
        for day,c in zip(range(num_days), conditions):
            if bars in ['temp', 'both']:
                ax.axvspan(c['start']+day*24, min(c['end']+day*24, plotRange[1]), 
                           ymin=bar_start_temp, ymax=bar_start_temp+bar_height, 
                           alpha=.2, **color_displays[c['temp']])
            if bars in ['light', 'both']:
                ax.axvspan(c['start']+day*24, min(c['end']+day*24, plotRange[1]),
                           ymin=bar_start_light, ymax=bar_start_light+bar_height, 
                           alpha=.2, **light_displays[c['light']])       
            
    ax.set_xbound(plot_range)

    
def plotRaster(data, ax=None):
    from matplotlib.colors import LinearSegmentedColormap
    
    if ax is None:
        ax=plt.gca()
    cdict1 = {'red':   ((0.0, 0.0, 0.0),
                        (0.5, 0.0, 0.1),
                        (1.0, 1.0, 1.0)),
    
             'green': ((0.0, 0.0, 1.0),
                       (0.5, 0.1, 0.0),
                       (1.0, 0.0, 0.0)),
    
             'blue':  ((0.0, 0.0, 0.0),
                       (1.0, 0.0, 0.0))
            }
    red_green1 = LinearSegmentedColormap('RedGreen1', cdict1)
    ax.imshow(data.as_matrix(), aspect=.02, cmap=red_green1)
    ax.set_yticks([])
    
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(map(int, data.columns));
    ax.set_xlabel('Time (h past zt0)');
    
def scaleMe(matrix):
    _min = np.min(matrix, axis=0)
    _max = np.max(matrix, axis=0)
    return (matrix-_min)/(_max-_min)