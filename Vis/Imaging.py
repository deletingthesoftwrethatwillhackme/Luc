'''
Created on Aug 16, 2012

@author: agross
'''
import matplotlib.pyplot as plt
from numpy import arange

def plot_image_traces(plate, trace1, trace2, r=10):
    fig,axs = plt.subplots(3,1, figsize=(30,5))
    length = (r*2)+1
    colors = ['b','g']
    for i,num in enumerate([trace1, trace2]):
        c = plate.centers[num]
        section = plate.stack[:,c[1]-r:c[1]+r+1,c[0]-r: c[0]+r+1]
        axs[i].imshow(section.reshape([len(section)*length,length]).T, aspect=1, 
                      interpolation='Nearest')
        axs[2].plot(arange(len(section))+.5, plate.traces[num], 'o', color=colors[i], 
                    ms=15)
        axs[2].plot(arange(len(section))+.5, plate.traces[num], '--', color=colors[i], 
                    lw=5, dash_capstyle='round', alpha=.5)
        axs[2].set_xbound(0,len(section))
    fig.tight_layout()
    
    
def plot_image_trace(plate, trace, r=10):
    fig,axs = plt.subplots(2,1, figsize=(30,3))
    length = (r*2)+1
   
    c = plate.centers[trace]
    section = plate.stack[:,c[1]-r:c[1]+r+1,c[0]-r: c[0]+r+1]
    axs[0].imshow(section.reshape([len(section)*length,length]).T, aspect=1, 
                  interpolation='Nearest')
    axs[1].plot(arange(len(section))+.5, plate.traces[trace], 'o', ms=15, color='b')
    axs[1].plot(arange(len(section))+.5, plate.traces[trace], '--', color='b', 
                lw=5, dash_capstyle='round', alpha=.5)
    axs[1].set_xbound(0,len(section))
    fig.tight_layout()