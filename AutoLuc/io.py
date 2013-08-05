'''
Created on May 10, 2012

@author: agross
'''
import numpy as np
import os as os

import pandas as pandas
from skimage import io, filter #@UnresolvedImport

def getFilesForRun(runNum, fList):
    run = fList[fList['Run']==runNum].sort(columns='Filename')
    def strTake(s): return s[:2]
    r = run['TimePoint'].map(strTake)
    unique = np.sort(np.unique(r)).index
    run = run.ix[unique].sort(columns='Filename')
    return list(run['Filename'])

def getPlates(fList, inPath):
    '''Takes a list of files and a path, reads in image files and quarters them'''
    stack = np.array([io.imread(inPath+f, as_grey=False, plugin=None, 
                                   flatten=None) for f in fList])
    stackQuartered = np.array([[stack[:,:240,:320], stack[:,:240,320:]], 
                               [stack[:,240:,:320], stack[:,240:,320:]]])
    #stackQuartered = np.array([[[filter.median_filter(stackQuartered[i,j,k], 
    #                                                     radius=5) for k in range(41)] 
    #                               for j in range(2)] for i in range(2)])
    return stackQuartered


def fillInMissingFiles(filesForRun, runNum):
    '''
    Fills in files that are missing in the stack.  Assumes the files are
    numbered 1-41.
    '''
    def getNum(s): return int(float(s.split('-')[1][:2]))
    s = set(map(getNum, filesForRun))
    missing = list(s.symmetric_difference(range(1,42)))

    for missingFile in missing:
        if set(range(1,missingFile+1)).issubset(set(missing)):
            filesForRun.append(min(filesForRun))
        else: 
            s = str(missingFile-1)
            if len(s)==1:
                filesForRun.append(str(runNum)+'-0'+s+'.TIF')
            else:
                filesForRun.append(str(runNum)+'-'+s+'.TIF')
    return sorted(filesForRun)

def convertRun(r): 
    try: return int(r)
    except: r = -1
    
def parse(f): 
    return np.append(f.split('.TIF')[0].split('-'), f)

def getFileList(inPath):
    '''Gets the list of .TIF files in the path and puts it in a DataFrame'''
    fList = [parse(f) for f in os.listdir(inPath) if f.find('.TIF') != -1]
    fList = pandas.DataFrame(fList, columns=['Run','TimePoint','Filename'])
    fList['Run'] = fList['Run'].map(convertRun)

    return fList

def getTimesFromStack(inPath, filesForRun):
    '''
    Digs through the image meta-data to pull time stamps.
    '''
    import datetime as datetime
    
    times = []
    for fileH in filesForRun:
        f = open(inPath+fileH,'rb')
        l = f.read()
        infoString = l[l.find('__HPDEXT__'):l.find('~WASABI~')]
        split = infoString.split(';')
        params = map(int, split[0][split[0].find('{')+1:split[0].find('}')].split(','))
        info = dict([tuple(f.split('=')) for f in split[1:] if f.find('=') >-1])
        y,mo,d = map(int, info['vDate'].split('/'))
        h,mi,s = map(int, info['vTime'].split(':'))
        time = datetime.datetime(y, mo, d, h, mi, s)
        times.append(time)
    times = pandas.Series(times)
    d = times - times[0]
    times = d.astype(int) / (1e9 * 3600) #confert nano-seconds to hours
    #times = pandas.Series(arange(0,102.5, 2.5))
    times.name = 'time'
    return times