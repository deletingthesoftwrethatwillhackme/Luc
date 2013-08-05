'''
Created on May 15, 2012

@author: agross
'''
import numpy as np
from matplotlib.mlab import detrend_linear

class JTK:
    '''
    Wrapper for JTK_CYCLE R script.
    '''
    
    def __init__(self, com):
        '''
        com is a instance of pandas.rpy.common.
        Here I'm porting my data over to R so I can use JTK cycle.
        I wrote a little wrapper script to make it a little easier to call.
        '''
        jtk_enviroment = com.robj.r('source("./Tools/JTK_CYCLE_WRAPPER.R")')
        self.initJTK = com.robj.globalenv['initJTK']
        self.runJTK = com.robj.globalenv['runJTK']
        self.com = com
        
    def createJTKBackground(self, times):
        start = times[0]
        interval = 2.5
        normTime = np.round((times - start) / interval)

        periods = self.com.robj.IntVector(range(4,16))
        self.initJTK(len(normTime)-3, periods, 2.5);
    
    def runJTK(self, data):
        d = dataset_to_data_frame(data.ix[3:].apply(detrend_linear).T)
        results = self.com.convert_robj(self.runJTK(d))
        return results
        
def dataset_to_data_frame(dataset, strings_as_factors=True):
    '''
    Taken from pandas forum... should be in the main package soon.
    '''
    import numpy as np
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    import rpy2.rlike.container as rlc


    base = importr("base")
    columns = rlc.OrdDict()

    '''Type casting is more efficient than rpy2's own numpy2ri'''
    vectors = {np.float64: robjects.FloatVector,
               np.float32: robjects.FloatVector,
               np.float: robjects.FloatVector,
               np.int: robjects.IntVector,
               np.int64: robjects.IntVector,
               np.object_: robjects.StrVector,
               np.str: robjects.StrVector}

    columns = rlc.OrdDict()

    for column in dataset:
        value = dataset[column]
        value = vectors[value.dtype.type](value)

        '''These SHOULD be fast as they use vector operations'''
        if isinstance(value, robjects.StrVector):
            value.rx[value.ro == "nan"] = robjects.NA_Character
        else:
            value.rx[base.is_nan(value)] = robjects.NA_Logical

        if not strings_as_factors:
            value = base.I(value)

        columns[column] = value

    dataframe = robjects.DataFrame(columns)
    del columns
    dataframe.rownames = robjects.StrVector(dataset.index)

    return dataframe
