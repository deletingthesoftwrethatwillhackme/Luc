'''
Created on Aug 4, 2013

@author: agross
'''
import os as os
import numpy as np

import skimage.io as io

def stackAuto(tmpDir, plate):
    outDir = tmpDir + 'Original/'
    if not os.path.isdir(outDir):
        os.makedirs(outDir)
    else:
        for f in os.listdir(outDir):
            os.remove(outDir+f)
    for k in range(len(plate)): 
        io.imsave(outDir+str(k)+'.TIF', plate[k])
        
    inDir = tmpDir + 'Aligned/'
    if not os.path.isdir(inDir):
        os.makedirs(inDir)
          
    #need to make sure there is signal to start the image alignment
    #don't want to take the max value or the first few values to avoid
    #image saturation
    startingPlate = np.argsort(plate.sum(1).sum(1)[5:])[-2]+5+1
        
    imageJ_macro = '''
    startImage = {};
    stackArg = "open={} number=100 starting=1 increment=1 scale=100 file=[] or=[] sort" 
    run("Image Sequence...", stackArg);
    setSlice(startImage);
    run("StackReg ", "transformation=[Rigid Body]");
    run("Image Sequence... ", "format=TIFF name=Aligned start=0 digits=4 save={}");
    '''.format(startingPlate, outDir, inDir)
    
    #get the directory to put an ImageJ plugin
    os.system('imagej -b fail > tmp.txt');
    fail_message = open('tmp.txt', 'rb').readlines()
    path = '/'.join(fail_message[-1].split('/')[:-1])
    
    f = open(path + '/macro.ijm', 'wb')
    f.write(imageJ_macro)
    f.close()
    
    out = os.system('imagej -b macro');
    
    fList = os.listdir(inDir)
    stack = np.array([io.imread(inDir+f, as_grey=False, plugin=None, flatten=None) 
                 for f in sorted(fList)])
    return stack