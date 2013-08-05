'''
Created on Feb 9, 2012

@author: agross
'''
import pandas as pandas
import os as os
import numpy as numpy

def extractROI(roi):
    '''
    So I'm geting mean ROI's from imageJ, I just want the label to be
    ROI so I can merge with annotations.
    '''
    return roi.split('(')[1].split(')')[0]

def reformatROI(roi):
    '''
    My formating on the ROIs was 0 indexed and not 0 padded.
    Here I'm converting to be compatible with imageJ.
    '''
    s = str(int(roi.split(' ')[1])+1)
    return 'ROI'+s.rjust(3, '0')


def readData(path):
    '''
    Read in the luciferase data given a path to a folder.
    '''
    data = pandas.read_csv(path+'/processedSignals.csv', index_col=0).T
    parsedIndex = map(extractROI, data.index)
    data = pandas.DataFrame(data.as_matrix(), index=parsedIndex, 
                            columns=data.columns)
    return data

def getVariantAnnotations(variantConditions, folder, run, plate):
    '''
    I screwed up a little in the form processing so I'm just having to loop through and 
    parse it out manually.  Each variant gets 4 annotations media, reporter, genotype, and
    selection in that order, here I'm just filling in the dictionary mapping
    '''
    numVariants = len(variantConditions)/4
    annotationLookup = {0:'media', 1:'reporter', 2:'genotype', 3:'selection'}
    variantAnnotations = dict([(variant, dict(media='', reporter='', genotype='', 
                                              selection='',folder=folder, run=run, 
                                              plate=plate, variantNum='')) 
                               for variant in range(numVariants)])
    for i,con in enumerate(variantConditions):
        if len(con)>3:
            ann = con[3]
        else: ann = 'None'
        variantAnnotations[i/4][annotationLookup[i%4]] = ann
        if i%4==0: variantAnnotations[i/4]['variantNum']=i/4
            
    variantAnnotations = pandas.DataFrame(variantAnnotations)
    return variantAnnotations

def getAnnotations(path, folder, run, plate):
    def inGroup(ann, group): return ann[0] == group
    annotationFile = path+folder+'/'+run+'/'+plate+'/out.txt'
    if not os.path.exists(annotationFile):
        return False
    annotations  = file.readlines(open(annotationFile,'rb'))
    annotations  = map(lambda s: s.strip().split('\t'), annotations)
    
    roiAssociations = [ann for ann in annotations if inGroup(ann, 'roiAssociations')]
    roiAssociations = dict([(reformatROI(roi[1]), int(roi[2])) for roi in roiAssociations])
    roiAssociations = pandas.Series(roiAssociations)
    
    plateConditions = [ann for ann in annotations if inGroup(ann, 'plateConditions')]
    plateAnnotations = dict([(con[1], con[2]) for con in plateConditions])
    plateAnnotations['folder'] = folder
    plateAnnotations['run']    = run
    plateAnnotations['plate']  = plate
    plateAnnotations = pandas.Series(plateAnnotations)
    plateAnnotations.name = (folder, run, plate)
    
    variantConditions = [ann for ann in annotations if inGroup(ann, 'variantConditions')]
    variantAnnotations = getVariantAnnotations(variantConditions, folder, run, plate)
    return dict(roiAssociations=roiAssociations, plateAnnotations=plateAnnotations,
                variantAnnotations=variantAnnotations)
    
def compileAnnotations(data, roiAssociations, variantAnnotations, plateAnnotations):
    def getVariantAnnotation(variantNum): 
        return variantAnnotations[variantNum]
    roiAnnotations = pandas.DataFrame(map(getVariantAnnotation, roiAssociations), 
                                      index=roiAssociations.index,
                                      columns=variantAnnotations.index)
    roi = pandas.Series(roiAnnotations.index, index=roiAnnotations.index, name='ROI')
    roiAnnotations = roiAnnotations.join(roi)
        
    testConditions = pandas.Series(numpy.repeat(plateAnnotations['testConditions'], 
                                          len(roiAnnotations)),
                                   index=roiAnnotations.index, name='testConditions')
    roiAnnotations = roiAnnotations.join(testConditions)
    
    lightIntensity = pandas.Series(numpy.repeat(plateAnnotations['lightIntensity'], 
                                          len(roiAnnotations)),
                                   index=roiAnnotations.index, name='lightIntensity')
    roiAnnotations = roiAnnotations.join(lightIntensity)
    
    trainingConditions = pandas.Series(numpy.repeat(plateAnnotations['trainingConditions'], 
                                              len(roiAnnotations)),
                                       index=roiAnnotations.index, name='trainingConditions')
    roiAnnotations = roiAnnotations.join(trainingConditions)
    
    treatment = pandas.Series(numpy.repeat(plateAnnotations['treatment'], len(roiAnnotations)),
                              index=roiAnnotations.index, name='treatment')
    roiAnnotations = roiAnnotations.join(treatment)    

    multiIndex = pandas.MultiIndex.from_arrays(roiAnnotations[['folder','run','plate','ROI']].as_matrix().T)
    roiAnnotations = pandas.DataFrame(roiAnnotations.as_matrix(), 
                                      index=multiIndex,
                                      columns = roiAnnotations.columns)

    data = data.ix[roiAssociations.index]
    data.index = multiIndex
    return data, roiAnnotations

def processPlate(path, folder, run, plate):
    data = readData(path+folder+'/'+run+'/'+plate)
    annotations = getAnnotations(path, folder, run, plate)
    if not annotations:
        return pandas.DataFrame([]), pandas.DataFrame([])
    roiAssociations = annotations['roiAssociations']
    plateAnnotations = annotations['plateAnnotations']
    variantAnnotations = annotations['variantAnnotations']
    data, roiAnnotations = compileAnnotations(data, roiAssociations, variantAnnotations,
                                              plateAnnotations)
    return data, roiAnnotations

def getAllDataInDirectory(dP):
    data = pandas.DataFrame([])
    annotations = pandas.DataFrame([])
    for folder in  os.listdir(dP):
        runs = os.listdir(dP+folder)
        for run in runs:
            plates  = os.listdir(dP+folder+'/'+run)
            for plate in plates:
                plateData, plateAnnotations = processPlate(dP, folder, run, plate)
                data = data.append(plateData)
                annotations = annotations.append(plateAnnotations)
    return data, annotations


