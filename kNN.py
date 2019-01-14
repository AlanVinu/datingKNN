from numpy import *
import operator
import importlib

def createDataSet () :
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

#k-Nearest Neighbours
def classify0 (inX, dataSet, labels, k) :
    #distance Calculation Start   
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    #distance Calculation End
    classCount = {}
    for i in range(k) :
        voteIlabel = labels[sortedDistIndicies[i]]
        #voting with Lowest k distances
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(),
                              key = operator.itemgetter(1), reverse = True) #sort Dictionary
    return sortedClassCount[0][0]

#Text record to NumPy parsing code
def file2matrix(filename) :
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    fr = open(filename)
    index = 0
    for line in arrayOLines :
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

#data-normalizing function
def autoNorm(dataSet) :
    minVals = dataSet.min(0)            #min value across rows, returns the row with the minimum value
    maxVals = dataSet.max(0)            #max value across rows
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet)) #initializing as a zero matrix(array)
    m = dataSet.shape[0]                #finding out the number of rows
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))   #element-wise division
    return normDataSet, ranges, minVals

#Classifier testing code
def datingClassTest() :
    hoRatio = 0.07
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs) :
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],\
                                     datingLabels[numTestVecs:m],4)
        print ("the classifier came back with : %d, the real answer is : %d"\
              %(classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]) :
            errorCount += 1.0
    print ("the total error rate is : %f" %(errorCount/float(numTestVecs)))

#Dating site predictor function
def classifyPerson() :
    resultList = ['not at all','in small doses','in large doses']
    percentTats = float(input(\
        "percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-\
                                  minVals)/ranges,normMat,datingLabels,4)
    print("You will probably like this person : ",\
          resultList[classifierResult - 1])
