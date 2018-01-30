import numpy as np
import pandas as pd


class treeNode():
    def __init__(self, feat, val, right, left):
        featureToSplit = feat
        valueOfSplit = val
        rightBranch = right
        leftBranch = left


def loadData(filePath):
    dataSet = pd.read_table(filePath).values
    dataSet.astype("float32")
    return dataSet


# 将数据集在属性feature里value值分裂
def binarySplitData(dataSet, feature, value):
    dataSet1 = dataSet[dataSet[:, feature] > value]
    dataSet2 = dataSet[dataSet[:, feature] <= value]
    return dataSet1, dataSet2


# 结点的目标值的均值
def refLeaf(dataSet):
    return np.mean(dataSet[:, -1])


# 结点的目标值总方差
def regError(dataSet):
    return np.var(dataSet[:, -1]) * len(dataSet)


# 选择最优分裂属性和分裂点
def chooseBestSplit(dataSet, leafType=refLeaf, errType=regError, ops=(1, 4)):
    tolS = ops[0];
    tolN = ops[1]
    if len(set(dataSet[:, -1])) == 1:
        return None, leafType(dataSet)
    m, n = dataSet.shape
    S = errType(dataSet)
    bestVar = float("inf")
    bestFeat = -1
    bestValue = -1
    for featIndex in range(n - 1):
        valueSet = set(dataSet[:, featIndex])
        for value in valueSet:
            leftData, rightData = binarySplitData(dataSet, featIndex, value)
            if len(leftData) + len(rightData) < tolN: continue
            valueVar = errType(leftData) + errType(rightData)
            if valueVar < bestVar:
                bestFeat = featIndex
                bestValue = value
                bestVar = valueVar
    if S - bestVar < tolS:
        return None, leafType(dataSet)
    leftData, rightData = binarySplitData(dataSet,bestFeat, bestValue)
    if len(leftData) + len(rightData) < tolN:
        return None, leafType(dataSet)
    return bestFeat, bestValue


# 创建树
def createTree(dataSet, leafType=refLeaf, errType=regError, ops=(1, 4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None: return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binarySplitData(dataSet, feat, val)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    return retTree

def isTree(obj):
    return type(obj).__name__=='dict'

def getMean(tree):
    if isTree(tree['right']): tree['right']=getMean(tree['right'])
    if isTree(tree['left']): tree['left']=getMean(tree['left'])
    return (tree['right']+tree['left'])/2.0

# 剪枝函数
def prune(tree,testData):
    if np.shape(testData)[0]==0: return getMean(tree)
    if (isTree(tree['left'])or isTree(tree['right'])):
        lSet,rSet=binarySplitData(testData,tree['spInd'],tree['spVal'])
    if isTree(tree['left']) : tree['left']=prune(tree['left'],lSet)
    if isTree(tree['right']): tree['right']=prune(tree['right'],rSet)
    # 子结点皆为叶子结点
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet,rSet=binarySplitData(testData,tree['spInd'],tree['spVal'])
        errorNoMerge=sum((lSet[:,-1]-tree['left'])**2)+sum((rSet[:,-1]-tree['right'])**2)
        treeMean=(tree['right']+tree['left'])/2.0
        errorMerge=sum((testData[:,-1]-treeMean)**2)
        if errorMerge<errorNoMerge:
            print("merging")
            return treeMean
        else :
            return tree
    else :return tree



if __name__ == '__main__':
    trainData = loadData(r'ex2.txt')
    myTree=createTree(trainData)
    testData=loadData(r'ex2test.txt')
    print(prune(myTree,testData))

