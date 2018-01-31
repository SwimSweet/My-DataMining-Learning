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
    tolS = ops[0]
    tolN = ops[1]
    if len(set(dataSet[:, -1])) == 1:
        return None, leafType(dataSet)
    m, n = dataSet.shape
    S = errType(dataSet)
    bestVar = float("inf")
    bestFeat = 0
    bestValue = 0
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
    leftData, rightData = binarySplitData(dataSet, bestFeat, bestValue)
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


# 判断是否是树函数
def isTree(obj):
    return type(obj).__name__ == 'dict'


# 树坍缩函数
def getMean(tree):
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['right'] + tree['left']) / 2.0


# 在结点中求线性模型
def linearSolve(dataSet):
    m, n = np.shape(dataSet)
    X = np.ones((m, n))
    Y = np.ones((m, 1))
    X[:, 1:n] = dataSet[:, 0:n - 1]
    Y = dataSet[:, -1]
    xTx = X.T.dot(X)
    if np.linalg.det(xTx) == 0:
        raise NameError("xTx 方阵是奇异矩阵，不可逆")
    w = np.linalg.inv(xTx).dot(X.T.dot(Y))
    return w, X, Y


# 叶子结点是线性模型
def modelLeaf(dataSet):
    w, X, Y = linearSolve(dataSet)
    return w


# 计算结点的平方差
def modelErr(dataSet):
    if len(dataSet) == 0:
        return 0
    w, X, Y = linearSolve(dataSet)
    yHat = X.dot(w)
    return sum((yHat - Y) ** 2)


# 回归树节点
def regTreeEval(model, inDat):
    return float(model)


# 模型树节点
def modelTreeEval(model, inDat):
    n = np.shape(inDat)[1]
    X = np.ones((1, n + 1))
    X[:, 1:n + 1] = inDat
    return model.dot(X.T)


# 对一个样本进行预测的函数
def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)


def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = np.zeros((m, 1))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, testData[:, i], modelEval)
    return yHat


# 剪枝函数
def prune(tree, testData):
    # 结点样本数为0。结点坍缩为叶子结点
    if np.shape(testData)[0] == 0:
        return getMean(tree)
    if isTree(tree['left']) or isTree(tree['right']):
        lSet, rSet = binarySplitData(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    # 子结点皆为叶子结点
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binarySplitData(testData, tree['spInd'], tree['spVal'])
        # 剪枝前结点误差
        errorNoMerge = sum((lSet[:, -1] - tree['left']) ** 2) + sum((rSet[:, -1] - tree['right']) ** 2)
        treeMean = (tree['right'] + tree['left']) / 2.0
        errorMerge = sum((testData[:, -1] - treeMean) ** 2)
        if errorMerge < errorNoMerge:
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree


if __name__ == '__main__':
    trainArray = loadData(r'bikeSpeedVsIq_train.txt')
    testArray = loadData(r'bikeSpeedVsIq_test.txt')
    regTree = createTree(trainArray, ops=(1, 20))
    print(regTree)
    modelTree = createTree(trainArray, modelLeaf, modelErr, (1, 20))
    yRegHat=createForeCast(regTree,testArray,regTreeEval)
    yModelHat=createForeCast(modelTree,testArray,modelTreeEval)
    print(yRegHat)
    print(yModelHat)
