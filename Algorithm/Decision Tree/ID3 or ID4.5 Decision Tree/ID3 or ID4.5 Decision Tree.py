import pickle
from collections import Counter
from math import log
import numpy as np


# 计算信息熵
def calcShannonEnt(dataSet):
    numberEntries = len(dataSet)
    labelCount = {}
    for dataline in dataSet:
        currentlabel = dataline[-1]
        labelCount[currentlabel] = labelCount.get(currentlabel, 0) + 1
    shannonEnt = 0
    for key in labelCount.keys():
        prob = labelCount[key] / numberEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


# 计算属性axis的固有值
def InvEnt(dataSet, axis):
    invent = 0
    valueSet = set(dataSet[:, axis])
    for value in valueSet:
        prob = len(dataSet[dataSet[:, axis] == value]) / len(dataSet)
        invent -= prob * log(prob, 2)
    return invent


def createDataSet():
    dataSet = np.array([[1, 1, 'yes'], [1, 1, 'yes'], [0, 1, 'no'], [0, 0, 'no'], [1, 0, 'no']])
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


# 得到以第axis轴值等于value进行分裂得到的数据集
def splitDataSet(dataSet, axis, value):
    return np.delete(dataSet[dataSet[:, axis] == value], axis, axis=1)


# 选择最佳分裂点，以信息增益或信息增益率方法选择最优分裂属性
# pattrn: 0:信息增益 1:信息增益率
def chooseBestSplitFeature(dataSet, pattern):
    featureNum = len(dataSet[0]) - 1
    baseEnt = calcShannonEnt(dataSet)
    bestInfo = 0
    bestFeature = -1
    for axis in range(featureNum):
        invEnt = InvEnt(dataSet, axis)
        info = 0
        valueSet = set([example[axis] for example in dataSet])
        for value in valueSet:
            valueList = splitDataSet(dataSet, axis, value)
            prob = len(valueList) / len(dataSet)
            info += prob * calcShannonEnt(valueList)
        #信息增益
        gainInfo = baseEnt - info
        # 信息增益
        if pattern == 0:
            if gainInfo > bestInfo:
                bestInfo = gainInfo
                bestFeature = axis
        # 信息增益率
        if pattern == 1:
            gainRatio=gainInfo / invEnt
            if (gainRatio> bestInfo):
                bestInfo = gainRatio
                bestFeature = axis
    return bestFeature



# 递归建树
def crateTree(dataSet, labels,pattern):
    classList = [temp[-1] for temp in dataSet]
    if (classList.count(classList[0]) == len(classList)):  # 同一节点的所有记录都属于同一类，结束
        return classList[0]
    if (len(dataSet[0]) == 1):  # 所有特征消耗完毕，结束
        return sorted(dict(Counter(classList)).items(), key=lambda x: x[1], reverse=True)[0][1]
    bestFeature = chooseBestSplitFeature(dataSet,pattern)
    bestFeatlabel = labels[bestFeature]
    myTree = {bestFeatlabel: {}}
    newlabels = labels[:]
    del (newlabels[bestFeature])  # 删除列标签的label
    featValue = [temp[bestFeature] for temp in dataSet]
    featValueSet = set(featValue)
    for value in featValueSet:
        valueDataSet = splitDataSet(dataSet, bestFeature, value)
        myTree[bestFeatlabel][value] = crateTree(valueDataSet, newlabels,pattern)
    return myTree

def storeTree(inputTree, filename):
    import pickle
    inputTree = dict(inputTree)
    fw = open(filename, 'wb')
    pickle.dump(dict(inputTree), fw)
    fw.close()


def grabTree(filename):
    fr = open(filename, 'rb')
    return pickle.load(fr)


def createLenses(filepath):
    fr = open(filepath)
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = crateTree(lenses, lensesLabels)
    print(lensesTree)


def main():
    dataSet, labels = createDataSet()
    myTree = crateTree(dataSet, labels,1)
    storeTree(myTree, r'myTree.txt')
    print(grabTree(r'myTree.txt'))


if __name__ == '__main__':
    main()
