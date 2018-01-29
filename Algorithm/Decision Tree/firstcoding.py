import numpy as np
import pandas as pd
from math import log
from collections import Counter


def calcShannonEnt(dataset):
    count = len(dataset)
    shannoEnt = 0
    for label in set(dataset[:, -1]):
        prob = len(dataset[dataset[:, -1] == label])/count
        shannoEnt -= prob*log(prob, 2)
    return shannoEnt


# 数据集在属性axis等于value的样本子集
def splitnode(dataset,axis,value):
    return dataset[dataset[:,axis]==value]

#选择最佳分裂特征
def choocebestfeature(dataset):
    entfather = calcShannonEnt(dataset)
    featureNum=len(dataset[0])-1
    bestfeature=-1;gainent=0
    for i in range(featureNum):
        featureset=set(featureNum[:,i])
        featureent=0
        for value in featureset:
            valuedata=splitnode(dataset,i,value)
            featureent += len(valuedata)/len(dataset)*calcShannonEnt(valuedata)
        if entfather-featureent >gainent:
            bestfeature=i
            gainent=entfather-featureent
    return bestfeature

#递归建树
def crateTree(dataSet,labels):
    classList = [temp[-1] for temp in dataSet]
    if (classList.count(classList[0]) == len(classList)):  # 同一节点的所有记录都属于同一类，结束
        return classList[0]
    if(len(dataSet[0])==1):
        return  sorted(dict(Counter(classList)).items(),key=lambda x:x[1],reverse=True)[0][1]
    bestFeature = choocebestfeature(dataSet)

def majorclass(classarray):




def main():
    dataSet = np.array([[1, 1, 'yes'], [1, 1, 'yes'], [0, 1, 'no'], [0, 0, 'no'], [1, 0, 'no']])
    print(calcShannonEnt(dataSet))

if __name__=='__main__':
    main()