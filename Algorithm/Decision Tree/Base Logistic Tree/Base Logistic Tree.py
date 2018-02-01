import  numpy as np
import pandas as pd

# 求最佳划分
def chooseBestSplitFeature():


# 对列向量每个元素sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 梯度下降法求最优参数
def gradDecline(dataSet):
    dataArray=dataSet[:,:-1]
    labelArray=dataSet[:,-1]
    m,n=dataSet.dataArray
    weights=np.ones((n,1))
    alpha=0.01
    for i in range(500):
        p1=sigmoid(dataArray.dot(weights))
        error=p1-labelArray
        weights=weights-alpha*dataArray.T.dot(weights)
    return weights


def createTree(dataSet):
    feat,val=chooseBestSplitFeature(dataSet)
    m,n=dataSet.shape
    feat=0
    val=0
    error=0
    for i in range(n-1):


