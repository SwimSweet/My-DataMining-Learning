import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


def loadData():
    dataSet=pd.read_table(r'testSet.txt',header=None)
    dataArr=dataSet.loc[:,0:1]
    labelArr=dataSet.loc[:,2]
    dataArr[2]=[1 for x in range(len(dataArr))]
    return dataArr,labelArr

def sigmoid(z):
    return 1/(1+np.exp(-z))



def timeCount(func):
    def wrapper(*args,**kwargs):
        start=time.clock()
        weight=func(*args,**kwargs)
        end=time.clock()
        print("梯度下降求最佳回归参数消耗时间 : {0}s".format(start-end))
        return weight
    return wrapper

@timeCount
def gradDecline(dataArr,labelArray):
    dataArray=np.array(dataArr)
    labelArray=np.array(labelArray).reshape((len(labelArray),1))
    m,n=np.shape(dataArray)
    alpha=0.001
    maxCtyles=500
    weights=np.ones((n,1))
    for k in range(maxCtyles):
        m=sigmoid(dataArray.dot(weights))
        error=(m-labelArray)
        weights=weights-alpha*dataArray.T.dot(error)
    return weights



def plotBestFit(Weights):
    dataMat,labelMat=loadData()
    dataArr=np.array(dataMat)
    n=dataArr.shape[0]
    xcord1=[];ycord1=[]
    xcord2=[];ycord2=[]
    for i in range(n):
        if  int(labelMat[i]==1):
            xcord1.append(dataArr[i,0])
            ycord1.append(dataArr[i,1])
        else:
            xcord2.append(dataArr[i,0])
            ycord2.append(dataArr[i,1])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x=np.arange(-3,3,0.1)
    y=(-Weights[2]-Weights[0]*x)/Weights[1]
    ax.plot(x,y)
    plt.xlabel('X1');plt.ylabel('X2')
    plt.show()



dataArr,labelArr=loadData()
Weights=gradDecline(dataArr,labelArr)
plotBestFit(Weights)