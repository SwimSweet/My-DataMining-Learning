import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

#Load训练数据
def loadData():
    dataSet=pd.read_table(r'testSet.txt',header=None)
    dataArr=dataSet.loc[:,0:1]
    labelArr=dataSet.loc[:,2]
    dataArr[2]=[1 for x in range(len(dataArr))]
    return np.array(dataArr),np.array(labelArr)

def sigmoid(z):
    return 1/(1+np.exp(-z))


#计算求最优化参数的算法运行时间
def timeCount(func):
    def wrapper(*args,**kwargs):
        start=time.clock()
        weight=func(*args,**kwargs)
        end=time.clock()
        print("梯度下降求最佳回归参数消耗时间 : {0}s".format(start-end))
        return weight
    return wrapper

#梯度下降求最优化参数
@timeCount
def gradDecline(dataArr,labelArray):
    dataArray=dataArr
    labelArray=labelArray.reshape((len(labelArray),1))
    m,n=np.shape(dataArray)
    alpha=0.001
    maxCtyles=500
    weights=np.ones((n,1))
    for k in range(maxCtyles):
        m=sigmoid(dataArray.dot(weights))
        error=(m-labelArray)
        weights=weights-alpha*dataArray.T.dot(error)
    return weights


#随机梯度下降求最优化参数
@timeCount
def stocGradAScent0(dataArr,labelArray ,iterNum=200):
    m,n=dataArr.shape
    weight=np.ones(n)
    alpha=0.01
    for i in range(iterNum):
        dataIndex = list(range(m))
        for j in range(m):
            index=int(np.random.uniform(0,len(dataIndex)))
            h=sigmoid(sum(dataArr[index]*weight))
            error=h-labelArray[index]
            weight=weight-alpha*error*dataArr[index]
            del (dataIndex[index])
    return weight



#画出决策边界
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

def classify(X,weights):
    prob=sigmoid(X.dot(weights))
    if prob>0.5:
        return 1
    else:
        return 0

#预测病马的死亡率
def colicTest():
    dataTrain=pd.read_table(r'colicTrain.txt',header=None)
    dataTest=pd.read_table(r'colicTest.txt',header=None )
    trainSet=np.array(dataTrain.iloc[:,:-1])
    trainLabel=np.array(dataTrain.iloc[:,-1])
    testSet=np.array(dataTest.iloc[:,:-1])
    testLabel=np.array(dataTest.iloc[:,-1])
    trainWeights=stocGradAScent0(trainSet,trainLabel,500)
    errorCount=0;numTestVec=len(testLabel)
    for i in range(numTestVec):
        if(classify(testSet[i],trainWeights)!=testLabel[i]):
            errorCount+=1
    errorRate=float(errorCount)/numTestVec
    print("the error rate of this test is : {0}".format(errorRate))
    return errorRate

def multiTest():
    numTestNum=10;errorSum=0
    for i in range(10):
        errorSum+=colicTest()
    print("after {0} iterations the average error rate is : {1}".format(numTestNum,float(errorSum)/numTestNum))

# dataArr,labelArr=loadData()
# Weights=gradDecline(dataArr,labelArr)
# print(Weights)
# weight=stocGradAScent0(dataArr,labelArr)
# print(weight)
# plotBestFit(Weights)
# plotBestFit(weight)

multiTest()