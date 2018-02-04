import  numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import OneHotEncoder

# 梯度下降法求最优参数
def gradDecline(dataSet):
    dataArray=dataSet[:,:-1]
    labelArray=dataSet[:,-1]
    m,n=dataArray.shape
    labelArray=labelArray.reshape((m,1))
    weights=np.ones((n,1))
    alpha=0.01
    # 数据集第一项对应的是参数常数项
    for i in range(500):
        p1=sigmoid(dataArray.dot(weights))
        error=p1-labelArray
        weights=weights-alpha*dataArray.T.dot(error)
    return weights

def splitDataSet(dataSet,fea,val,sep):
    if sep==True:
        return dataSet[dataSet[:,fea]==val]
    else :
        dataSet1=dataSet[dataSet[:,fea]>val]
        dataSet2=dataSet[dataSet[:,fea]<=val]
        return dataSet1,dataSet2


# 求最佳划分
def chooseBestSplitFeature(dataSet,enc,seplist,feaList):
    baseError=logisticError(dataSet,enc)
    bestError=1
    bestFea=0
    bestVal=0
    m,n=dataSet.shape
    for fea in  range(n):
        # 离散属性
        feaError=0
        if  fea in seplist:
            if fea not in feaList:
                continue
            feaValueSet=set(dataSet[:,fea])
            for value in feaValueSet:
                feaValdataSet=splitDataSet(dataSet,fea,value,True)
                feaError+=logisticError(feaValdataSet,enc)
            feaMeanError=feaError/len(feaValueSet)
            if feaMeanError<bestError:
                bestError=feaMeanError
                bestFea=fea
        # 连续属性
        else :
            valueList=sorted(list(set(dataSet[:,fea])),reverse=False)
            if len(valueList)>=2:
                valueList=[(valueList[i+1]+valueList[i])/2.0 for i in range(len(valueList)-1)]
            for value in valueList:
                feaValdataSet1,feaValdataSet2=splitDataSet(dataSet,fea,value,False)
                feaError+=logisticError(feaValdataSet1,enc)
                feaError += logisticError(feaValdataSet2, enc)
                feaMeanError=feaError/2.0
                if feaMeanError < bestError:
                    bestError = feaMeanError
                    bestFea = fea
                    bestVal=value
    return bestFea,bestVal




def logisticError(dataSet,enc):
    transFormDataSet=enc.transform(dataSet[:,:6]).toarray()
    m,n=dataSet.shape
    transFormDataSet=np.hstack((np.ones((m,1)),transFormDataSet,dataSet[:,6:]))
    weights=gradDecline(transFormDataSet)
    error=0
    for i in range(m):
        if classfy(transFormDataSet[i,:-1],weights)!=transFormDataSet[i,-1]:
            error+=1
    return error/m

def classfy(X,weights):
    if  sigmoid(X.dot(weights))>0.5:
        return 1
    else:
        return 0

# 对列向量每个元素sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def loadData():
    dataSet = pd.read_table(r'watermelon.txt', index_col=0, sep=',')
    for i in range(6):
        feaset = set(dataSet.iloc[:, i])
        feaToNum = {}
        k = 0
        for fea in feaset:
            feaToNum[fea] = k
            k += 1
        dataSet.iloc[:, i] = dataSet.iloc[:, i].map(lambda x: feaToNum[x])
    labelset=set(dataSet.iloc[:, -1])
    labelToNum={}
    k=0
    for label in labelset:
        feaToNum[label] = k
        k += 1
    dataSet.iloc[:, -1] = dataSet.iloc[:, -1].map(lambda x: feaToNum[x])
    dataSet=dataSet.values
    print(dataSet)
    # OneHotEncoder
    enc=OneHotEncoder()
    enc.fit(dataSet[:,:6])
    return dataSet,enc



def createTree(dataSet,enc,sepList,feaList,labels,resultsLabel):
    classList=set(dataSet[:,-1])
    # 所有样本同属于一类
    if len(set(classList))==1:
        return resultsLabel[int(dataSet[0][-1])]
    feat,val=chooseBestSplitFeature(dataSet,enc,sepList,feaList)
    m,n=dataSet.shape
    bestFeatlabel = labels[feat]
    myTree={bestFeatlabel: {}}
    newFeaList=feaList[:]
    if feat in sepList:
        newFeaList.remove(feat)
        featValue =set( [temp[feat] for temp in dataSet])
        for value in featValue:
            valueDataSet=splitDataSet(dataSet,feat,value,True)
            myTree[bestFeatlabel][value]=createTree(valueDataSet,enc,sepList,newFeaList,labels,resultsLabel)
    else :
        myTree['spInd'] = feat
        myTree['spVal'] = val
        lSet, rSet = splitDataSet(dataSet, feat, val,False)
        myTree[bestFeatlabel]['left']=createTree(lSet,feat,val,sepList,feaList,labels,resultsLabel)
        myTree[bestFeatlabel]['right']=createTree(rSet,feat,val,sepList,feaList,labels,resultsLabel)
    return myTree


if __name__=='__main__':
    # 经过数字转换的数据集和ont-hot转换模型
    dataSet,enc=loadData()
    # 标记哪些属性是离散属性
    sepList=[0,1,2,3,4,5]
    labels=['色泽','根蒂','敲声','纹理','脐部','触感','密度','含糖率']
    resultsLabel=['烂瓜','好瓜']
    waterMelonTree=createTree(dataSet,enc,sepList,sepList,labels,resultsLabel)
    print(waterMelonTree)