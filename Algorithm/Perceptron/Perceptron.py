__author__ = 'Huang'

import numpy as np
import pandas as pd

def loadData(fileName):
    dataSet=pd.read_table(fileName,header=None).values
    dataArr=dataSet[:,:-1]
    labelArr=dataSet[:,-1]
    return dataArr,labelArr


#感知机原始形式求超平面参数,损失函数选误分类到超平面的距离
def calPar(dataArr,labelArr):
    dataMat=np.mat(dataArr);labelMat=np.mat(labelArr).transpose()
    m,n=np.shape(dataArr)
    # 学习率
    alpha=1
    w=np.mat(np.zeros((n,1)))
    b=0
    ErrorCount=1
    while(ErrorCount>0):
        ErrorCount=0
        for i in range(m):
            if labelMat[i]*(w.T*dataMat[i,:].T+b)<=0:
                w+=np.multiply(alpha*labelMat[i],dataMat[i,:].T)
                b+=alpha*labelMat[i]
                ErrorCount+=1
    return w,b

#感知机的对偶形式
def dualCalPar(dataArr,labelArr):
    dataMat=np.mat(dataArr)
    labelMat=np.mat(labelArr).transpose()
    m,n=np.shape(dataArr)
    #求训练集的Gram矩阵
    gramMat=dataMat*dataMat.T
    alpha=np.mat(np.zeros((m,1)))
    b=0
    eta=1
    errCount=1
    while errCount>0:
        errCount=0
        for i in range(m):
            if labelMat[i]*(np.multiply(alpha,labelMat).T*gramMat[:,i]+alpha.T*labelMat)<=0:
                errCount+=1
                alpha[i]+=eta
                b+=eta*labelMat[i]
    w=dataMat.T*np.multiply(alpha,labelMat)
    return w,b


def main():
    dataArr,labelArr=loadData(r'test.txt')
    w,b=calPar(dataArr,labelArr)
    w1,b1=dualCalPar(dataArr,dataArr)
    print(w)
    print(b)
    print(w1)
    print(b1)

if __name__=='__main__':
    main()