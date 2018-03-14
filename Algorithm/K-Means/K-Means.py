import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def loadData(fileName):
    df=pd.read_table(fileName,header=None)
    return df.values

def calDistance(vecA,vecB):
    return np.sum(np.power((vecA-vecB),2))**0.5

def randPlot(dataSet,k):
    m,n=dataSet.shape
    KCentroid=np.mat(np.zeros((k,n)))
    for i in range(n):
        Max_I=dataSet[:,i].max()
        Min_I=dataSet[:,i].min()
        Range_I=Max_I-Min_I
        KCentroid[:,i]=Min_I+Range_I*np.random.rand(k,1)
    return KCentroid

def KMeans(dataSet,k):
    dataSet=np.mat(dataSet)
    m,n=dataSet.shape
    #第一列记录样本的类簇，第二列记录距离类质点的距离
    recordMat=np.mat(np.zeros((m,2)))
    KCentroid=randPlot(dataSet,k)
    stop=True
    while(stop):
        stop=False
        for i in range(m):
            min_dist=float("inf")
            min_index=-1
            for j in range(k):
                j_dist=calDistance(dataSet[i,:],KCentroid[j,:])
                if j_dist<min_dist:
                    min_dist=j_dist
                    min_index=j
            #质心改变
            if recordMat[i,0]!=min_index:
                stop=True
            recordMat[i,:]=min_index,min_dist**2
        for cent in range(k):
            centInClust=dataSet[np.nonzero(recordMat[:,0].A==cent)[0]]
            KCentroid[cent,:]=np.mean(centInClust,axis=0)
    return KCentroid,recordMat

dataSet=loadData('testSet.txt')
A,B=KMeans(dataSet,4)
B=np.array(B)
A=np.array(A)
plt.scatter(x=dataSet[B[:,0]==0][:,0],y=dataSet[B[:,0]==0][:,1],marker='*',c='g')
plt.scatter(x=dataSet[B[:,0]==1][:,0],y=dataSet[B[:,0]==1][:,1],marker='o',c='b')
plt.scatter(x=dataSet[B[:,0]==2][:,0],y=dataSet[B[:,0]==2][:,1],marker='s',c='r')
plt.scatter(x=dataSet[B[:,0]==3][:,0],y=dataSet[B[:,0]==3][:,1],marker='*',c='y')
plt.scatter(x=A[:,0],y=A[:,1],marker='^',c='m',s=90)
plt.show()

    
    

    
