import pandas as pd
import numpy as np

def distEclud(vecA,vecB):
    return np.sum(np.power((vecA-vecB),2))

def biKeans(dataSet,k,distMeas=distEclud):
    m=np.shape(dataSet)[0]
    clusterAssment=np.zeros((m,2))
    centroid0=np.mean(dataSet,axis=0).tolist()
    #质心表
    centList=[centroid0]
    for j in range(m):
        clusterAssment[j,1]=distMeas(np.mat(centroid0),dataSet[j,:])
    while len(centList)<k:
        lowestSSE=float("inf")
        for i in range(len(centList)):
            #pstInCurrCluster=dataSet[np.nonzero(clusterAssment[:,0].A==i)[0],:]
            pstInCurrCluster=dataSet[clusterAssment[:,0]==i]
            #2分裂簇的质心和相关样本的分到簇
            centroidMat,splitClustAss=KMeans(pstInCurrCluster,2,distMeas)
            sseSplit=np.sum(splitClustAss[:,1])
            #sseNotSplit=np.sum(clusterAssment[np.nonzero(clusterAssment[:,0].A!=i)[0],1])
            sseNotSplit=np.sum(clusterAssment[clusterAssment[:,0]!=i][:,1])
            print("sseSplit:{0}   sseNotSplit:{1}".format(sseSplit,sseNotSplit))
            if (sseSplit+sseNotSplit)<lowestSSE:
                #选择最佳分裂结点
                bestCentToSplit=i
                #新质心
                bestNewCents=centroidMat
                bestClustAss=splitClustAss.copy()
                lowestSSE=sseSplit+sseNotSplit
            #bestClustAss[np.nonzero(bestClustAss[:,0].A==1)[0],0]=len(centList)
            #bestClustAss1=bestClustAss
            bestClustAss[bestClustAss[:,0]==1,0]=len(centList)
            bestClustAss[bestClustAss[:,0]==0,0]= bestCentToSplit
            #bestClustAss[bestClustAss[:,0]==1,0]=len(centList)
            #bestClustAss[bestClustAss[:,0]==0,0]=bestCentToSplit
            #bestClustAss[np.nonzero(bestClustAss[:,0].A==0)[0],0]=bestCentToSplit
            print("the bestCentToSplit is :" ,bestCentToSplit)
            print("the len of bestClustAss is " ,len(bestClustAss))
            #最佳分裂簇的质心换掉
            centList[bestCentToSplit]=bestNewCents[0,:]
            #加入新的质点
            centList.append(bestNewCents[1:])
            clusterAssment[clusterAssment[:,0]==bestCentToSplit]=bestClustAss
            #clusterAssment[np.nonzero(clusterAssment[:,0].A==bestCentToSplit)[0],:]=bestClustAss
    return np.mat(centList),clusterAssment



def randPlot(dataSet,k):
    m,n=dataSet.shape
    KCentroid=np.mat(np.zeros((k,n)))
    for i in range(n):
        Max_I=dataSet[:,i].max()
        Min_I=dataSet[:,i].min()
        Range_I=Max_I-Min_I
        KCentroid[:,i]=Min_I+Range_I*np.random.rand(k,1)
    return KCentroid

def KMeans(dataSet,k,distMeas=distEclud):
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
                j_dist=distMeas(dataSet[i,:],KCentroid[j,:])
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
    return np.array(KCentroid),np.array(recordMat)

if __name__=='__main__':
    dataSet=pd.read_table(r'testSet.txt',header=None)
    dataArr=dataSet.values
    centList,MyNewAssments=biKeans(dataArr,3)
    print(centList,MyNewAssments)