__author__ = 'Huang'


import numpy as np
import pandas as pd
def lda (dataArray0,dataArray1):
    miu0 = np.mean(dataArray0,axis=0)
    miu1 = np.mean(dataArray1,axis=0)
    mean_vec=np.vstack((miu0,miu1))
    #散内矩阵
    #散内矩阵刚好是每个类中少除以类样本数-1的协方差矩阵
    Sw=np.cov(dataArray0.T)*(len(dataArray0)-1)+np.cov(dataArray1.T)*(len(dataArray1)-1)
    #散间矩阵
    Sb=np.cov(mean_vec.T,bias=False)
    diffMeanVec=(miu0-miu1).reshape(4,1)
    Sbb=diffMeanVec.dot(diffMeanVec.T)
    mainArray=np.linalg.inv(Sw).dot(Sb)
    eig_vals, eig_vecs = np.linalg.eig(mainArray)
    for i in range(len(eig_vals)):
        eigvec_sc=eig_vecs[:,i].reshape(4, 1)
        print('\nEigenvector {}: \n{}'.format(i + 1, eigvec_sc.real))
        print('Eigenvalue {:}: {:.2e}'.format(i + 1, eig_vals[i].real))


dataSet=pd.read_csv(r'iris.csv')
dataSet.columns=[0,1,2,3,4]
dataArray0=dataSet[dataSet[4]==0].loc[:,:3].values
dataArray1=dataSet[dataSet[4]==1].loc[:,:3].values
lda(dataArray0,dataArray1)
