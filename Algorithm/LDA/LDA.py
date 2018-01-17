__author__ = 'Huang'


import numpy as np
import pandas as pd
def lda (weight,dataArray0,dataArray1,labelArray):
    miu0 = np.mean(dataArray0,axis=0)
    miu1 = np.mean(dataArray1,axis=0)
    #散内矩阵
    Sw=np.cov(dataArray0.T)+np.cov(dataArray1.T)
    #散间矩阵
    Sb=np.cov(np.vstack((miu0, miu1)))
    mainArray=np.linalg.inv(Sw).dot(Sb)
    eig_vals, eig_vecs = np.linalg.eig(mainArray)
    for i in range(len(eig_vals)):
        eigvec_sc=eig_vecs[:,i].reshape(4, 1)
        print('\nEigenvector {}: \n{}'.format(i + 1, eigvec_sc.real))
        print('Eigenvalue {:}: {:.2e}'.format(i + 1, eig_vals[i].real))


dataSet=pd.read_csv(r'iris.csv')
dataSet.columns=[0,1,2,3,4]
print(dataSet[dataSet[4]==0])