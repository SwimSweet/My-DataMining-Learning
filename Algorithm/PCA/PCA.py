import numpy as np
import pandas as pd

def loadData():
    data=pd.read_table(r'testSet.txt').values
    return data

def pca(dataSet,topFeat):
    meanvector=np.mean(dataSet,axis=0)
    dataSet=dataSet-meanvector
    covMat=np.cov(dataSet,rowvar=0)
    covMat1=dataSet.T.dot(dataSet)
    eigvals,eigvec=np.linalg.eig(covMat)
    eigValInd = np.argsort(eigvals)  # sort, sort goes smallest to largest
    eigValInd = eigValInd[:-(topFeat + 1):-1]  # cut off unwanted dimensions
    redEigVects = eigvec[:, eigValInd]  # reorganize eig vects largest to smallest
    lowDDataMat = dataSet .dot(redEigVects)  # transform data into new dimensions
    reconMat = (lowDDataMat.dot(redEigVects.T)) + meanvector
    return lowDDataMat,reconMat

dataSet=loadData()
lowDMat,reconMat=pca(dataSet,1)
print(lowDMat)
print(reconMat)
