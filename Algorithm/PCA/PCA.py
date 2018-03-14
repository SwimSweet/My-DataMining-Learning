import numpy as np
import pandas as pd

def loadData():
    data=pd.read_table(r'testSet.txt').values
    return data

def pca(dataSet,topFeat):
    # 求均值向量
    meanvector=np.mean(dataSet,axis=0)
    # 中心化后数据
    dataSet=dataSet-meanvector
    # 协方差矩阵
    covMat=np.cov(dataSet,rowvar=0)
    # 求特征值，特征向量
    eigvals,eigvec=np.linalg.eig(covMat)
    # 对特征值进行排列，得到由小到大元素的索引
    eigValInd = np.argsort(eigvals)  # sort, sort goes smallest to largest
    # 得到最大的topFeat个特征值
    eigValInd = eigValInd[:-(topFeat + 1):-1]
    # 得到最大的topFeat个特征值对应的特征向量
    redEigVects = eigvec[:, eigValInd]
    # 降维后的数据
    lowDDataMat = dataSet .dot(redEigVects)
    # 用投影后的数据和投影矩阵重构原始样本
    reconMat = (lowDDataMat.dot(redEigVects.T)) + meanvector
    return lowDDataMat,reconMat

def main():
    dataSet = loadData()
    lowDMat, reconMat = pca(dataSet, 1)
    print(lowDMat)
    print(reconMat)

if __name__=='__main__':
    main()


