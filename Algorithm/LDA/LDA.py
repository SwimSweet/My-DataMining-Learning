__author__ = 'Huang'

import numpy as np
import pandas as pd


# 将二分类的特征降至一维
def calulate_w(dataArray, labelArray):
    dataArray0 = dataArray[labelArray == 0]
    dataArray1 = dataArray[labelArray == 1]
    miu0 = np.mean(dataArray0, axis=0)
    miu1 = np.mean(dataArray1, axis=0)
    mean_vec = np.vstack((miu0, miu1))
    # 散内矩阵
    # 散内矩阵刚好是每个类中少除以类样本数-1的协方差矩阵
    Sw = np.cov(dataArray0.T) * (len(dataArray0) - 1) + np.cov(dataArray1.T) * (len(dataArray1) - 1)
    # 散间矩阵
    diffMeanVec = (miu0 - miu1).reshape(4, 1)
    invSw = np.linalg.inv(Sw)
    W = invSw.dot(diffMeanVec)
    return W


# 将N分类降至N-1维
def lda_muliti_class(dataArray, labelkclass):
    m, n = dataArray.shape
    classset = set(labelkclass)
    meanallvec = np.mean(dataArray, axis=0).reshape(n, 1)
    mean_vecs = []
    dataNumArray = []
    for i in classset:
        dataArray_i = dataArray[labelkclass == i]
        mean_vec = np.mean(dataArray_i, axis=0)
        dataNumArray.append(len(dataArray_i))
        mean_vecs.append(mean_vec)
        print("{0}类样本的样本均值为{1}".format(i, mean_vec))
    Sw = np.zeros((n, n))
    # 散内矩阵刚好是每个类中少除以类样本数-1的协方差矩阵
    for i in classset:
        Sw += np.cov(dataArray[labelkclass == i].T) * (dataNumArray[i] - 1)
    Sb = np.zeros((n, n))
    for i in range(len(mean_vecs)):
        mean_vec = mean_vecs[i].reshape(n, 1)
        Sb += dataNumArray[i] * (mean_vec - meanallvec).dot((mean_vec - meanallvec).T)
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
    for i in range(len(eig_vals)):
        eigvec_sc = eig_vecs[:, i].reshape(n, 1)
        print('\n特征向量 {}: \n{}'.format(i + 1, eigvec_sc.real))
        print('特征值 {:}: {:.2e}'.format(i + 1, eig_vals[i].real))
    # 将特征值和对应的特征向量对应起来
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
    # 按照特征值的大小排序
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
    print('\n按照特征值大小降序排列的特征值特征向量:')
    for i in eig_pairs:
        print(i[0], i[1])
    # 我们通过特征值的比例来体现方差的分布：
    print('\n特征值比例:')
    eigv_sum = sum(eig_vals)
    for i, j in enumerate(eig_pairs):
        print('特征值占比 {0:}: {1:.2%}'.format(i + 1, (j[0] / eigv_sum)))
    W = []
    for i in range(len(classset) - 1):
        W.append(eig_pairs[i][1])
    return np.array(W).T


def main():
    dataSet = pd.read_csv(r'iris.csv')
    dataSet.columns = [0, 1, 2, 3, 4]
    dataArray = dataSet.loc[:, :3].values
    labelclass = dataSet.loc[:, 4].values
    # 特征向量矩阵
    W = lda_muliti_class(dataArray, labelclass)
    # 将样本投影到新的样本空间
    print("\n新的样本空间:")
    X_laded=dataArray.dot(W)
    print(X_laded)
if __name__ == '__main__':
    main()


