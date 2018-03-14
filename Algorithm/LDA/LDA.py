__author__ = 'Huang'

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
matplotlib.rcParams['axes.unicode_minus'] = False


# 将二分类的特征降至一维
def calulate_w(dataArray, labelArray):
    m, n = dataArray.shape
    labelset = list(set(labelArray))
    dataArray0 = dataArray[labelArray == labelset[0]]
    dataArray1 = dataArray[labelArray == labelset[1]]
    miu0 = np.mean(dataArray0, axis=0)
    miu1 = np.mean(dataArray1, axis=0)
    # 散内矩阵
    # 散内矩阵刚好是每个类中少除以类样本数-1的协方差矩阵
    Sw = np.cov(dataArray0.T) * (len(dataArray0) - 1) + np.cov(dataArray1.T) * (len(dataArray1) - 1)
    # 散间矩阵
    diffMeanVec = (miu0 - miu1).reshape(len(miu0), 1)
    invSw = np.linalg.inv(Sw)
    W = invSw.dot(diffMeanVec)
    return W


# 将N分类降至N-1维
def lda_muliti_class(dataArray, labelkclass):
    m, n = dataArray.shape
    classset = set(labelkclass)
    # 所有样本的均值向量
    meanallvec = np.mean(dataArray, axis=0).reshape(n, 1)
    mean_vecs = []
    dataNumArray = []
    # 各个类别的均值向量
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
    # 求类间散度矩阵
    for i in range(len(mean_vecs)):
        mean_vec = mean_vecs[i].reshape(n, 1)
        Sb += dataNumArray[i] * (mean_vec - meanallvec).dot((mean_vec - meanallvec).T)
    # 求特征值特征向量
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


# 样例用的数据集刚好可以降到二维，可以用可视化查看数据分布
def plot_step_lda(x_lda, labelkclass):
    ax = plt.subplot(111)
    label_dict = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    for label, marker, color in zip(
            range(3), ('^', 's', 'o'), ('blue', 'red', 'green')):
        plt.scatter(x=x_lda[:, 0].real[labelkclass == label],
                    y=x_lda[:, 1].real[labelkclass == label],
                    marker=marker,
                    color=color,
                    alpha=0.5,
                    label=label_dict[label]
                    )
    plt.xlabel('LD1')
    plt.ylabel('LD2')
    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title('LDA: Iris projection onto the first 2 linear discriminants')
    # hide axis ticks
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                    labelbottom="on", left="off", right="off", labelleft="on")
    # 移除图的边界
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    # 显示网格
    plt.grid()
    plt.tight_layout
    plt.show()


def plot_melon_lda(x, label, w):
    ax = plt.subplot(111)
    label_dict = {0: '是', 1: '否'}
    for i, marker, color in zip(range(2), ['s', 'o'], ['g', 'r']):
        plt.scatter(x=x[:, 0][label == label_dict[i]], y=x[:, 1][label == label_dict[i]], marker=marker,
                    color=color,
                    alpha=0.5,
                    label=label_dict[i])
    plt.xlim(-0.2, 1)
    plt.ylim(-0.5, 0.7)
    p0_x0 = -x[:, 0].max()
    p0_x1 = (w[1, 0] / w[0, 0]) * p0_x0
    p1_x0 = x[:, 0].max()
    p1_x1 = (w[1, 0] / w[0, 0]) * p1_x0
    plt.title('watermelon_3a - LDA')
    plt.xlabel('density')
    plt.ylabel('ratio_sugar')
    # plt.scatter(x[y ==label_dict[0], 0], x[y == 0, 1], marker='o', color='k', s=10, label='bad')
    # plt.scatter(x[y == 1, 0], x[y == 1, 1], marker='o', color='g', s=10, label='good')
    # plt.legend(loc='upper right')
    plt.plot([p0_x0, p1_x0], [p0_x1, p1_x1])
    plt.xlabel("密度")
    plt.ylabel("含糖  率")
    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                    labelbottom="on", left="off", right="off", labelleft="on")
    # 移除图的边界
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.xlim(-0.2, 1)
    plt.ylim(-0.5, 0.7)
    # 显示网格
    plt.grid()
    plt.tight_layout
    plt.show()


def main():
    dataSet = pd.read_csv(r'iris.csv')
    dataarray = dataSet.iloc[:, :4].values
    labelclass = dataSet.iloc[:, 4].values
    # 特征向量矩阵
    w = lda_muliti_class(dataarray, labelclass)
    # 将样本投影到新的样本空间
    print("\n新的样本空间:")
    x_laded = dataarray.dot(w)
    print(x_laded)
    plot_step_lda(x_laded, labelclass)


def watermelon(dataarray):
    x = dataarray[:, -3:-1].astype("float32")
    y = dataarray[:, -1]
    w = calulate_w(x, y)
    return w


if __name__ == '__main__':
    main()
    # datamelon = pd.read_table(r'watermelon.txt', sep=',', index_col=0).values
    # x = datamelon[:, -3:-1].astype("float32")
    # y = datamelon[:, -1]
    # w = watermelon(datamelon)
    # plot_melon_lda(x,y,w)
