import pandas as pd
import numpy as np
import time

#计算时间
def timecount(func):
    def wap(*args,**kwargs):
        t0=time.clock()
        label=func(*args,**kwargs)
        print('process time is {0}s'.format(time.clock()-t0))
        return label
    return wap

#导入数据
def loadData(path):
    train_data=pd.read_csv(path,sep='\t',header=None)
    arraydata=np.array(train_data.iloc[:,:-1])
    labels=np.array(train_data.iloc[:,-1])
    return arraydata,labels

#数据归一化
def normailed(data):
    pro_max=data.max(axis=0)
    pro_min=data.min(axis=0)
    return (data-pro_min)/(pro_max-pro_min),pro_max-pro_min,pro_min


def kNNClassify(newInput,dataSet,labels,k,type='Weighted'):
    row_num=dataSet.shape[0]                 #取得行数
    diff_matrix=np.tile(newInput,(row_num,1))-dataSet
    distance=(diff_matrix**2).sum(axis=1)**0.5  #计算距离
    sort_index=np.argsort(distance,kind='quicksort') #按照距离的大小从小到大排序
    classCount ={}
    if(type=='Weighted'):
        for i in range(k):
            value=labels[sort_index[k]]
            classCount[value]=classCount.get(value,0)+1/(distance[sort_index[k]]**2)
    else:
        for i in range(k):
            value=labels[sort_index[k]]
            classCount[value]=classCount.get(value,0)+1
    return sorted(classCount.items(),key=lambda x :x[1],reverse=True)[0][0]  #返回K个样本中的最多类

@timecount
def datingClassTest():
    interval=0.1
    dataSet, labels=loadData(r'C:\Users\Huang\Documents\WeChat Files\hjw_love\Files\datingTestSet.txt')
    Normdata,ranges,minvals=normailed(dataSet)
    row_num=dataSet.shape[0]
    numTest=int(row_num*interval)
    error=0
    for i in range(numTest):
        classifilerResult=kNNClassify(Normdata[i,:],Normdata[numTest:,:],labels[numTest:],3)
        print("分类结果为：{0}, 原结果为{1}".format(classifilerResult,labels[i]) )
        if(classifilerResult!=labels[i]):  error+=1
    print("总错误率为{0}".format(error/float(numTest)))



def classifyPerson():
    resultList=["不喜欢","喜欢","非常喜欢"]
    gameTimes = float(input("玩游戏时间时间(分钟)"))
    miles = float(input("每年飞行里程"))
    iceCream = float(input("每周消费的冰淇淋公升数"))
    dataSet, labels = loadData(r'C:\Users\Huang\Documents\WeChat Files\hjw_love\Files\datingTestSet.txt')
    Normdata, ranges, minvals=normailed(dataSet)
    classifilerResult = kNNClassify(np.array([gameTimes,miles,iceCream])/ranges, Normdata, labels, 3)
    print("你对这个人的喜欢程度:{0}".format(resultList[classifilerResult-1]))

def main():
    datingClassTest()
    classifyPerson()
if __name__=='__main__':
    main()
