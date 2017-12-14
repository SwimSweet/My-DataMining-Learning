import pandas as pd
import numpy as np
from functools import reduce

def loadData():
    #train_data=pd.read_csv(path)
    train_data=np.array([[1.0, 0.9], [1.0, 1.0], [0.1, 0.2], [0.0, 0.1]])
    labels=['A', 'A', 'B', 'B']
    #feature_data=np.array(train_data.iloc[:,0:-1])
    #labels=np.array(train_data.iloc[:,-1])
    dataSet,rangeMax,rangeMin=Normailed(train_data)
    return dataSet,labels,rangeMax,rangeMin

def Normailed(data):
    pro_max=data.max(axis=0)
    pro_min=data.min(axis=0)
    return (data-pro_min)/(pro_max-pro_min),pro_max,pro_min

def kNNClassify(newInput,dataSet,labels,rangeMax,rangeMin,k):
    norm_input=(newInput-rangeMin)/(rangeMax-rangeMin)
    row_num=dataSet.shape[0]
    diff_matrix=np.tile(norm_input,(row_num,1))-dataSet
    distance=(diff_matrix**2).sum(axis=1)**0.5
    sort_index=np.argsort(distance,kind='quicksort')
    classCount ={}
    for i in range(k):
        value=labels[sort_index[k]]
        classCount[value]=classCount.get(value,0)+1
    return sorted(classCount.items(),key=lambda x :x[1],reverse=True)[0][0]



dataSet,labels,rangeMax,rangeMin=loadData()
test_X=np.array([1.2, 1.0])
k=3
output_label=kNNClassify(test_X,dataSet,labels,rangeMax,rangeMin,k)
print(test_X, output_label )