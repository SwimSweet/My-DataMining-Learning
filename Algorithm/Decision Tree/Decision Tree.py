from math import log
from collections import Counter
import pickle

#计算信息熵
def calcShannonEnt(dataSet):
    numberEntries=len(dataSet)
    labelCount={}
    for dataline in dataSet:
        currentlabel=dataline[-1]
        labelCount[currentlabel]=labelCount.get(currentlabel,0)+1
    shannonEnt=0
    for key in labelCount.keys():
        prob=labelCount[key]/numberEntries
        shannonEnt-=prob*log(prob,2)
    return shannonEnt

def calGini(dataSet):
    numberEntries=len(dataSet)
    labelCount={}
    for labelVec in dataSet[:,-1]:
        labelCount[labelVec]=labelCount.get(labelVec,0)+1
    gini=1
    for key in labelCount.keys():
        prob=labelCount[key]/numberEntries
        gini-=prob**2
    return gini

def createDataSet():
    dataSet=[[1,1,'yes'],[1,1,'yes'],[0,1,'no'],[0,0,'no'],[1,0,'no']]
    labels=['no surfacing','flippers']
    return dataSet,labels

# 得到以第axis轴值等于value进行分裂得到的数据集
def splitDataSet(dataSet,axis,value):
    retDataSet=[]
    for line in dataSet:
        if line[axis]==value:
            templist=line[:axis]
            templist.extend(line[axis+1:])
            retDataSet.append(templist)
    return retDataSet

#选择最佳分裂点
def chooseBestSplitFeature(dataSet):
    featureNum=len(dataSet[0])-1
    baseEnt=calcShannonEnt(dataSet)
    bestInfo=0; bestFeature=-1
    for axis in range(featureNum):
        info=0
        valueSet=set(dataSet[axis])
        for value in valueSet:
            valueList=splitDataSet(dataSet,axis,value)
            prob=len(valueList)/len(dataSet)
            info+=prob*calcShannonEnt(valueList)
        if(baseEnt-info>bestInfo):
            bestInfo=baseEnt-info
            bestFeature=axis
    return bestFeature

#递归建树
def crateTree(dataSet,labels):
    classList=[ temp[-1] for temp in dataSet]
    if(classList.count(classList[0])==len(classList)):  #同一节点的所有记录都属于同一类，结束
        return classList[0]
    if(len(dataSet[0])==1):                             #所有特征消耗完毕，结束
        return MajorCnt(classList)
    bestFeature=chooseBestSplitFeature(dataSet)
    bestFeatlabel=labels[bestFeature]
    myTree={bestFeatlabel:{}}
    newlabels = labels[:]
    del(newlabels[bestFeature])    #删除列标签的label
    featValue=[temp[bestFeature] for temp in dataSet]
    featValueSet=set(featValue)
    for value in featValueSet:
        valueDataSet=splitDataSet(dataSet,bestFeature,value)
        myTree[bestFeatlabel][value]=crateTree(valueDataSet,newlabels)
    return myTree

#计算出现次数最多的类
def MajorCnt(classList):
    return Counter(classList).most_common(1)[0][0]


def classify(inputTree, featLabels, testVec):
   firstStr=list(inputTree.keys())[0]
   featIndex=featLabels.index(firstStr)
   seconddict=inputTree[firstStr]
   result=testVec[featIndex]
   valueofFeat=seconddict[result]
   if isinstance(valueofFeat,dict):
       return classify(valueofFeat,featLabels,testVec)
   else:
       return valueofFeat

def storeTree(inputTree,filename):
    import pickle
    inputTree=dict(inputTree)
    fw=open(filename,'wb')
    pickle.dump(dict(inputTree),fw)
    fw.close()

def grabTree(filename):
    fr=open(filename,'rb')
    return pickle.load(fr)


def createLenses(filepath):
    fr=open(filepath)
    lenses=[inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels=['age','prescript','astigmatic','tearRate']
    lensesTree=crateTree(lenses,lensesLabels)
    print(lensesTree)
def main():
    dataSet, labels=createDataSet()
    myTree=crateTree(dataSet,labels)
    storeTree(myTree,r'myTree.txt')
    print(grabTree(r'myTree.txt'))


if __name__=='__main__':
   main()
   #createLenses(r'F:\machinelearninginaction\Ch03\lenses.txt')