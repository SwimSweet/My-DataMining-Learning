import numpy as np
import re

def loadData():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList,classVec

def createVec(dataSet):
    vocabSet=set([])
    for document in dataSet:
        vocabSet=vocabSet| set(document)
    return list(vocabSet)

def setOfWordsVec(vocabList,inputSet):
    retutnVec=[0 for  x in range(len(vocabList))]
    for word in inputSet:
        try:
            index=vocabList.index(word)
            retutnVec[index]+=1
        except Exception  as e:
            print("the word {0} is not in my Vocabulary".format(e))
    return retutnVec


def main():
    testingNB()
    spamTest()

def trainNbO(wordVec,tarinlebel):
    docNum=len(wordVec)
    pro1=sum(tarinlebel)/len(tarinlebel)
    numWord=len(wordVec[0])
    #labelwordcount1=np.zeros(numWord)
    #labelwordcount0=np.zeros(numWord)
    #防止分子分母概率为0情况，把单词出现初始设为1，分母设为2
    labelwordcount1=np.ones(numWord)
    labelwordcount0=np.ones(numWord)
    lebelSum1=2
    labelSum0=2
    for i in range(docNum):
        if tarinlebel[i] == 1:
            labelwordcount1 += wordVec[i]
            lebelSum1 += sum(wordVec[i])
        else:
            labelwordcount0 += wordVec[i]
            labelSum0 += sum(wordVec[i])
    #转换为对数
    proVec1=np.log(labelwordcount1/lebelSum1)
    proVec0=np.log(labelwordcount0/labelSum0)
    return proVec0, proVec1,pro1

def testingNB():
    listOfpast, lebels = loadData()
    wordVec = createVec(listOfpast)
    wordVecCount = []
    for listWord in listOfpast:
        wordVecCount.append(setOfWordsVec(wordVec, listWord))
    proVec0, proVec1, pro1 = trainNbO(wordVecCount, lebels)
    testEntry = ['love', 'my', 'dalmation']
    thisDoc=np.array(setOfWordsVec(wordVec,testEntry))
    print(testEntry, 'classified as:{0} '.format(classifyNB(thisDoc, proVec0, proVec1, pro1)))
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWordsVec(wordVec, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, proVec0, proVec1, pro1))


def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1=sum(vec2Classify*p1Vec)+np.log(pClass1)
    p0=sum(vec2Classify*p0Vec)+np.log(1-pClass1)
    if(p1>p0):
        return 1
    else:
        return 0

def spamTest():
    docList=[];classList=[];fullText=[]
    for i in range(1,26):
        path1=r'email/spam/{0}.txt'.format(i)
        wordList=textParse(path1)
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        path0=r'email/ham/{0}.txt'.format(i)
        wordList=textParse(path0)
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList=createVec(docList)
    trainingSet=[i for i in range(50)];testSet=[]
    #随机选出10个测试样本
    for i in range(10):
        randIndex=int(np.random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[];trainClasses=[]
    #训练样本训练模型
    for docIndex in trainingSet:
        trainMat.append(setOfWordsVec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    poV,p1V,PsPam=trainNbO(np.array(trainMat),np.array(trainClasses))
    errorCount=0
    for docIndex in testSet:
        wordVector=setOfWordsVec(vocabList,docList[docIndex])
        if classifyNB(wordVector,poV,p1V,PsPam)!=classList[docIndex]:
            errorCount+=1
    print(errorCount)

def textParse(bigString):
    with open(bigString,'rb') as f:
        email=str(f.read())
        listOfTokens=re.split(r'\W*',email)
        return [tok.lower() for tok in listOfTokens if len(tok)>2]

if __name__=='__main__':
    main()


