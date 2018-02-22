import numpy as np
import pandas as pd


def loadData(fileName):
    dataSet=pd.read_table(fileName,header=None).values
    dataArray=dataSet[:,:-1]
    labelArray=dataSet[:,-1]
    return dataArray,labelArray

def selectJrand(i,m):
    j=i
    while(j==i):
        j=int(np.random.uniform(0,m))
    return j

#将alpha限定在对角线内，就是这里的L和H
def clipAlgha(aj,H,L):
    if aj>H:
        aj=H
    if aj<L:
        aj=L
    return aj


def smoSimple(dataSet,classLabel,C,toler,maxIter):
    dataMatrix=np.mat(dataSet);labelMatrix=np.mat(classLabel).transpose()
    m,n=dataMatrix.shape
    b=0
    iter=0
    alphas=np.mat(np.zeros((m,1)))
    while(iter<maxIter):
        alphaPairsChanged=0
        for i in range(m):
            fXi=float((np.multiply(alphas,labelMatrix)).T*(dataMatrix*dataMatrix[i,:].T))+b
            Ei=fXi-float(labelMatrix[i])
            if ((labelMatrix[i]*Ei<-toler)and (alphas[i]<C)) or ((labelMatrix[i]*Ei>toler)and (alphas[i]>0)):
                j=selectJrand(i,m)
                fXj=float((np.multiply(alphas,labelMatrix)).T*(dataMatrix*dataMatrix[j,:].T))+b
                Ej=fXj-float(labelMatrix[j])
                alphaIold=alphas[i].copy()
                alphaJold=alphas[j].copy()
                # 当yi!=yj时，alphaJ取值范围
                if labelMatrix[i]!=labelMatrix[j]:
                    L=max(0,alphas[j]-alphas[i])
                    H=min(C,C+alphas[j]-alphas[i])
                else:
                    L=max(0,alphas[j]-alphas[i]-C)
                    H=min(C,alphas[j]+alphas[i])
                if L==H :
                    print("L==H")
                    continue
                eta=2.0*dataMatrix[i,:]*dataMatrix[j,:].T-dataMatrix[i,:]*dataMatrix[i,:].T-dataMatrix[j,:]*dataMatrix[j,:].T
                if eta>=0:
                    print("eta>=0")
                    continue
                # 解二元规划得到迭代后未约束的解
                alphas[j] -= labelMatrix[j] * (Ei - Ej) / eta
                # 将解约束到取值范围里
                alphas[j]=clipAlgha(alphas[j],H,L)
                if(abs(alphas[j]-alphaJold)<0.00001):
                    print("j is not moving enough")
                    continue
                # 更新alpha[i]
                alphas[i]+=labelMatrix[i]*labelMatrix[j]*(alphaJold - alphas[j])
                # 在SM0算法中除了更新alpha，还需要更新b
                b1=b-Ei-labelMatrix[i]*dataMatrix[i]*dataMatrix[i].T*(alphas[i]-alphaIold)-labelMatrix[j]*dataMatrix[i]*dataMatrix[j].T*(alphas[j]-alphaJold)
                b2=b-Ej-labelMatrix[i]*dataMatrix[i]*dataMatrix[i].T*(alphas[i]-alphaIold)-labelMatrix[j]*dataMatrix[j]*dataMatrix[j].T*(alphas[j]-alphaJold)
                if alphas[i]>0 and alphas[i]<C:
                    b=b1
                if alphas[j]>0 and alphas[j]<C:
                    b=b2
                else:
                    b=(b1+b2)/2.0
                alphaPairsChanged+=1
                print("iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
        if(alphaPairsChanged ==0): iter+=1
        else:iter=0
        print("iteration number: %d" % iter)
    return b,alphas


class optStruct:
    def __init__(self,dataMatIn,classLabels,C,toler):
        self.X=dataMatIn
        self.labelMat=classLabels
        self.C=C
        self.tol=toler
        self.m=dataMatIn.shape()[0]
        self.b=0
        self.alphas=np.mat(np.zeros((self.m,1)))
        self.eCache=np.mat(np.zeros((self.m,2)))

def calcEk(oS,k):
    fXk=float(np.multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T))+oS.b
    Ek=fXk-float(oS.labelMat[k])
    return Ek

def selectJ(i,oS,Ei):
    maxK=-1 ;maxDeltaE=0;Ej=0
    oS.eCache[i]=[1,Ei]
    validEcacheList=np.nonzero(oS.eCache[:,0].A)[0]
    if (len(validEcacheList))>1:
        for k in validEcacheList:
            if k==i :
                continue
            Ek=calcEk(oS,k)
            deltaE=np.abs(Ei-Ek)
            if (deltaE>maxDeltaE):
                maxK=k;maxDeltaE=deltaE;Ej=Ek
        return maxK,Ej
    else:
        j=selectJrand(i,oS.m)
        Ej=calcEk(oS,j)
    return j,Ej

def updateEk(oS,k):
    Ek=calcEk(oS,k)
    oS.eCache[k]=[1,Ek]




alphas=np.ones((100,1))
dataSet,label=loadData(r'testSet.txt')
b1,alphas1=smoSimple(dataSet,label,0.6,0.001,40)
print(b1)
print(b)
