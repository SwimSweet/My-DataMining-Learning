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
    dataMatrix=np.mat(dataSet[:,:-1]);labelMatrix=np.mat(classLabel).transpose()
    m,n=dataMatrix.shape
    b=0
    iter=0
    alphas=np.zeros((m,1))
    while(iter<maxIter):
        alphaPairsChanged=0
        for i in range(m):
            fXi=float((np.multiply(alphas,labelMatrix)).T*(dataMatrix*dataMatrix[i,:].T))+b
            Ei=fXi-float(labelMatrix[i])
            if ((labelMatrix[i]*Ei<-toler)and (alphas[i]<C)) or ((labelMatrix[i]*Ei>toler)and (alphas[i]>0)):
                j=selectJrand(i,m)
                fXj=float((np.multiply(alphas,labelMatrix)).T*(dataMatrix*dataMatrix[j,:].T))+b
                Ej=fXj-labelMatrix[j]
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
                eta=2.0*dataMatrix[i,:]*dataMatrix[j,:]-dataMatrix[i,:]*dataMatrix[i,:].T-dataMatrix[j,:]-dataMatrix[j,:].T
                if eta>=0:
                    print("eta>=0")
                    continue
                # 解二元规划得到迭代后未约束的解
                #alphas[j]-=labelMatrix[j]*(Ei-Ej)/eta
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

def smoSimple1(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = np.mat(dataMatIn); labelMat = np.mat(classLabels).transpose()
    b = 0; m,n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros((m,1)))
    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b
            Ei = fXi - float(labelMat[i])#if checks if an example violates KKT conditions
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i,m)
                fXj = float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy();
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H:
                    print ("L==H")
                    continue
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0:
                    print ("eta>=0")
                    continue
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlgha(alphas[j],H,L)
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print ("j not moving enough")
                    continue
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])#update i by the same amount as j
                                                                        #the update is in the oppostie direction
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print ("iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
        if (alphaPairsChanged == 0): iter += 1
        else: iter = 0
        print ("iteration number: %d" % iter)
    return b,alphas
alphas=np.ones((100,1))
dataSet,label=loadData(r'testSet.txt')
b,alphas=smoSimple1(dataSet,label,0.6,0.001,40)
#b1,alphas1=smoSimple(dataSet,label,0.6,0.001,40)
print(b)
print(alphas)
