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
                    print("j 几乎没有移动")
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


'''
#######********************************
带核函数部分
#####*************************************
'''

class optStructK:
    def __init__(self,dataMatIn, classLabels, C, toler, kTup):  # Initialize the structure with the parameters
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m,1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m,2))) #first column is valid flag
        self.K = np.mat(np.zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)
def calcEkK(oS,k):
    fXk=float(np.multiply(oS.alphas,oS.labelMat).T*oS.K[:,k])+oS.b
    Ek=fXk-float(oS.labelMat[k])
    return Ek

def selectJK(i,oS,Ei):
    maxK=-1 ;maxDeltaE=0;Ej=0
    oS.eCache[i]=[1,Ei]
    # 找出已经计算过Ei的alpha
    validEcacheList=np.nonzero(oS.eCache[:,0].A)[0]
    if (len(validEcacheList))>1:
        for k in validEcacheList:
            if k==i :
                continue
            Ek=calcEkK(oS,k)
            deltaE=np.abs(Ei-Ek)
            if (deltaE>maxDeltaE):
                maxK=k;maxDeltaE=deltaE;Ej=Ek
        return maxK,Ej
    else:
        j=selectJrand(i,oS.m)
        Ej=calcEkK(oS,j)
    return j,Ej

def updateEk(oS,k):
    Ek=calcEk(oS,k)
    oS.eCache[k]=[1,Ek]

def innerLK(i,oS):
    Ei=calcEkK(oS,i)
    if((oS.labelMat[i]*Ei<-oS.tol) and (oS.alphas[i]<oS.C)) or ((oS.labelMat[i]*Ei>oS.tol)and (oS.alphas[i]>0)):
        j,Ej=selectJK(i,oS,Ei)
        alphaIold=oS.alphas[i].copy();alphaJold=oS.alphas[j].copy()
        if (oS.labelMat[i]!=oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] +oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L==H")
            return 0
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
        if eta>=0:
            print("eta>=0")
            return 0
        oS.alphas[j]-=oS.labelMat[j]*(Ei-Ej)/eta
        oS.alphas[j]=clipAlgha(oS.alphas[j],H,L)
        updateEk(oS,j)
        if (abs(oS.alphas[j]-alphaJold)<0.00001):
            print("j 几乎没有移动")
            return 0
        oS.alphas[i]+=oS.labelMat[i]*oS.labelMat[j]*(alphaJold-oS.alphas[j])
        updateEk(oS,i)
        b1=oS.b-Ei-oS.labelMat[i]*oS.K[i,i]*(oS.alphas[i]-alphaIold)-oS.labelMat[j]*oS.K[i,j]*(oS.alphas[j]-alphaJold)
        b2=oS.b-Ej-oS.labelMat[i]*oS.K[i,j]*(oS.alphas[i]-alphaIold)-oS.labelMat[j]*oS.K[j,j]*(oS.alphas[j]-alphaJold)
        if (oS.alphas[i]>0) and (oS.alphas[i]<oS.C) :
            oS.b=b1
        elif (oS.alphas[j]>0 )and (oS.alphas[j]<oS.C):
            oS.b=b2
        else:
            oS.b=(b1+b2)/2.0
        #表示有一对alpha被改变
        return 1
    else :return 0

# Smo算法
def smoPK(dataMatIn,classLabels,C,toler,maxIter,kTup):
    oS=optStructK(np.mat(dataMatIn),np.mat(classLabels).transpose(),C,toler,kTup)
    iter=0
    entireSet=True
    #计算每次迭代alphas中的aplha对改变了多少次
    alphaPairsChanged=0
    while(iter<maxIter) and ((alphaPairsChanged>0) or (entireSet)):
        alphaPairsChanged=0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged+=innerLK(i,oS)
            iter+=1
        else:
            # 优化在0<aplha<C范围的alpha，即扫描间隔边界上样本点
            nonBoundIs=np.nonzero((oS.alphas.A>0)*(oS.alphas.A<C))[0]
            for i in nonBoundIs:
                alphaPairsChanged+=innerLK(i,oS)
            iter+=1
        if entireSet:
            entireSet=False
        #如果间隔边界上的样本点都满足KKT条件就遍历整个训练集
        elif alphaPairsChanged==0:
            entireSet=True
    return oS.b,oS.alphas

def calcWs(alphas,dataArr,classLabels):
    X=np.mat(dataArr);labelMat=np.mat(classLabels).transpose()
    m,n=np.shape(X)
    #利用smo求解出来的alpha和kkt条件得到超平面法向量
    w=X.T*np.multiply(alphas,labelMat)
    return w


def kernelTrans(X,A,kTup):
    # kTup为包含核函数信息的元组
    m,n=np.shape(X)
    K=np.mat(np.zeros((m,1)))
    #线性，不做核转换，可换为多项式核
    if kTup[0]=='lin':
        K=X*A.T
    elif kTup[0]=='rbf':
        for i in range(m):
            deltaRow=X[i,:]-A
            K[i]=deltaRow*deltaRow.T
        K=np.exp(K/(-1*kTup[1]**2))
    else:
        raise NameError("还不支持这个kernel函数")
    return K

def testRbf(k1=1.3):
    dataArr,labelArr=loadData(r'testSetRBF.txt')
    b,alphas=smoPK(dataArr,labelArr,200,0.0001,10000,('rbf',k1))
    dataMat=np.mat(dataArr);labelMat=np.mat(labelArr).transpose()
    svInd=np.nonzero(alphas.A>0)[0]
    #支持向量
    sVs=dataMat[svInd]
    labelSV=labelMat[svInd]
    print("有%d个支持向量" %(np.shape(sVs)[0]))
    m,n=np.shape(dataMat)
    errorCount=0
    for i in range(m):
        kernelEval=kernelTrans(sVs,dataMat[i,:],('rbf',k1))
        predict=kernelEval.T*np.multiply(labelSV,alphas[svInd])+b
        if np.sign(predict)!= labelMat[i]:
            errorCount+=1
    print("训练误差为 : %f" %(float(errorCount/m)))
    test_dataArr,test_labelArr=loadData(r'testSetRBF2.txt')
    test_errorCount=0
    test_dataMat=np.mat(test_dataArr)
    test_labelMat=np.mat(test_labelArr).transpose()
    test_m,test_n=np.shape(test_dataMat)
    for i in range(test_m):
        kernelEval=kernelTrans(sVs,test_dataMat[i,:],('rbf',k1))
        predict=kernelEval.T*np.multiply(labelSV,alphas[svInd])+b
        if np.sign(predict)!=np.sign(test_labelMat[i]):
            test_errorCount+=1
    print("测试误差 为 : %f"  %(float(test_errorCount/test_m)))

def testRbf1(k1=1.3):
    dataArr,labelArr = loadData('testSetRBF.txt')
    b,alphas = smoPK(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1)) #C=200 important
    datMat=np.mat(dataArr); labelMat = np.mat(labelArr).transpose()
    svInd=np.nonzero(alphas.A>0)[0]
    sVs=datMat[svInd] #get matrix of only support vectors
    labelSV = labelMat[svInd];
    print ("there are %d Support Vectors" % np.shape(sVs)[0])
    m,n = np.shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b
        if np.sign(predict)!=np.sign(labelArr[i]): errorCount += 1
    print ("the training error rate is: %f" % (float(errorCount)/m))
    dataArr,labelArr = loadData('testSetRBF2.txt')
    errorCount = 0
    datMat=np.mat(dataArr); labelMat = np.mat(labelArr).transpose()
    m,n = np.shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b
        if np.sign(predict)!=np.sign(labelArr[i]): errorCount += 1
    print ("the test error rate is: %f" % (float(errorCount)/m) )


'''
#######********************************
不应用核函数部分
#######********************************
'''
class optStruct:
    def __init__(self,dataMatIn,classLabels,C,toler):
        self.X=dataMatIn
        self.labelMat=classLabels
        self.C=C
        self.tol=toler
        self.m=dataMatIn.shape[0]
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
    # 找出已经计算过Ei的alpha
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


def innerL(i,oS):
    Ei=calcEk(oS,i)
    if((oS.labelMat[i]*Ei<-oS.tol) and (oS.alphas[i]<oS.C)) or ((oS.labelMat[i]*Ei>oS.tol)and (oS.alphas[i]>0)):
        j,Ej=selectJ(i,oS,Ei)
        alphaIold=oS.alphas[i].copy();alphaJold=oS.alphas[j].copy()
        if (oS.labelMat[i]!=oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] +oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L==H")
            return 0
        eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - oS.X[j, :] * oS.X[j, :].T
        if eta>=0:
            print("eta>=0")
            return 0
        oS.alphas[j]-=oS.labelMat[j]*(Ei-Ej)/eta
        oS.alphas[j]=clipAlgha(oS.alphas[j],H,L)
        updateEk(oS,j)
        if (abs(oS.alphas[j]-alphaJold)<0.00001):
            print("j is not moving enough")
            return 0
        oS.alphas[i]+=oS.labelMat[i]*oS.labelMat[j]*(alphaJold-oS.alphas[j])
        updateEk(oS,i)
        b1=oS.b-Ei-oS.labelMat[i]*oS.X[i,:]*oS.X[i,:].T*(oS.alphas[i]-alphaIold)-oS.labelMat[j]*oS.X[i,:]*oS.X[j,:].T*(oS.alphas[j]-alphaJold)
        b2=oS.b-Ej-oS.labelMat[i]*oS.X[i,:]*oS.X[j,:].T*(oS.alphas[i]-alphaIold)-oS.labelMat[j]*oS.X[j,:]*oS.X[j,:].T*(oS.alphas[j]-alphaJold)
        if (oS.alphas[i]>0) and (oS.alphas[i]<oS.C) :
            oS.b=b1
        elif (oS.alphas[j]>0 )and (oS.alphas[j]<oS.C):
            oS.b=b2
        else:
            oS.b=(b1+b2)/2.0
        #表示有一对alpha被改变
        return 1
    else :return 0

# Smo算法
def smoP(dataMatIn,classLabels,C,toler,maxIter):
    oS=optStruct(np.mat(dataMatIn),np.mat(classLabels).transpose(),C,toler)
    iter=0
    entireSet=True
    #计算每次迭代alphas中的aplha对改变了多少次
    alphaPairsChanged=0
    while(iter<maxIter) and ((alphaPairsChanged>0) or (entireSet)):
        alphaPairsChanged=0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged+=innerL(i,oS)
            iter+=1
        else:
            # 优化在0<aplha<C范围的alpha，即扫描间隔边界上样本点
            nonBoundIs=np.nonzero((oS.alphas.A>0)*(oS.alphas.A<C))[0]
            for i in nonBoundIs:
                alphaPairsChanged+=innerL(i,oS)
            iter+=1
        if entireSet:
            entireSet=False
        #如果间隔边界上的样本点都满足KKT条件就遍历整个训练集
        elif alphaPairsChanged==0:
            entireSet=True
    return oS.b,oS.alphas


def calcWs(alphas,dataArr,classLabels):
    X=np.mat(dataArr);labelMat=np.mat(classLabels).transpose()
    m,n=np.shape(X)
    #利用smo求解出来的alpha和kkt条件得到超平面法向量
    w=X.T*np.multiply(alphas,labelMat)
    return w


'''
#######********************************
用svm对手写识别
#######********************************
'''
def img2vector(filename):
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def loadImages(dirName):
    from os import listdir
    hwLabels=[]
    trainingFileList=listdir(dirName)
    m=len(trainingFileList)
    trainingMat=np.zeros((m,1024))
    for i in range(m):
        fileNameStr=trainingFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        if classNumStr==9: hwLabels.append(-1)
        else:hwLabels.append(1)
        trainingMat[i,:]=img2vector('%s/%s' %(dirName,fileNameStr))
    return trainingMat,hwLabels

def testDigits(kTup=('rbf',10)):
    dataArr,labelArr=loadImages('trainingDigits')
    b,alphas=smoPK(dataArr,labelArr,200,0.0001,10000,kTup)
    dataMat=np.mat(dataArr)
    labelMat=np.mat(labelArr).transpose()
    #支持向量的下标
    svInd=np.nonzero(alphas.A>0)[0]
    #支持向量
    sVs=dataMat[svInd]
    labelSV=labelMat[svInd]
    print("有%d个支持向量" %len(labelSV))
    m,n=np.shape(dataMat)
    errorCount=0
    for i in range(m):
        kernelEval=kernelTrans(sVs,dataMat[i,:],kTup)
        predict=kernelEval.T*np.multiply(labelSV,alphas[svInd])+b
        if np.sign(predict)!=np.sign(labelMat[i]):
            errorCount+=1
    with open(r'indicate.txt',"a") as f:
            f.writelines('#######********************************\n\
      %s,参数为%s                '  %(kTup[0],kTup[1]))
            f.writelines("支持向量个数 : %d              " %len(labelSV))
            f.writelines("训练误差为 : %f                " %(errorCount/m))
    print("训练误差为 : %f" %(errorCount/m))
    test_dataArr,test_labelArr=loadImages('testDigits')
    test_errorCount=0
    test_dataMat=np.mat(test_dataArr)
    test_labelMat=np.mat(test_labelArr)
    test_m,test_n=np.shape(test_dataMat)
    for i in range(test_m):
        kernelEval=kernelTrans(sVs,test_dataMat[i,:],kTup)
        predict=kernelEval.T*np.multiply(labelSV,alphas[svInd])+b
        if np.sign(predict)!=np.sign(test_labelArr[i]):
            test_errorCount+=1
    with open(r'indicate.txt',"a") as g:
        g.writelines("测试误差为 : %f          \n" %(test_errorCount/test_m))
        g.writelines("#######********************************\n\n")
    print("测试误差为 : %f" %(test_errorCount/test_m))




#dataSet,label=loadData(r'testSet.txt')
#b1,alphas1=smoSimple(dataSet,label,0.6,0.001,40)
#b2,alphas2=smoP(dataSet,label,0.6,0.001,40)

#print(alphas2)
#testRbf()
#testRbf(k1=0.1)
testDigits(('rbf',20))
testDigits(('rbf',5))
testDigits(('rbf',10))
testDigits(('rbf',50))
testDigits(('lin',0))
