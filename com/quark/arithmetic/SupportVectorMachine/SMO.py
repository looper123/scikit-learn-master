# 支持向量机



# SMO算法中的辅助函数
import random

from numpy import mat, zeros, shape, multiply


def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('1.txt')
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat


def selectJrand(i,m):
    j = i
    while (j==i):
        j =int(random.uniform(0,m))
    return j

def clipAlpha(aj,h,l):
    if aj > h:
        aj = h
    if l > aj:
        aj = l
    return aj


# 简化版SVM算法
def smoSimple(dataMatin,classlabels,C,toler ,maxIter):
    dataMatrix = mat(dataMatin)
    labelMat = mat(classlabels).transpose()
    b = 0
    m,n = shape(dataMatrix)
    alphas = mat(zeros(m,1))
    iter = 0
    while(iter < maxIter):
        alphaParisChanged = 0
        for i in range(m):
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T))+b
            Ei = fXi - float(labelMat[i])
            if((labelMat[i]*Ei < -toler) and  (alphas[i]<C) or (labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i,m)
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j:].T))+b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if(labelMat[i] != labelMat[j]):
                    L = max(0,alphas[j]-alphas[i])
                    H = min(C,C+alphas[j]-alphas[i])
                else:
                    L = max(0,alphas[j]+alphas[i]-C)
                    H = min(C,alphas[j]+alphas[i])
                if L==H:
                    print('L==H')
                    continue
                eta = 2.0*dataMatrix[j,:]*dataMatrix[i,:].T-dataMatrix[i,:]*dataMatrix[i,:].T-dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0:
                    print('eta >=0')
                    continue
                alphas[j] -=labelMat[j]*(Ei-Ej)/eta
                alphas[j] =clipAlpha(alphas[j],H,L)
                if(abs(alphas[j] -alphaJold) <0.00001):
                    print('j not moving enough!')
                    continue
                alphas[i] +=labelMat[j]*labelMat[i]*(alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i]*(alphas[i] -alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T-labelMat[j,:]*(
                    alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej - labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(
                    alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if(0 < alphas[i]  and (C > alphas[i])):
                    b = b1
                elif(0 < alphas[j] and C >alphas[j]):
                    b = b2
                else:
                    b = (b1+b2)/2.0
                alphaParisChanged +=1
                print('iter :%d i :%d,paris changed %d'%(iter,i,alphaParisChanged))
            if(alphaParisChanged == 0):
                iter += 1
            else :iter = 0
            print('iteration number:%d' %iter)
        return b,alphas



if __name__ == '__main__':
    print()