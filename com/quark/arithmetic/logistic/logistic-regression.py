import random

import matplotlib.pyplot as plt

# logistic回归算法


# 回归梯度上升优化算法 流程
# 1 准备数据
# 2 准备sigmoid函数
# 3 使用梯度上升算法/随机梯度上升算法  求回归系数函数
#       note:梯度上升算法中使用的是矩阵  而随机梯度上升算法中使用的是numpy数组
#        指定迭代次数（经过多少次计算后最后得出回归系数）  指定步长: 沿梯度上升每步的长度  初始化回归系数单列多行numpy数组
#         [[1.]
#          [1.]
#          [1.]]  初始值选择 sigmoid 函数的边界点1
#        在每一个迭代次数中需要做：以训练数据*回归系数为输入 求出sigmoid函数值作为预测分类值 ; 求真实分类值和预测分类值的差值 计为h
#         回归系数 weights(n) = weights(n-1)+ 步长 * (训练样本转置矩阵) * (真实分类值和预测分类值的差值)  ???怎么推导的

# 随机梯度上升算法
# 和梯度上升算法基本一致
# 区别 ：1 随机梯度上升算法中不用矩阵 使用numpy数组做计算
#       2 随机梯度上升算法中预测值 和差值都是数值 而不是向量
# 精准度改进：
#       1 步长是不固定的 每次迭代后都会缩小 但是不会≤0（不然会使后续训练数据无法对结果造成影响）
#       2 随机取训练样本中的数据进行计算 （不要忘了把计算过的数据从数组中删除）

# 家在训练数据
from numpy import mat, shape, ones, array, arange
from numpy.ma import exp


# 加载数据
def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('test.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,
    labelMat


# sigmoid函数
def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


# 梯度上升算法 求回归系数
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)  # 一个二维数组    行 表示每一个训练样本  列 表示数据的特征
    labelMat = mat(classLabels).transpose()  # 矩阵的转置   labelMat是一个 100行单列矩阵(标识出了100条数据的真实结果)
    m, n = shape(dataMatrix)  # m表示行数  n表示列数
    alpha = 0.001  # 向目标移动的步长
    maxCycles = 500  # 迭代次数
    weights = ones((n, 1))  # 初始化回归系数数组   这里n=3
    #[[1.]
    #[1.]
    # [1.]]
    for k in range(maxCycles):
        # 矩阵相乘
        h = sigmoid(
            dataMatrix * weights)  # 训练数据(100*3矩阵) 和 初始回归系数(3*1)  矩阵相乘 得到 100*1矩阵  并对结果矩阵求sigmoid函数  得到的h也是一个100*1矩阵
        error = (labelMat - h)  #  代价函数  真实类别和预测类别的差值
        # 如何推导出来的
        weights += alpha * dataMatrix.transpose() * error  # 按照差值方向调整回归系数  步长*训练样本矩阵（3*100矩阵）*差值矩阵(100*1矩阵)
    return weights  # 回归系数  3*1 矩阵


# 画出决策边界
def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xCord1 = []
    yCord1 = []
    xCord2 = []
    yCord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xCord1.append(dataArr[i, 1])
            yCord1.append(dataArr[i, 2])
        else:
            xCord2.append(dataArr[i, 1])
            yCord2.append(dataArr[i, 2])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(xCord1, yCord1, s=30, c='red', marker='s')
        ax.scatter(xCord2, yCord2, s=30, c='green')
        x = arange(-3.0, 3.0, 0.1)
        y = (-weights[0] - weights[1] * x) / weights[2]  # 最佳拟合线
        # 公式推导：
        #         因为当sigmoid函数输入为0时为该函数的分界线
        #         即: 0 =w0*x0+w1*x1+w2*x2
        ax.plot(x, y)
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.show()


# 随机梯度上升算法
# 与普通梯度上升算法的区别
# 1 h和error都是值 而不是向量
# 2 这里没有矩阵转换过程 所有的变量数据类型都是numpy数组
def stocGradeAscent0(dataMatrix, classLabels):
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


# 随机梯度上升算法改进版
# 方法参数中给变量赋值表示：如果没有传值则用该默认值 ，如果传值了 就使用传来的值
def stocGradeAscent1(dataMatrix,classLabels ,numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.1   #改动1  步长随着迭代会变小 但是不会 ≤ 0  不然会使后续的数据无法对结果产生影响
            randIndex = int(random.uniform(0,len(dataIndex))) #改动2 随机选取数据更新回归系数
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha *error*dataMatrix[randIndex]
            del (dataIndex[randIndex])  #删掉已经计算过回归系数的数据
    return weights

# 使用logistic回归算法预测病马的生死问题
def classifyVector(inx , weights):
    # 对特征向量和回归系数的乘积求和 作为sigmoid函数的输入值
    prob = sigmoid(sum(inx * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def colicTest():
    frTrain = open('horstraining.txt')
    frTest = open('horstest.txt')
    trainingSet = []
    trainingLabels = []
    # 按行读取训练数据
    for line in frTrain.readlines():
        currLine = line.strip().split("\t")
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    # 随机梯度上升算法
    trainWeights = stocGradeAscent1((array(trainingSet),trainingLabels,500))
    errorCount = 0
    numTestVec = 0.0
    # 按行读取测试数据
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split("\t")
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        # 求sigmoid函数结果  和真实结果比较
        if int(classifyVector(array(lineArr),trainWeights))!= int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("error rate of test is:%f" %errorRate)
    return errorRate

def  multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average errorrate is :%f" %(numTests,errorSum/float(numTests)))


if __name__ == '__main__':
    print(ones((3, 1)) )
