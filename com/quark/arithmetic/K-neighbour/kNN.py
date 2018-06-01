from os import listdir

from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt

# k-近邻算法

# 待分类的数据   训练样本集   标签向量（类别）   选择近邻的数目
def classify0(inX, dataSet, labels, k):
    # 计算已知类别的数据集中的点与当前点之间的距离 start
    dataSetSize = dataSet.shape[0]
    # 把待分类数据复制四份 组成二维数组的形式 并和训练样本集相减 得到输入数据和各个样本集x，y的差值
    # [[0 0]            [[-1. - 1.1]
    #  [0 0]            [-1. - 1.]
    #  [0 0]            [0.   0.]
    #  [0 0]]           [0. - 0.1]]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # 使用欧式距离公式计算（x0,y0）和（x1,y1）两个点间的最短距离公式    d= √(x0-x1)²+（y0-y1）²
    # 求各组距离的平方 [x²，y²]
    # [[1.   1.21]
    #  [1.   1.]
    # [0.   0.]
    # [0.   0.01]]
    sqDiffMat = diffMat ** 2
    # 各组平方相加 [x²+y²]
    # [2.21
    #    2.
    #    0.
    #    0.01]
    sqDistance = sqDiffMat.sum(axis=1)
    # 求平方根
    distances = sqDistance ** 0.5
    # 计算已知类别的数据集中的点与当前点之间的距离 end
    # 选择距离最小的k个点 start
    # 对距离进行排序
    # [2 3 1 0]  返回从小到大的索引值
    sortedDistIndicies = distances.argsort()
    classCount = {}
    # 挑选邻近的k个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        # 邻近的点按分类叠加放入 classCount -->  classCount {'A':1,'B':2}
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 选择距离最小的k个点 end
    # 排序 获取票数最高的分类 得到该点所属的分类
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# read data from file
def files2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines,3))
    classLabelsVector = []
    index = 0
    for line in arrayOLines:
        line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] =listFromLine[0:3]
        classLabelsVector.append(int(listFromLine[-1]))
        index +=1
    return returnMat,classLabelsVector


# 在使用欧式距离公式计算距离的时候 可能出现两个属性之间之间差值较大的情况 ，这会使该属性对计算结果的影响远远大于其他属性（而现实情况是所有属性同等重要）如何处理？？
# 使用归一化数值  公式  newValue = (oldValue - minVal)/(maxVal - minVal)  把所有原始数据取值范围处理到0~1 或者 -1~0之间提高最后结果的精确度
# 训练数据归一化
def autoNorm(dataSet):
    # 每列最小值
    minVals = dataSet.min(0)
    # 每列最大值
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    # 因为特征值（训练数据）组成的矩阵是1*3 列的矩阵 所以minVals 和 range的值也必须为 1*3 列的矩阵 （不然无法做数学计算）
    normDataSet = dataSet-tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))  #note that:  '/' in this place means  特征值相除而不是矩阵除法  在numpy包中要使用矩阵除法：linalg.solve(matA,matB)
    return normDataSet,ranges,minVals


# 分类器对约会网站的测试代码
def datingClassTest():
    horatio = 0.10
    # read data from file
    datingDataMat,datingLabels = files2matrix('datingTestSet.txt')
    # 数据归一化
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*horatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        # 待处理数据分类
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("classifyResult:%d;realAnwer:%d"%(classifierResult,datingLabels[i]))
        # 结果正确性判断
        if(classifierResult !=datingLabels[i]):errorCount +=1.0
    # 计算错误率
    print("total error rate is :%f"%(errorCount/float(numTestVecs)))



    # 手写识别系统
    # 把图像转化为向量  创建 1*1024 的numpy数组  打开给定文件 循环读出文件前32行 并把每行头32个字符存储在numpy数组中 返回该数组
    def img2Vector(filename):
        returnVect = zeros((1,1024))
        fr = open(filename)
        for i in range(32):
            lineStr = fr.readline()
            for j in range(32):
                returnVect[0,32*i+j] = int(lineStr[j])
        return returnVect


    # 使用k近邻算法识别手写数字
    def  handWritingClassTest():
        hwLabels = []
        # 获取目录内容
        trainingFileList = listdir('trainingDigits')
        m = len(trainingFileList)
        trainingMat = zeros((m,1024))
        for i in range(m):
            # 从文件名解析分类数字
            fileNameStr = trainingFileList[i]
            fileStr = fileNameStr.split('.')[0]
            classNumStr = int(fileStr.split('_')[0])
            hwLabels.append(classNumStr)
            trainingMat[i:]  = img2Vector('traningDigits/%s'%fileNameStr)
        testFileList = listdir('testDigits')
        errorCount = 0.0
        mTest = len(testFileList)
        for i in range(mTest):
            fileNameStr = testFileList[i]
            fileStr = fileNameStr.split('.')[0]
            classNumStr = int(fileStr.split('_')[0])
            vectorUnderTest = img2Vector('testDigits%s'%fileNameStr)
            classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
            print("classifyResult:%d;realAnwer:%d" % (classifierResult, datingLabels[i]))
            # 结果正确性判断
            if (classifierResult != classNumStr): errorCount += 1.0
        # 错误率
        print("total error rate is :%f" % (errorCount / float(mTest)))


def createDataset():
    # 训练样本集
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    # 所属类别
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def myKnnArithMetic(orginData ,orginResult ,testData ,k):
    size = orginData.shape[0]
    # 获取最大 最小值
    maxData = group.max(0)
    minData = group.min(0)
    # 训练数据归一化
    normalData =( group-tile(minData,(size,1)))/(tile(maxData-minData,(size,1)))
    difValSquare = (tile(testData,(size,1))-normalData)**2
    difValSum = difValSquare.sum(axis=1)
    difValSqrt = difValSum**0.5
    indexOrder = difValSqrt.argsort()
    rate = {}
    for i in range(k):
        # 存在+1 不存在初始化为0
        key =  orginResult[indexOrder[i]]
        rate[key] = rate.get(key,0)+1
    #获取属性占比最多的数据
    sortedDict = sorted(rate, key=operator.itemgetter(0),reverse=True)
    return sortedDict[0][0]




if __name__ == '__main__':
    # group, labels = createDataset()
    # result = classify0([0, 0], group, labels, 3)
    # print(result)
    # print(sorted(group,key= lambda a: a[0],reverse= True)[0][0])
    group = array([[100.0, 123.1], [131.0, 161.0], [140, 200], [100, 212.1],[147,174]])
    myData = [150,150]
    labels = ['A','B','A','B','B']
    print(myKnnArithMetic(group,labels,myData,4))

