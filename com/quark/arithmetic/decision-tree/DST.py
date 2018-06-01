import operator
from math import log

import matplotlib.pyplot as plt


# 决策树

# 求给定数据集的香农熵  note:当dataset中有一种类型的数据时 熵为0  当存在两种类型的数据 并且数量相同时 熵为1
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}  # {'yes': 2, 'no': 3}
    # 为所有分类创建字典 start
    for featVec in dataSet:
        currentlabel = featVec[-1]
        if currentlabel not in labelCounts.keys():
            # 如果分类不在labelCounts 中 就添加进去
            labelCounts[currentlabel] = 0
        labelCounts[currentlabel] += 1
    # 为所有分类创建字典 end
    shannonEnt = 0.0
    for key in labelCounts:
        # labelCounts中各分类占总分类的比例
        prob = float(labelCounts[key] / numEntries)
        # 求以2为底的对数
        #                c
        # Entropy(S) =   Σ -pi log2pi
        #               i=1
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


# 数据
def createDataSet():
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [1, 1, 'no']
    ]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


# 按照给定的特征划分数据集:找到对应角标的属性值==value数据 并且删掉该角标对应的属性值 最后输出
# 参数            数据集  属性对应的角标 属性值
# [[1, 'yes'], [1, 'yes'], [0, 'no'], [1, 'no']]
def splitDataSet(dataset, axis, value):
    retDataSet = []
    for featVec in dataset:
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet


# 求最大增益属性对应的角标
def chooseMaxBenefitProperty(dataset):
    numFeatures = len(dataset[0]) - 1
    # 香农熵
    baseEntroy = calcShannonEnt(dataset)
    bestInfoGain = 0.0;
    bestFeature = -1
    for i in range(numFeatures):
        # 获取训练集中的某一属性的所有值比如dataset中第一个属性 [1,1,1,0,1]组成元组
        featList = [example[i] for example in dataset]
        # 去重复 [0,1]
        uniqueVals = set(featList)
        newEntroy = 0.0
        for value in uniqueVals:
            # 按照给定的特征划分数据集
            subDataSet = splitDataSet(dataset, i, value)
            # 求该特征集占总量的比例
            prob = len(subDataSet) / float(len(dataset))
            newEntroy += prob * calcShannonEnt(subDataSet)
        # 该属性的增益
        infoGain = baseEntroy - newEntroy
        # 最大增益属性
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            # 最大增益属性对应的角标
            bestFeature = i
    return bestFeature


# 当数据集已经处理了所有属性，但是类标签已然不是唯一的，如何定义该叶子节点
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 使用dict类型来模拟决策树结构
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # 类别相同停止划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 遍历完所有特征时返回出现次数最多的类别
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 最大增益
    bestFeat = chooseMaxBenefitProperty(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


# 决策节点文本框格式
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
# 叶子节点文本框格式
leafNode = dict(boxstyle="round4", fc="0.8")
# 箭头格式
arrow_args = dict(arrowstyle="<-")
# matplotlib中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# matplotlib绘图测试
def createPlotTest():
    # x和y轴长度   颜色
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    createPlotTest.axl = plt.subplot(111, frameon=False)
    # 节点名称  节点坐标
    plotNode('决策节点', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode('叶子节点', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.axl.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


# 获取叶子节点的数目 (叶子节点:已经获得最终分类结果 )
def getNumLeaf(myTree):
    numLeaf = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        # 如果该节点下还有数据 -> 递归
        if type(secondDict[key]).__name__ == 'dict':
            numLeaf += getNumLeaf(secondDict[key])
        else:
            numLeaf += 1
    return numLeaf


# 获取树的层数
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        # 取最高的层数
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


# 决策树绘图数据准备
def retrieveTree(i):
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                   {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}]
    return listOfTrees[i]


# 父子节点填充文本信息
def plotMidText(cntrpt, parentPt, txtString):
    xMid = (parentPt[0] - cntrpt[0]) / 2.0 + cntrpt[0]
    yMid = (parentPt[1] - cntrpt[1]) / 2.0 + cntrpt[1]
    createPlot.axl.text(xMid, yMid, txtString)


def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeaf(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


# 创建决策树
def createPlot(intree):
    # create figure
    flg = plt.figure(1, facecolor='white')
    flg.clf()
    axprops = dict(xticks=[], yticks=[])
    # 创建坐标轴
    createPlot.axl = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeaf(intree))
    plotTree.totalD = float(getTreeDepth(intree))
    plotTree.xOff = -0.5 / plotTree.totalW;
    plotTree.yOff = 1.0
    plotTree(intree, (0.5, 1.0), '')
    plt.show()


# 使用决策树对数据进行分类
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    # 这里假设测试数据中的属性排列顺序和 label 中一致
    featIndex = featLabels.index(firstStr)
    for key in secondDict:
        if testVec[featIndex] ==key:
            if type(secondDict[key]).__name__ =='dict':
                classLabel = classify(secondDict[key],featLabels,testVec)
            else: classLabel = secondDict[key]
    return classLabel


# 求香农熵
def  myShannonEnt(dataSet):
    dataCount = {}  #{'yes': 2, 'no': 3}
    for data in dataSet:
        key = data[-1]
        dataCount[key] = dataCount.get(key,0)+1
    #求香农熵
    shannonEnt = 0.0
    totalEntries = len(dataSet)
    # 字典类型迭代
    for key in dataCount:
        shannonEnt -=dataCount.get(key)/totalEntries*log(dataCount.get(key)/totalEntries,2)
    return shannonEnt

#按照属性划分特征集
#找出第index个
# def  splitDataSetByProp(dataSet,index ,value):




if __name__ == '__main__':
    myDat, labels = createDataSet()
    # 求数据集熵值
    # print(calcShannonEnt(myDat))
    # 属性分类
    print(splitDataSet(myDat,0,1))
    # 最大增益属性
    # print(chooseMaxBenefitProperty(myDat))
    # 创建tree的dict结构
    # print(createTree(myDat, labels))
    # matplotlib绘图测试
    # myTree = retrieveTree(0)
    # myTree1 = retrieveTree(1)
    # print(getTreeDepth(myTree))
    # createPlot(myTree1)
    # print(matplotlib.matplotlib_fname())  #获取matplotlib包所在路径
    # print(classify(myTree,labels,[1,0]))
    # print(myShannonEnt(myDat))
