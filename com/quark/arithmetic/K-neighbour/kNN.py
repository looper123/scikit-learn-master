from numpy import *
import operator


def createDataset():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


if __name__ == '__main__':
    group, labels = createDataset()


def clssfify0(inX, dataSet, labels, k):
    # 计算已知类别的数据集中的点与当前点之间的距离
    # start
    dataSetSize = dataSet[0]
    diffMat = tile(inX, (dataSetSize,1))-dataSet
    sqDiffMat = diffMat**2
    sqDistance = sqDiffMat.sum(axis=1)
    distances = sqDistance**0.5
    # end
    # 选择距离最小的k个点
    # start
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    sortedClassCount = sorted(classCount.iteritems,key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


