from numpy import *
import operator


# 待分类的数据   训练样本集   标签向量（类别）   选择近邻的数目
def clssfify0(inX, dataSet, labels, k):
    # 计算已知类别的数据集中的点与当前点之间的距离 start
    dataSetSize = dataSet.shape[0]
    # 把待分类数据复制四份 组成二维数组的形式 并和训练样本集相减 得到输入数据和各个样本集x，y的差值
    # [[0 0]            [[-1. - 1.1]
    #  [0 0]            [-1. - 1.]
    #  [0 0]            [0.   0.]
    #  [0 0]]           [0. - 0.1]]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # 求各组距离的平方 [x²，y²]
    # [[1.   1.21]
    #  [1.   1.]
    # [0.   0.]
    # [0.   0.01]]
    sqDiffMat = diffMat ** 2
    # 各组平方相加 [x²+y²]
    # [2.21  2.  0.  0.01]
    sqDistance = sqDiffMat.sum(axis=1)
    # 求平方根
    distances = sqDistance ** 0.5
    # 计算已知类别的数据集中的点与当前点之间的距离 end
    # 选择距离最小的k个点 start
    # 对距离进行排序
    # [2 3 1 0]
    sortedDistIndicies = distances.argsort()
    classCount = {}
    # 挑选邻近的k个点
    for i in range(k):
        # 使用欧式距离公式计算（x0,y0）和（x1,y1）两个点间的最短距离公式    d= √(x0-x1)²+（y0-y1）²
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 选择距离最小的k个点 end
    # 排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createDataset():
    # 训练样本集
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    # 所属类别
    labels = ['A', 'A', 'B', 'B']
    return group, labels


if __name__ == '__main__':
    group, labels = createDataset()
    result = clssfify0([0, 0], group, labels, 3)
    print(result)

