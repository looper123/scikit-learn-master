# 贝叶斯分类算法流程：
# 1 准备数据
# 2 训练数据（总数据的80%）求并集
# 3 根据并集和输入数据求词集/词袋模型 （用在并集矩阵中标识出输入数据）
# 4 准备贝叶斯分类函数
# 5 根据所有训练数据的词集/词袋模型 和训练数据的分类情况求出各个分类的概率分布矩阵 和各分类占总类别比例
#     note: 优化矩阵中为0的元素(初始化为1)  优化值下溢出 即得出的概率值太小 四舍五入后等于0(初始值设置为2)
# 6 测试准确率 将总数据中随机取20%作为测试数据 代入贝叶斯函数  比较由分类器得出的结果和测试数据自身的结果  相同：测试通过  不同：测试未通过
#     最后结果 = 未通过测试数据的数量/测试数据总数
import operator
import random
import re

from numpy.ma import ones, log, array


def loadDataSet():
    postingList = [
        ['I', 'always', 'wanted', 'an', 'Alaskan', 'sled', 'dog'],
        ['It', 'is', 'comfortable', 'to', 'sleep', 'on', 'your', 'stomach', 'in', 'class'],
        ['You', 'are', 'such', 'a', 'bad', 'talker'],
        ['It', 'is', 'impolite', 'to', 'disturb', 'my', 'work'],
        ['It', 'is', 'foolish', 'to', 'buy', 'cheap', 'dog', 'food'],
        ['I', 'really', 'can', 'not', 'communicate', 'with', 'you']
    ]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


# 求并集
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


# 词集模型 （当遇到一个单词时 把对应的词向量值设为1）
# 输入的数据在文档(训练数据的并集)中是否出现 0 否  1 是 组成的矩阵
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('the word:%s is not in my Vocabulary !' % word)
    return returnVec


# 词袋模型
# 当遇到一个单词时 对应的词向量值+1
def pacakgeOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for input in inputSet:
        if input in vocabList:
            returnVec[vocabList.index(input)] += 1
    return returnVec


# 贝叶斯分类训练函数
def trainNB0(trainMatrix, trainCateGory):
    numTrianDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCateGory) / float(numTrianDocs)
    # 初始化一个所有元素为0 的maskedArray
    # 优化：初始化maskedArray所有元素为1
    # p0Num = zeros(numWords);p1Num = zeros(numWords)
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    # 优化：初始化所有的分母为2
    # p0Denom = 0.0; p1Denom =0.0
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrianDocs):
        # 计算非正常言论单词数量
        if trainCateGory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        # 正常言论单词数量
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 优化值下溢出（过于小 四舍五入后等于0）
    # p1Vect = p1Num / p1Denom
    # p0Vect = p0Num / p0Denom
    # 原理： log(f(x)) 随着 f(x)增大而增大 证明过程如下：
    # 当x1 > x2 证明 ln(x1) > ln(x2)
    # ∵ x1 > x2
    # ∴ x1 = x2 * Δx(Δx > 1)
    #   ln(x1) = ln(x2 * Δx) = lnx2 + lnΔx
    # ∵ Δx > 1
    # ∴ lnΔx > 0 即 lnx1 - lnx2 = lnΔx > 0
    # ∴ 当x1 > x2 ln(x1) > ln(x2)
    # 得证
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)
    return p1Vect, p0Vect, pAbusive


# 朴素贝叶斯分类函数
# 词集/词袋  0类型数据的概率分布  1类型数据的概率分布
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass):
    p1 = sum(vec2Classify * p1Vec) + log(pClass)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass)
    if p1 > p0:
        return 1
    else:
        return 0


# 准备文本数据
def txtData():
    mySent = 'this book is the best book on python or M.L. I have laied eyes upon.'
    # 正则切分(按照所有非数字、字母的符号来切分)
    listOfTokens = re.split('\\W*', mySent, flags=re.I)
    # 返回长度大于0的内容  且全部用小写表示
    print([token.lower() for token in listOfTokens if len(token) > 0])


# 算法测试
# 文件解析及完整的垃圾邮件测试函数
def textParse(bigString):
    listOfTokens = re.split(r'\W*', bigString)
    return [token.lower() for token in listOfTokens if len(token) > 2]


# 测试
def spamTest():
    # 一个单词一个元素
    docList = []
    # 文档数量组成的矩阵[1,1,1,1....]
    classList = []
    # 一个txt文档一个元素
    fullText = []
    # spam 文件夹中的单词分类为1  ham文件夹中单词分类为0
    for i in range(1, 26):
        worldList = textParse(open('email/spam/%d.txt' % i, ).read())
        docList.append(worldList)
        fullText.extend(worldList)
        classList.append(1)
        worldList = textParse(open('email/ham/%d.txt' % i, ).read())
        docList.append(worldList)
        fullText.extend(worldList)
        classList.append(0)
    # 求出所有文件中单词的并集
    vocabList = createVocabList(docList)
    trainingSet = range(50)
    testSet = []
    # 从所有的数据中取10个作为测试数据
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        # 分离训练数据和测试数据
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = [];
    trainClasses = []
    for doc in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[doc]))
        trainClasses.append(classList[doc])
    # 训练数据概率分布
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        # 把根据训练集的分类器得出的结果和测试集的正真的分类结果作比较  一致则认为测试通过  否则测试失败
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is:', float(errorCount) / len(testSet))


# RSS源分类器及高频词去除函数
# 返回数量排名前30的词汇
def calcMostFreq(vocabList, fullText):
    freqDict = {}
    for token in vocabList:
        # 计算fulltext各词汇的数量
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]


# feed1 和 feed0是RSS源
# feed1 = feedParser.parse('http://newyork.craigslist.rog/stp/index.rss')
# feed0 = feedParser.parse('http://sfbay.craigslist.rog/stp/index.rss')
def localWords(feed1, feed0):
    docList = []
    classList = []
    fullText = []
    # 选择较小的那个长度
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    # 数据准备
    for i in range(minLen):
        worldList = textParse(feed1['entries'[i]['summary']])
        docList.append(worldList)
        fullText.extend(worldList)
        classList.append(1)
        worldList = textParse(feed0['entries'[i]['summary']])
        docList.append(worldList)
        fullText.extend(worldList)
        classList.append(0)
    vocabList = createVocabList(docList)
    # 使用频率排名前30的单词
    top30Words = calcMostFreq(vocabList, fullText)
    # 从并集中删除 使用频率排名前30的单词
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    # 训练集大小
    trainingSet = range(2 * minLen)
    testSet = []
    # 总数据中取20个作为测试集
    for i in range(20):
        # 测试集
        randIndex = int(random, random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        # 删除总数据中的测试集
        del (trainingSet[randIndex])
    trainMat = []
    trainClass = []
    for docIndex in trainingSet:
        # 求训练集词袋
        trainMat.append(pacakgeOfWords2Vec(vocabList, docList[docIndex]))
        # 训练集结果类型
        trainClass.append(classList[docIndex])
    # 训练集结果分布
    p0v, p1v, pSpam = trainNB0(array(trainMat), array(trainClass))
    errorCount = 0
    for docIndex in testSet:
        # 测试集的词袋
        wordVector = pacakgeOfWords2Vec(vocabList, docList[docIndex])
        # 训练结果对比真实结果
        if classifyNB(array(wordVector), p0v, p1v, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is:', float(errorCount) / len(testSet))
    return vocabList, p0v, p1v


# 最具表征性的词汇现实函数
def getTopWords(ny, sf):
    vocabList, p0v, p1v = localWords(ny, sf)
    topNY = []
    topSF = []
    # 求出p0v和p1v中概率 >-6.0的数据和概率
    for i in range(len(p0v)):
        if p0v[i] > -6.0: topSF.append(vocabList[i], p0v[i])
        if p1v[i] > -6.0: topNY.append(vocabList[i], p1v[i])
    sortedSf = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    for item in sortedSf:
        print(item[0])
    sortedNy = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    for item in sortedNy:
        print(item[0])


if __name__ == '__main__':
    postingList, classVec = loadDataSet();
    vocabList = createVocabList(postingList)
    # trainMat = []
    # for post in postingList:
    # trainMat.append(setOfWords2Vec(vocabList, post))
    # 侮辱性言论分布概率   正常言论分布概率  两种类别占比
    # p1Vect, p0Vect, pAbusive = trainNB0(trainMat, classVec)
    # print(vocabList[list(p1Vect).index(max(p1Vect))])
    # print(vocabList[list(p0Vect).index(max(p0Vect))])
    # print(p1Vect)
    # print(p0Vect)
    # print(pAbusive)
    # testEntry = ['Alaskan', 'talker', 'It']
    # thisDoc = array(setOfWords2Vec(vocabList, testEntry))
    # print(testEntry, 'classified as :', classifyNB(thisDoc, p0Vect, p1Vect, pAbusive))
    # testEntry = ['really', 'disturb']
    # thisDoc = array(setOfWords2Vec(vocabList, testEntry))
    # print(testEntry, 'classified as :', classifyNB(thisDoc, p0Vect, p1Vect, pAbusive))
    # print(pacakgeOfWords2Vec(vocabList, postingList[0]))
    # txtData()
