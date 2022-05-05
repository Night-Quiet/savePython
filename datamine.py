"""
 目录 ------------------------------------------------------
 * 数据的基本统计描述-集中趋势
 * 计算每个维度的均值
 * 计算每个维度的中位数
 * 求众数方法一（使用numpy）
 * 求众数方法二（直接利用SciPy下的stats模块）
 * 数据的统计描述-离散趋势
 * 求极差方法一
 * 求极差方法二
 * 求四分位数（利用np.percentile()函数）
 * 四分位数极差（四分位距）
 * 五数概括
 * 箱线图
 * 方差和标准差
 * DataFrame描述性统计
 * 数据的基本统计描述-基本统计图
 * 条形图
 * 饼状图
 * 折线图
 * 直方图
 * 散点图
 * 分位数-分位数图
 * 雷达图
 * 设置rc参数显示中文标题，设置字体为SimHei显示中文
 * 数组拼接
 * 《政府工作报告2019》
 * 欧氏距离
 * 曼哈顿距离
 * 明可夫斯基距离
 * 切比雪夫距离
 * 测试，关联分析
 * 1、决策树
 * 2、朴素贝叶斯
 * 3、K近邻分类
 * 4、模型评价
 * 1、线性回归
 * 2、回归树
 * 3、k近邻回归
 * 1、KMeans
 * 2、层次聚类之凝聚聚类
 * 3、DBSCAN
"""


# 第二章

# 数据的基本统计描述-集中趋势
"""
import numpy as np
from SkLearn.DataSets import load_iris
iris_data = load_iris()     # 调用load函数，返回数据在iris_data中
print(iris_data.data)     # 数据
print(iris_data.data.shape)       # 数据维度
print(iris_data.data[0, :])       # 获得第一条数据
print(iris_data.data[:, 0])       # 获得第一个维度的所有值，作为一维向量
print(iris_data.data[:, np.NewAxis, 0])       # 获得第一个维度的所有值，作为二维数组
"""
# 计算每个维度的均值
"""
print(np.mean(iris_data.data, axis=0))      # axis=0为竖向求（维度），axis=1为横向求（各个数组）
"""
# 计算每个维度的中位数
"""
print(np.median(iris_data.data, axis=0))
"""
# 求众数方法一（使用numpy）
"""
import random
data = [random.choice(range(1, 5)) for i in range(100)]     # 随机产生100个数，范围为1-4
# print(data)
counts = np.BinCount(data)      # 记录0-该数组最大值（4）出现的次数，并返回到一维数组中
# print(counts)
np.ArgMax(counts)       # 返回数组最大值对应的索引
# print(np.ArgMax(counts))
"""
# 求众数方法二（直接利用SciPy下的stats模块）
"""
from SciPy import stats
stats.mode(data)      # 返回两个维度，众数，以及众数出现的次数
print(stats.mode(data))
stats.mode(data)[0][0]      # 单独返回众数
print(stats.mode(data)[0][0])
"""
# 数据的统计描述-离散趋势
"""
import numpy as np
from SkLearn.DataSets import load_iris
iris_data = load_iris()     # 参考第4行
feature_1 = iris_data.data[:, 0]     # 参考第8行
print(feature_1)
"""
# 求极差方法一
"""
print(feature_1.max()-feature_1.min())
"""
# 求极差方法二
"""
print(np.max(feature_1)-np.min(feature_1))
"""
# 求四分位数（利用np.percentile()函数）
"""
Q3 = np.percentile(feature_1, 0.75)     # 计算上四分位数，返回到Q3
print(Q3)
Q1 = np.percentile(feature_1, 0.25)     # 计算下四分位数，返回到Q1
print(Q1)
"""
# 四分位数极差（四分位距）
"""
IQR = Q3 - Q1
print(IQR)
"""
# 五数概括
"""
max_value = np.max(feature_1)
Q3 = np.percentile(feature_1, 0.75)
median_value = np.median(feature_1)
Q1 = np.percentile(feature_1, 0.25)
min_value = np.min(feature_1)
print([min_value, Q1, median_value, Q3, max_value])
"""
# 箱线图
"""
# 单一箱线图
'''
import MatPlotLib.PyPlot as plt
plt.BoxPlot(x=feature_1)      # 指定绘图数据导入x，必须是导入到x中
plt.yLabel('values of ' + iris_data.feature_names[0])       # 命名y轴
plt.xLabel(iris_data.feature_names[0])      # 命名x轴
plt.show()
print(iris_data.feature_names)        # feature_names的实体
'''
# 多个箱线图
'''
from pandas import DataFrame
iris_df = DataFrame(iris_data.data, columns=iris_data.feature_names)        # 将多维数组装化成字典，每个维度归类于导入columns的数组的对应名称
fig, axes = plt.subplots(1, 4)      # 参数1，4代表子图的行数和列数，函数返回主图像到fig和子图像集到axes
iris_df.plot(kind='box', ax=axes, subplots=True, title='All feature boxplots')
# kind为选择作图样式（此为箱线图），ax为将要绘制的对象，subplots为确定图片是否有子图，title为图片的标题
axes[0].set_yLabel(iris_df.columns[0])      # 设置第一个子图y轴名称
axes[1].set_yLabel(iris_df.columns[1])      # 第二个
axes[2].set_yLabel(iris_df.columns[2])      # 第三个
axes[3].set_yLabel(iris_df.columns[3])      # 第四个
fig.subplots_adjust(wSpace=1, hSpace=1)     # 设置子图之间保留的宽度和高度
plt.show()
'''
"""
# 方差和标准差
"""
# 方差
'''
var = np.var(feature_1)     # 计算方差返回到var
print(var)
'''
# 所有维度的方差
'''
var_all = np.var(iris_data.data, axis=0)        # 计算各维度方差组成一维数组返回到var_all，axis参考第12行
print(var_all)
'''
# 标准差
'''
std = np.std(feature_1)     # 计算标准差返回到std
print(std)
'''
"""
# DataFrame描述性统计
"""
print(iris_df.describe())       # 返回各维度的和，均值，标准差，最小值，下四分位数，中位数，上四分位数，最大值
"""

# 第三章

# 数据的基本统计描述-基本统计图
"""
import numpy as np
import MatPlotLib.PyPlot as plt
from SciPy import stats
from skLearn.DataSets import load_iris
iris_data = load_iris()
sample_1 = iris_data.data[0, :]
"""
# 条形图
"""
p1 = plt.bar(range(1, len(sample_1)+1),     # bar代表条形图，range生成y轴下标
             height=sample_1,       # 条形的高度分别对应为sample_1内的值
             tick_label=iris_data.feature_names,        # x轴的下标的标签
             width=0.3)     # 条形的宽度设置
plt.yLabel('cm')        # y轴名称
plt.title('bar of first data')      # 图像标题
plt.show()
"""
# 饼状图
"""
labels = 'Sunny', 'Windy', 'FroGy', 'Snow'
sizes = [15, 30, 45, 10]
explode = (0, 0.1, 0, 0)        # 引用会突出第二部分饼状图
fig1, ax1 = plt.subplots()      # 无参数默认为无分图
ax1.pie(x=sizes, explode=explode, labels=labels, AutoPct='%1.1f%%', shadow=True, startangle=90)
# x为数据，explode为突出各部分比例，labels为饼状图标签，AutoPct为数据保留，此为保留一位小数且数长度为>1
# shadow为是否显示阴影，StartAngle为开始绘图的角度
ax1.axis('equal')       # x轴与y轴的单位长度相同
plt.show()
"""
# 折线图
"""
x = np.arAnge(0, 5, 0.1)        # 生成一维数组0-5，间隔0.1，不包括5，返回到x
y = np.sin(x)       # 生成一维数组x对应y，返回到y
plt.plot(x, y)
plt.show()
"""
# 直方图
"""
iris_data = load_iris()     # 参考第5行
feature_2 = iris_data.data[:, 1]        # 参考第9行
plt.hist(feature_2, bins=10, alpha=0.5)        # bins指定条状的个数，alpha指定该图透明度(范围0-1)
plt.show()
"""
# 散点图
"""
feature_1 = iris_data.data[:, 0]        # 参考第9行
feature_3 = iris_data.data[:, 2]        # 同上
plt.scatter(x=feature_1, y=feature_3)       # 对应x，y作点
plt.show()
"""
# 分位数-分位数图
"""
res = stats.ProbPlot(feature_3, plot=plt)
# 生成样本数据相对于指定理论分布（默认正态分布）的分位数概率图
# 网站参考：https://stackoom.com/question/3FrEs/%E5%A6%82%E4%BD%95%E8%A7%A3%E9%87%8Ascipy-stats-probplot%E7%BB%93%E6%9E%9C
plt.show()
"""

# 第四章

# 雷达图
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams['font.sans-serif'] = ['SimHei']
"""
# 设置rc参数显示中文标题，设置字体为SimHei显示中文
"""
labels = np.array(['语文', '数学', '英语', '物理', '化学', '生物'])
n_attr = len(labels)
scores = np.array([88.7, 85, 90, 95, 70, 96])       # 将列表转化成数组
angles = np.linspace(start=0, stop=2*np.pi, num=n_attr, endpoint=False)
# start为序列起始点，stop为序列结束点，num为生成样本数，endpoint=False时，序列由num+1除去最后一个后等距采样组成
scores = np.concatenate((scores, [scores[0]]))
"""
# 数组拼接
"""
angles = np.concatenate((angles, [angles[0]]))
# 同上
fig = plt.figure(facecolor='white')     # 图像背景颜色设置
plt.subplot(111, polar=True)        # 1行1列的第一个图的绘制，True表示绘制极坐标图
plt.plot(angles, scores, 'bo-', color='g', linewidth=2)
# 'bo-'为作图风格，LineWidth为作图线宽
plt.fill(angles, scores, facecolor='g', alpha=0.2)
# 填充函数曲线与坐标轴之间的区域，alpha为颜色透明度（范围0-1）
plt.thetagrids(angles*180/np.pi, labels)
# 区域划分同时命名
plt.title('成绩雷达图', ha='center')
plt.grid(True)      # 增加网格，其余设置默认
plt.show()
"""
# 《政府工作报告2019》
"""
import jieba
from wordcloud import WordCloud
import numpy as np
import PIL
import os


def chinese_jieba(txt):
    wordlist_jieba = jieba.cut(txt)     # 将⽂本分割，返回列表
    txt_jieba = " ".join(wordlist_jieba)      # 将列表拼接为以空格为间断的字符串
    return txt_jieba


stopwords = {'这些':0, '那些':0, '因为':0, '所以':0}        # 欲删除词汇
alice_mask = np.array(PIL.Image.open('air.jpg'))        # 填充的背景图片
with open('./政府工作报告.txt', encoding='utf8') as fp:       # ./表示和运行文件同一个文件夹，.../表示当前文件的上一级目录
    txt = fp.read()     # 读取文本
    cutted_text = chinese_jieba(txt)        # 文本分割返回
    # print(txt)
    wordcloud = WordCloud(font_path = r'C:\Windows\Fonts\STKAITI.TTF',      # 字体
        background_color='white',     # 背景⾊
        max_words=80,     # 最⼤显示单词数
        max_font_size=80,     # 频率最⼤单词字体⼤⼩
        mask=alice_mask,        # 词云形状填充
        stopwords=stopwords     # 过滤停⽤词
        ).generate(cutted_text)     # 采用文本
    image = wordcloud.to_image()        # 词云输出导入image中
    image.show()
"""
# 计算数值属性的三种距离
"""
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
iris_data = load_iris()
sample_1 = iris_data.data[0, :]
sample_2 = iris_data.data[1, :]
"""
# 欧氏距离
"""
def euclidean_distance(vec_1, vec_2):
    return np.linalg.norm(vec_1 - vec_2, ord=2)
# 计算x=(vec_1-vec_2)下√(x1^2+x2^2+...+xn^2)
euclidean_dist = euclidean_distance(sample_1, sample_2)
print(euclidean_dist)
"""
# 曼哈顿距离
"""
def manhattan_distance(vec_1, vec_2):
    return np.linalg.norm(vec_1 - vec_2, ord=1)
# 计算x=(vec_1-vec_2)下|x1|+|x2|+...+|xn|
manhattan_dist = manhattan_distance(sample_1, sample_2)
print(manhattan_dist)
"""
# 明可夫斯基距离
"""
def minkowski_distance(vec_1, vec_2, ord=3):
    return np.linalg.norm(vec_1 - vec_2, ord=ord)
# 计算x=(vec_1-vec_2)下(|x1|^ord+|x2|^ord+...+|xn|^ord)^(1/ord)
minkowski_dist = minkowski_distance(sample_1, sample_2, ord=3)
print(minkowski_dist)
"""
# 切比雪夫距离
"""
def chebyshev_distance(vec_1, vec_2, ord=np.inf):
    return np.linalg.norm(vec_1-vec_2, ord=np.inf)
# 计算x=(vec_1-vec_2)下max(|xi|)
chebyshev_dist = chebyshev_distance(sample_1, sample_2, ord=np.inf)
print(chebyshev_dist)
"""
# 测试，关联分析
"""
from efficient_apriori import apriori
transactions = [('eggs', 'bacon', 'soup'),
 ('eggs', 'bacon', 'apple'),
 ('soup', 'bacon', 'banana')]
itemsets, rules = apriori(transactions, min_support=0.5, min_confidence=1)
# 第一空为数据集，min_support为最小支持度，min_confidence为最小置信度
print(rules)    # [{eggs} -> {bacon}, {soup} -> {bacon}]
print(itemsets)
rules_rhs = filter(lambda rule: len(rule.lhs) == 2 and len(rule.rhs) == 1, rules)
"""

# 课中例子
"""
from efficient_apriori import apriori
transactions = [('A', 'C', 'D'),
                ('B', 'C', 'E'),
                ('A', 'B', 'C', 'E'),
                ('B', 'E')]
itemsets, rules = apriori(transactions, min_support=0.5, min_confidence=1)
print(itemsets)
print(rules)
"""
# 1、决策树
"""
# 1.1、简单例子
from sklearn import tree
X = [[0, 0], [1, 1]]
Y = [0, 1]
dt_test_clf = tree.DecisionTreeClassifier()
dt_test_clf = dt_test_clf.fit(X, Y)
print(dt_test_clf.predict([[2., 2.]]))
print(dt_test_clf.predict_proba([[2., 2.]]))

# 1.2、iris数据集
# 1.2.1、训练模型
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
dt_clf = tree.DecisionTreeClassifier()
dt_clf = dt_clf.fit(iris.data, iris.target)

# 1.2.2、模型预测
y_pred = dt_clf.predict(iris.data)
print(y_pred)

# 1.2.3、绘图
import graphviz
dot_data = tree.export_graphviz(dt_clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render(filename="iris")
graph.view()
"""
# 2、朴素贝叶斯
"""
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
iris = datasets.load_iris()
gnb = GaussianNB()

#  训练模型
gnb.fit(iris.data, iris.target)

#  进⾏预测
y_pred = gnb.predict(iris.data)
print(y_pred)
"""
# 3、K近邻分类
"""
from sklearn import neighbors, datasets
iris = datasets.load_iris()

#  指定近邻个数
n_neighbors = 15
# weights  可选： 'uniform', 'distance'
weights = 'distance'
knn_clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
knn_clf.fit(iris.data, iris.target)
knn_pre_y = knn_clf.predict(iris.data)
print(knn_pre_y)
"""
# 4、模型评价
"""
# 准备数据+训练模型
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn import metrics
#  准备数据
iris = datasets.load_iris()
X = iris.data
y = iris.target
#  取出第⼀类和第⼆类
X, y = X[y != 2], y[y != 2]
n_samples, n_features = X.shape
print(n_samples)
#  加⼊噪声特征
random_state = np.random.RandomState(0)
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
#  随机选取 80 个样本作为训练集，其余作为测试集
t = np.array(range(100))
np.random.shuffle(t)
train_idx = t >= 20
train_X = X[train_idx, :]
train_y = y[train_idx]
text_X = X[~train_idx, :]
test_y = y[~train_idx]
svc_clf = svm.SVC(kernel='linear', probability=True, random_state=random_state)
svc_clf = svc_clf.fit(train_X, train_y)

# 4.1、准确率
y_pre = svc_clf.predict(text_X)
print(metrics.accuracy_score(test_y, y_pre))

# 4.2、混淆矩阵
cnf_matrix = metrics.confusion_matrix(test_y, y_pre)
print(cnf_matrix)

# 4.3、ROC曲线
from sklearn.metrics import roc_curve, auc
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
probas_ = svc_clf.predict_proba(text_X)
fpr, tpr, thresholds = roc_curve(test_y, probas_[:, 1])
tprs.append(np.interp(mean_fpr, fpr, tpr))
tprs[-1][0] = 0.0
roc_auc = auc(fpr, tpr)
aucs.append(roc_auc)
plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC Curve (AUC = %0.2f)' % roc_auc)
plt.show()

# 4.4、交叉验证与ROC曲线
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
#  载⼊数据 只取第⼀类和第⼆类
iris = datasets.load_iris()
X = iris.data
y = iris.target
X, y = X[y != 2], y[y != 2]
n_samples, n_features = X.shape
# Add noisy features
random_state = np.random.RandomState(0)
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
#  采⽤交叉验证的⽅式训练模型
cv = StratifiedKFold(n_splits=5)
classifier = svm.SVC(kernel='linear', probability=True, random_state=random_state)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
#  统计每次结果，并绘制相应的 ROC 曲线
i = 0
for train, test in cv.split(X, y):
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
#  计算平均结果，绘制平均 ROC 曲线
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)
#  将均值线上下⼀个标准差内的区域上⾊
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
"""

# 1、线性回归
"""
# 1.1 简单例子
from sklearn import linear_model
X = [[0, 0], [1, 1], [2, 2]]
y = [0, 1, 2]
reg = linear_model.LinearRegression()
pri1 = reg.fit(X, y)
pri2 = reg.coef_
pri3 = reg.predict([[3, 3]])
print(pri1)
print(pri2)
print(pri3)

# 1.2 糖尿病数据集
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# 载入数据集
diabetes = datasets.load_diabetes()

# 只取第三个属性进行一元回归
diabetes_X = diabetes.data[:, np.newaxis, 2]
print(diabetes_X)

# 划分训练数据集和测试数据集
# 对特征进行划分
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# 对标签进行划分
diabetes_y_train = diabetes.target[:-20]
diabetes_y_text = diabetes.target[-20:]

#  创建模型对象
regr = linear_model.LinearRegression()

# 在训练集上训练模型
regr.fit(diabetes_X_train, diabetes_y_train)

# 在测试集上测试
diabetes_y_pred = regr.predict(diabetes_X_test)

# 查看回归系数
print('Coefficients: \n', regr.coef_)

# MSE
print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_text, diabetes_y_pred))

# 解释方差R^2
print('Variance score: %.2f' % r2_score(diabetes_y_text, diabetes_y_pred))

# 绘图查看预测结果
plt.scatter(diabetes_X_test, diabetes_y_text, color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()
"""
# 2、回归树
"""
# 2.1 简单例子
from sklearn import tree
X = [[0, 0], [1, 1], [2, 2]]
y = [0, 1, 2]
dt_reg_test = tree.DecisionTreeRegressor()
dt_reg_test = dt_reg_test.fit(X, y)
pri4 = dt_reg_test.predict([[1, 1]])
print(pri4)

# 2.2 生成的随机数据集
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# 生成数据集， 并加入随机误差
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))

# 用不同参数， 分别拟合模型
dt_regr_1 = DecisionTreeRegressor(max_depth=2)
dt_regr_2 = DecisionTreeRegressor(max_depth=5)
pri5 = dt_regr_1.fit(X, y)
pri6 = dt_regr_2.fit(X, y)
print(pri5)
print(pri6)

# 分别进行模型预测
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = dt_regr_1.predict(X_test)
y_2 = dt_regr_2.predict(X_test)

# 绘制结果
plt.figure()
plt.scatter(X, y, s=20, edgecolors="black", c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue", label="max_depth=2", linewidth=2)
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()
"""
# 3、k近邻回归
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors

# 生成样本数据， 并加入随机误差
np.random.seed(0)
X = np.sort(5 * np.random.rand(40, 1), axis=0)
T = np.linspace(0, 5, 500)[:, np.newaxis]
y = np.sin(X).ravel()

# 加入随机误差
y[::5] += 1 * (0.5 - np.random.rand(8))

# 拟合模型，并画图对比
n_neighbors = 5
for i, weights in enumerate(['uniform', 'distance']):
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
    y_ = knn.fit(X, y).predict(T)
    plt.subplot(2, 1, i + 1)
    plt.scatter(X, y, c='k', label='data')
    plt.plot(T, y_, c='g', label='prediction')
    plt.axis('tight')
    plt.legend()
    plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors, weights))
plt.show()
"""

# 1、KMeans
"""
# 1.1 简单例子
from sklearn.cluster import KMeans
import numpy as np
X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
pri7 = kmeans.labels_
pri8 = kmeans.predict([[0, 0], [12, 3]])
pri9 = kmeans.cluster_centers_
print(pri7)
print(pri8)
print(pri9)

# 1.2 多种情况效果展示
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
n_samples = 1500
random_state = 170
X, y = make_blobs(n_samples=n_samples, random_state=random_state)

# 给定错误聚类个数
y_pred = KMeans(n_clusters=2, random_state=random_state).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("Incorrect Number of Blobs")
plt.show()

# 各向异性分布的数据
# 旋转数据
transformation = [[0.60834549, -0.6366734], [-0.40887718, 0.85253229]]
X_aniso = np.dot(X, transformation)
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_aniso)
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred)
plt.title("Anisotropicly Distrubuted Blobs")
plt.show()

# 每簌的方差不同
X_varied, y_varied = make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_varied)
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=y_pred)
plt.title("Unequal Variance")
plt.show()

# 簌的大小分布不均衡
X_filtered = np.vstack((X[y == 0][:500], X[y == 1][:100], X[y == 2][:10]))
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_filtered)
plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=y_pred)
plt.title("Unevenly Sized Blobs")
plt.show()
"""
# 2、层次聚类之凝聚聚类
"""
# 2.1 简单例子
from sklearn.cluster import AgglomerativeClustering
import numpy as np
X = np.array([[1, 2],[1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
clustering = AgglomerativeClustering().fit(X)
print(clustering)
pri10 = clustering.labels_
print(pri10)
 
# 2.2 聚类层次树-dendrogram
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

# 随机生成数据
np.random.seed(1234)
variables = ['X', 'Y', 'Z']
labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']
X = np.random.random_sample([5, 3])*10
df = pd.DataFrame(X, columns=variables, index=labels)
print(df)
row_clusters = linkage(pdist(df, metric='euclidean'),method='complete')
print(pd.DataFrame(row_clusters, columns=['row label1', 'rowlabel2', 'distance', 'n0, of items in clust.'], index=['cluster %d' %(i+1) for i in range(row_clusters.shape[0])]))
row_dendr = dendrogram(row_clusters, labels=labels)
plt.tight_layout()
plt.ylabel('Euclidean distance')
plt.show()
"""
# 3、DBSCAN
"""
# 3.1 简单例子
from sklearn.cluster import DBSCAN
import numpy as np
X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])
clustering = DBSCAN(eps=3, min_samples=2).fit(X)
print(clustering)
pri11 = clustering.labels_
print(pri11)

# 3.2 复杂例子
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

# 生成数据
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4, random_state=0)
X = StandardScaler().fit_transform(X)

# 计算DBSCAN模型
db = DBSCAN(eps=0.3, min_samples=10).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# 统计基本结果
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print('估计的聚类个数： %d' % n_clusters_)
print('估计的噪声点个数： %d' % n_noise_)

# 绘图展示结果
import matplotlib.pyplot as plt
# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]
    class_member_mask = (labels == k)
    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=14)
    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
"""