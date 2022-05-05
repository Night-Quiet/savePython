"""
 目录 ------------------------------------------------------
 * numpy数组使用方式
 * pandas数组使用方式
 * pandas excel使用方式
 * Matplotlib绘图使用方式
 * Matplotlib数据可视化
 * 股票数据读取
 * 新版使用方式, 不到万不得已不使用
 * K线制作 老版操作
 * K线制作 新版操作
 * 简单一元回归
 * 简单二元回归
 * 线性回归模型评估
 * 符号函数
 * 训练分类
"""

# numpy数组使用方式
"""
import numpy as np
# 一维数组
oneDim_np_Simple = np.array([1, 2, 3, 4, 5])
# 1x3 按顺序 5-10 步长2 整数 数组
oneDim_np_medium = np.arange(5, 10, 2)
# 随机数数组
# 1x3 正态分布 数组
oneDim_np_normal = np.random.randn(3)
# 1x3 大小0-1 小数 数组
oneDim_np_random = np.random.rand(3)
# 1x5 大小0-10 整数 数组
oneDim_np_randint = np.random.randint(0, 10, 5)

# 二维数组
twoDim_np_Simple = np.array([[1, 2], [3, 4], [5, 6]])
# 3x4 按顺序 1-12 数组
twoDim_np_medium = np.arange(12).reshape(3, 4)
# 随机数数组
# 5x4 大小0-10 整数 数组
twoDim_np_randInt = np.random.randint(0, 10, (5, 4))
# 5x4 正态分布 数组
twoDim_np_normal = np.random.randn(5, 4)
# 5x6 大小3-4  小数 数组
twoDim_np_randFloat = np.random.uniform(3, 4, (5, 6))
"""
# pandas数组使用方式
"""
import pandas as pd
# 一维数组
oneDim_pd = pd.Series([['王1', '王2', '王3'], ['王4', '王5', '王6']])
# 二维数组
twoDim_pd = pd.Series([['王1', '王2', '王3'], ['王4', '王5', '王6']])

# 标签更换数组
twoDim_pd_one = pd.DataFrame([[1, 2], [3, 4], [5, 6]], columns=['date', 'score'], index=['A', 'B', 'C'])
twoDim_pd_two = pd.DataFrame({'date': [1, 3, 5], 'score': [2, 4, 6], 'what': [3, 5, 7]}, index=['A', 'B', 'C'])
# 转置
# 较为垃圾
twoDim_pd_transSimple = pd.DataFrame.from_dict({'date': [1, 3, 5], 'score': [2, 4, 6]}, orient="index")
# 较为正常
twoDim_pd_transMedium = twoDim_pd_two.T
# 修改索引值
twoDim_pd_rename = twoDim_pd_two.rename(index={'A': '活下去', 'B': '不行', 'C': '不行也得行'},
                                        columns={'date': '信仰', 'score': '是什么', 'what': '是现在'})
# 改变数据索引
twoDim_pd_two.rename(index={'A': '活下去', 'B': '不行', 'C': '不行也得行'},
                     columns={'date': '信仰', 'score': '是什么', 'what': '是现在'},
                     inplace=True)
# 将一列变成行索引
twoDim_pd_two_setIndex = twoDim_pd_two.set_index('信仰')
# 直接改为数字索引,行索引变成新列
twoDim_pd_two_setInt = twoDim_pd_two.reset_index()
"""
# pandas excel使用方式
"""
import pandas as pd
# 文件读取
'''
# sheet_name: 获取第几个工作表; encoding: 指定编码方式(utf-8,gbk避免中文乱码); index_col: 设置某一列为行索引; delimiter: 指定csv分割符号,默认逗号
# 读取 .xlsx .xls文件: read_excel; csv: read_csv
data = pd.read_excel('D:\\作业\\比赛\\美赛\\预处理数据.xls', sheet_name=0, encoding='utf-8')
# data = pd.read_csv('D:\\作业\\比赛\\美赛\\美赛题目\\2021_ICM_Problem_D_Data\\2021_ICM_Problem_D_Data\\data_by_artist.csv',
#                    delimiter=',', encoding='utf-8')
'''
# 文件写入
'''
data = pd.DataFrame([[1, 4, 7], [2, 8, 14], [3, 12, 21]], columns=['A数据', 'B数据', 'C数据'], index=['one', 'two', 'three'])
# 写csv: to_csv; 乱码解决: encoding="utf_8_sig"
# data.to_excel('C:\\Users\\YeShenRen\\Desktop\\python.play\\数据\\大数据分析\\data_testOne.xlsx')
# 写法2
# data.to_excel(r'C:\\Users\\YeShenRen\Desktop\python.play\数据\大数据分析\data_testOne.xlsx')
'''
# 数据读取
'''
# 获取行数据
# 1-2 行 数据 推荐 数字索引
data_line12_iloc = data.iloc[1:3]
# 1 3 行 数据 推荐 字符串索引
data_line13_iloc = data.loc[['one', 'three']]
# 1-5 行数据
data_line15 = data.head()
# 1-2 行 数据
data_line12 = data[1:3]
# -1 行 数据
data_line3_iloc = data.iloc[-1]

# 获取列数据
data_column13 = data[['A数据', 'C数据']]
'''
# 数据筛选
'''
# data = pd.DataFrame([[1, 4, 7], [2, 8, 14], [3, 12, 21]], columns=['A数据', 'B数据', 'C数据'], index=['one', 'two', 'three'])
data_screen = data[data['A数据'] > 2]
data_screen_two = data[(data['A数据'] > 2) & (data['C数据'] > 2)]
'''
# 数据分析
'''
# data = pd.DataFrame([[1, 4, 7], [2, 8, 14], [3, 12, 21]], columns=['A数据', 'B数据', 'C数据'], index=['one', 'two', 'three'])
# size
data_size = data.shape
# 数据个数 平均值 标准差 最小值 25分位数 50分位数 75分位数 最大值
data_desc = data.describe()
# 每种数据出现频次
data_vCount = data.value_counts()
'''
# 数据运算 排序 删除
'''
data = pd.DataFrame([[1, 4, 7], [2, 8, 14], [3, 12, 21]], columns=['A数据', 'B数据', 'C数据'], index=['one', 'two', 'three'])
# 增数据
# 增列数据
data['D数据'] = data['A数据'] * data['B数据']
# 增行数据
data.loc['four'] = data.iloc[0] + data.iloc[1]
# 排序 by: 按哪列数据 ascending: False降序
data_sort = data.sort_values(by='C数据', ascending=False)
# 排序 按索引升序排 需数字
data_sort_index = data.index()
# 删数据
# 模板 index: 删除行; columns: 删除列; inplace: True 删除原表
# data.drop(index=None, columns=None, inplace=False)
# 删行数据
data.drop(index=['four'], inplace=True)
# 删列数据
data.drop(columns=['D数据'], inplace=True)
'''
# 数据表拼接
'''
data = pd.DataFrame([[1, 4, 7], [2, 8, 14], [3, 12, 21]], columns=['A数据', 'B数据', 'C数据'], index=['one', 'two', 'three'])
data_other = pd.DataFrame([[1, 4, 7], [2, 8, 14], [3, 12, 21]],
                          columns=['A数据', 'D数据', 'E数据'],
                          index=['one', 'four', 'five'])
# 将数据行索引生成一列, 列名: index
data.reset_index(inplace=True)
data_other.reset_index(inplace=True)
# 默认共有列内容合并
data_merge = pd.merge(data, data_other)
# 指定列合并
data_merge_specif = pd.merge(data, data_other, on='index')
# 合并 交集
data_merge_intersect = pd.merge(data, data_other, on='index', how='outer')
# 合并 保留左表所有内容 右同理
data_merge_left = pd.merge(data, data_other, how='left')
# 合并 行 合并
data_merge_line = pd.merge(data, data_other, left_index=True, right_index=True)
# 类操作 join
data_merge_join = data.join(data_other, lsuffix='_x', rsuffix='_y')
# 类操作 concat 全连接合并, 不对重复进行合并 axis: 0行合并 1列合并
data_merge_concat = pd.concat([data, data_other], axis=0)
# 增行
# 直接增加一个表的行
data_append_table = data.append(data_other)
# 一行一行增加 ignore_index=True: 忽略原索引
data_append_line = data.append({'index': 'four', 'A数据': '4', 'B数据': '16', 'C数据': '28'}, ignore_index=True)
'''
"""
# Matplotlib绘图使用方式
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# 折线图
'''
x1 = np.arange(1, 10, 1)
y1 = x1 + 1
plt.plot(x1, y1)
y2 = x1 * 2
plt.plot(x1, y2, color='red', linewidth=3, linestyle='--')
plt.show()
'''
# 柱形图
'''
x = np.arange(1, 10, 1)
y = np.arange(10, 1, -1)
plt.bar(x, y)
plt.show()
'''
# 散点图
'''
x = np.random.rand(10)
y = np.random.rand(10)
plt.scatter(x, y)
plt.show()
'''
# 直方图
'''
data = np.random.randn(1000)
# 40 颗粒度-柱形数量; 黑色柱形边框颜色
plt.hist(data, bins=40, edgecolor='black')
plt.show()
# 第2种方式
# data = np.random.randn(1000)
# df = pd.DataFrame(data)
# df.hist(bins=40, edgecolor='black')
# # df.plot(kind='hist')
# plt.show()
'''
# 联合作图
'''
df = pd.DataFrame([np.random.rand(3), np.random.rand(3), np.random.rand(3)],
                  columns=['A数据', 'B数据', 'C数据'], index=['A', 'B', 'C'])
df['A数据'].plot(kind='line')
df['A数据'].plot(kind='bar')
plt.show()
'''
# 饼状图
'''
df = pd.DataFrame([np.random.rand(3), np.random.rand(3), np.random.rand(3)],
                  columns=['A数据', 'B数据', 'C数据'], index=['A', 'B', 'C'])
df['A数据'].plot(kind='pie')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.show()
'''
# 箱体图
'''
df = pd.DataFrame([np.random.rand(3), np.random.rand(3), np.random.rand(3)],
                  columns=['A数据', 'B数据', 'C数据'], index=['A', 'B', 'C'])
df['A数据'].plot(kind='box')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.show()
'''
"""
# Matplotlib数据可视化
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# x = np.random.randn(1000)
x = np.linspace(-10, 10, 10000)
data_normal = np.random.randn(10000)
data_normal_another = np.random.randn(10000)
data_random = np.random.uniform(0, 10, 10000)
# 中文乱码解决 SimHei:黑体; Microsoft YaHei; 宋体: SimSun; 新宋体: NSimSun; 仿宋: FangSong; 楷体: KaiTi; 细明体: MingLiU; 新细明体: PMingLiU
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 绘制多图
'''
# 第1个参数为多少行,第2个参数为多少列,第3个参数为序号
ax1 = plt.subplot(221)
plt.pie(data_random)
ax2 = plt.subplot(222)
plt.boxplot(data_normal)
ax3 = plt.subplot(223)
plt.scatter(data_normal_another, data_normal)
ax4 = plt.subplot(224)
plt.hist(data_normal, bins=40)
# # 第2种画法
# # 两行两列4个图,图像大小为10:8; fig: 画布; axes: 子图集合
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
# 获得子图
# ax1, ax2, ax3, ax4 = axes.flatten()
# ax1.pie(data_random)
# ax2.boxplot(data_normal)
# ax3.scatter(data_normal_another, data_normal)
# ax4.hist(data_normal, bins=40)
'''

plt.hist(data_normal, bins=40, edgecolor='black', label="正态分布")
# 添加图例 需要有label
plt.legend(loc='upper left')
# 设置双坐标轴
# plt.twinx()
plt.hist(data_random, bins=40, edgecolor='blue', label="随机分布")
plt.legend(loc='upper right')
# 设置图表大小
plt.rcParams['figure.figsize'] = (8, 6)
# 标题
plt.title('正态玩法')
# 横纵轴说明
plt.xlabel('X')
plt.ylabel('Y')

plt.show()
"""
# 股票数据读取
import tushare as ts
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import date2num as dn
import numpy as np
import datetime as dt
import mplfinance as mpf
import seaborn as sns
# 新版使用方式, 不到万不得已不使用
'''
pro = ts.pro_api()
df = pro.daily(ts_code='000001.SZ', start_date='200901101', end_date='20191101' )
print(df.head())
'''
'''
# 获取数据
df = ts.get_k_data('000002', start='2009-01-01', end='2019-01-01')
# 获取数据前5行
data = df.head()
# 数据的date提取年份处理
df['date'] = df['date'].apply(lambda x: dt.strptime(x, '%Y-%m-%d'))
plt.plot(df['date'], df['close'])
plt.show()
'''

# K线制作 老版操作
'''
df = ts.get_k_data('000002', start='2009-01-01', end='2019-01-01')


def date_to_num(dates):
    num_time = []
    for date in dates:
        date_time = dt.datetime.strptime(date, '%Y-%m-%d')
        num_date = dn(date_time)
        num_time.append(num_date)
    return num_time


df_arr = df.values
df_arr[:, 0] = date_to_num(df_arr[:, 0])
df_arr_d = pd.DataFrame(df_arr)
fig, ax = plt.subplots(figsize=(15, 6))
# mpf.candlestick_ochl(ax, df_arr, width=0.6, colorup='r', colordown='g', alpha=1.0)
# 绘制网格线
plt.grid(True)
# 设置x轴的刻度格式为常规日期格式
ax.xaxis_date()
plt.show()
'''
# K线制作 新版操作
'''
df = ts.get_k_data('000002', start='2019-01-01', end='2019-04-01')
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df.set_index('date', inplace=True)
# 画对比线
addplot = mpf.make_addplot(df['close'])
# style: 三价图; addplot: 对比线; vulume: 成交量对比; mav: 均线; type: 箱线图
mpf.plot(df, style='charles', type='candle', addplot=addplot, volume=True, mav=(2, 5, 10))
# 填均线
df['MA5'] = df['close'].rolling(5).mean()
df['MA10'] = df['close'].rolling(10).mean()
plt.plot(df.index.values, df['MA5'])
plt.plot(df.index.values, df['MA10'])

# 网格线
plt.grid()

# 标题 标签
plt.title('万科A')
plt.xlabel('日期')
plt.ylabel('价格')

plt.show()
'''
# 一元线性回归函数
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
# 简单一元回归
'''
X = [[1], [2], [4], [5]]
Y = [2, 4, 6, 8]
regr = LinearRegression()
# 训练
regr.fit(X, Y)
# 预测
y = regr.predict([[1.5], [2.5], [4.5]])
plt.scatter(X, Y)
plt.plot(X, regr.predict(X))
plt.show()
# 一元系数 截距
print('系数a: ' + str(regr.coef_[0]))
print('截距b: ' + str(regr.intercept_))
'''
# 简单二元回归
'''
# 最高项设置为2
poly_reg = PolynomialFeatures(degree=2)
X = [[1], [2], [4], [5]]
Y = [2, 4, 6, 8]
# 原有的X转换为一个新的二维数组X_
X_ = poly_reg.fit_transform(X)
regr = LinearRegression()
# 训练
regr.fit(X_, Y)
# 预测
# y = regr.predict([[1.5], [2.5], [4.5]])
plt.scatter(X, Y)
plt.plot(X, regr.predict(X_), color='red')
plt.show()
# 一元系数 截距
# 1:常数项系数:无影响; 2:一次项系数; 3:二次项系数
print('系数a: ' + str(regr.coef_))
# 常数项系数
print('截距b: ' + str(regr.intercept_))
'''
# 线性回归模型评估
'''
import statsmodels.api as sm
X = [[1], [2], [4], [5]]
Y = [2, 4, 6, 8]
X2 = sm.add_constant(X)
est = sm.OLS(Y, X2).fit()
print(est.summary())
'''
# 逻辑回归模型
# 符号函数
'''
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-6, 6)
y = 1.0 / (1.0 + np.exp(-x))
plt.plot(x, y)
plt.show()
'''
# 训练分类
'''
from sklearn.linear_model import LogisticRegression
X = [[1, 0], [5, 1], [6, 4], [4, 2], [3, 2]]
Y = [0, 1, 1, 0, 0]
model = LogisticRegression()
model.fit(X, Y)  # fit训练
result = 
'''