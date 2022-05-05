"""
 目录 ------------------------------------------------------
 * 免费看剧-vip免费
 * 文档下载-百度文库
 * 历史价格查看
 * 文件隐藏到图片中

 * 校验数字的表达式
 * 校验字符的表达式
 * 特殊需求表达式

 * 获取数组或列表中最大|最小的N个数及索引
 * 每次获取数组|列表最小-并删除弹出

 * 数据增广的八种常用方式

 * 获取等间隔的数组实例

 * xlrd - 读取excel数据
 * xlwt - 写入excel数据
 * xlutils - excel追加数据

 * 错误类型

 * 读写csv文件

 * python设置字典-值为列表|字典
 * NetworkX-画图
 * 随机生成均匀分布在单位圆内的点
"""

import heapq
import xlrd
import xlwt
import xlutils.copy
import pandas as pd
from collections import defaultdict
import numpy as np
import csv
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# 免费看剧-vip免费
"""
视频地址前 + wn.run/
example: wn.run/https
"""
# 文档下载-百度文库
"""
文档地址后 + vvv
example: xxx.xxxvvv.com
"""
# 历史价格查看
"""
商品地址后 + vvv
example: xxx.xxxvvv.com
"""
# 文件隐藏到图片中
"""
copy 1.jpg/b+2.rar=3.jpg  -- bat文件
"""
# 校验数字的表达式
"""
 1 数字：^[0-9]*$
 2 n位的数字：^\d{n}$
 3 至少n位的数字：^\d{n,}$
 4 m-n位的数字：^\d{m,n}$
 5 零和非零开头的数字：^(0|[1-9][0-9]*)$
 6 非零开头的最多带两位小数的数字：^([1-9][0-9]*)+(.[0-9]{1,2})?$
 7 带1-2位小数的正数或负数：^(\-)?\d+(\.\d{1,2})?$
 8 正数、负数、和小数：^(\-|\+)?\d+(\.\d+)?$
 9 有两位小数的正实数：^[0-9]+(.[0-9]{2})?$
10 有1~3位小数的正实数：^[0-9]+(.[0-9]{1,3})?$
11 非零的正整数：^[1-9]\d*$ 或 ^([1-9][0-9]*){1,3}$ 或 ^\+?[1-9][0-9]*$
12 非零的负整数：^\-[1-9][]0-9"*$ 或 ^-[1-9]\d*$
13 非负整数：^\d+$ 或 ^[1-9]\d*|0$
14 非正整数：^-[1-9]\d*|0$ 或 ^((-\d+)|(0+))$
15 非负浮点数：^\d+(\.\d+)?$ 或 ^[1-9]\d*\.\d*|0\.\d*[1-9]\d*|0?\.0+|0$
16 非正浮点数：^((-\d+(\.\d+)?)|(0+(\.0+)?))$ 或 ^(-([1-9]\d*\.\d*|0\.\d*[1-9]\d*))|0?\.0+|0$
17 正浮点数：^[1-9]\d*\.\d*|0\.\d*[1-9]\d*$ 或 ^(([0-9]+\.[0-9]*[1-9][0-9]*)|([0-9]*[1-9][0-9]*\.[0-9]+)|([0-9]*[1-9][0-9]*))$
18 负浮点数：^-([1-9]\d*\.\d*|0\.\d*[1-9]\d*)$ 或 ^(-(([0-9]+\.[0-9]*[1-9][0-9]*)|([0-9]*[1-9][0-9]*\.[0-9]+)|([0-9]*[1-9][0-9]*)))$
19 浮点数：^(-?\d+)(\.\d+)?$ 或 ^-?([1-9]\d*\.\d*|0\.\d*[1-9]\d*|0?\.0+|0)$
"""
# 校验字符的表达式
"""
 1 汉字：^[\u4e00-\u9fa5]{0,}$
 2 英文和数字：^[A-Za-z0-9]+$ 或 ^[A-Za-z0-9]{4,40}$
 3 长度为3-20的所有字符：^.{3,20}$
 4 由26个英文字母组成的字符串：^[A-Za-z]+$
 5 由26个大写英文字母组成的字符串：^[A-Z]+$
 6 由26个小写英文字母组成的字符串：^[a-z]+$
 7 由数字和26个英文字母组成的字符串：^[A-Za-z0-9]+$
 8 由数字、26个英文字母或者下划线组成的字符串：^\w+$ 或 ^\w{3,20}$
 9 中文、英文、数字包括下划线：^[\u4E00-\u9FA5A-Za-z0-9_]+$
10 中文、英文、数字但不包括下划线等符号：^[\u4E00-\u9FA5A-Za-z0-9]+$ 或 ^[\u4E00-\u9FA5A-Za-z0-9]{2,20}$
11 可以输入含有^%&',;=?$\"等字符：[^%&',;=?$\x22]+
12 禁止输入含有~的字符：[^~\x22]+
"""
# 特殊需求表达式
"""
 1 Email地址：^\w+([-+.]\w+)*@\w+([-.]\w+)*\.\w+([-.]\w+)*$
 2 域名：[a-zA-Z0-9][-a-zA-Z0-9]{0,62}(/.[a-zA-Z0-9][-a-zA-Z0-9]{0,62})+/.?
 3 InternetURL：[a-zA-z]+://[^\s]* 或 ^http://([\w-]+\.)+[\w-]+(/[\w-./?%&=]*)?$
 4 手机号码：^(13[0-9]|14[0-9]|15[0-9]|16[0-9]|17[0-9]|18[0-9]|19[0-9])\d{8}$ (由于工信部放号段不定时，所以建议使用泛解析 ^([1][3,4,5,6,7,8,9])\d{9}$)
 5 电话号码("XXX-XXXXXXX"、"XXXX-XXXXXXXX"、"XXX-XXXXXXX"、"XXX-XXXXXXXX"、"XXXXXXX"和"XXXXXXXX)：^(\(\d{3,4}-)|\d{3.4}-)?\d{7,8}$ 
 6 国内电话号码(0511-4405222、021-87888822)：\d{3}-\d{8}|\d{4}-\d{7} 
 7 18位身份证号码(数字、字母x结尾)：^((\d{18})|([0-9x]{18})|([0-9X]{18}))$
 8 帐号是否合法(字母开头，允许5-16字节，允许字母数字下划线)：^[a-zA-Z][a-zA-Z0-9_]{4,15}$
 9 密码(以字母开头，长度在6~18之间，只能包含字母、数字和下划线)：^[a-zA-Z]\w{5,17}$
10 强密码(必须包含大小写字母和数字的组合，不能使用特殊字符，长度在8-10之间)：^(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{8,10}$  
11 日期格式：^\d{4}-\d{1,2}-\d{1,2}
12 一年的12个月(01～09和1～12)：^(0?[1-9]|1[0-2])$
13 一个月的31天(01～09和1～31)：^((0?[1-9])|((1|2)[0-9])|30|31)$ 
14 钱的输入格式：
15    1.有四种钱的表示形式我们可以接受:"10000.00" 和 "10,000.00", 和没有 "分" 的 "10000" 和 "10,000"：^[1-9][0-9]*$ 
16    2.这表示任意一个不以0开头的数字,但是,这也意味着一个字符"0"不通过,所以我们采用下面的形式：^(0|[1-9][0-9]*)$ 
17    3.一个0或者一个不以0开头的数字.我们还可以允许开头有一个负号：^(0|-?[1-9][0-9]*)$ 
18    4.这表示一个0或者一个可能为负的开头不为0的数字.让用户以0开头好了.把负号的也去掉,因为钱总不能是负的吧.下面我们要加的是说明可能的小数部分：^[0-9]+(.[0-9]+)?$ 
19    5.必须说明的是,小数点后面至少应该有1位数,所以"10."是不通过的,但是 "10" 和 "10.2" 是通过的：^[0-9]+(.[0-9]{2})?$ 
20    6.这样我们规定小数点后面必须有两位,如果你认为太苛刻了,可以这样：^[0-9]+(.[0-9]{1,2})?$ 
21    7.这样就允许用户只写一位小数.下面我们该考虑数字中的逗号了,我们可以这样：^[0-9]{1,3}(,[0-9]{3})*(.[0-9]{1,2})?$ 
22    8.1到3个数字,后面跟着任意个 逗号+3个数字,逗号成为可选,而不是必须：^([0-9]+|[0-9]{1,3}(,[0-9]{3})*)(.[0-9]{1,2})?$ 
23    备注：这就是最终结果了,别忘了"+"可以用"*"替代如果你觉得空字符串也可以接受的话(奇怪,为什么?)最后,别忘了在用函数时去掉去掉那个反斜杠,一般的错误都在这里
24 xml文件：^([a-zA-Z]+-?)+[a-zA-Z0-9]+\\.[x|X][m|M][l|L]$
25 中文字符的正则表达式：[\u4e00-\u9fa5]
26 双字节字符：[^\x00-\xff]    (包括汉字在内，可以用来计算字符串的长度(一个双字节字符长度计2，ASCII字符计1))
27 空白行的正则表达式：\n\s*\r    (可以用来删除空白行)
28 HTML标记的正则表达式：<(\S*?)[^>]*>.*?</\1>|<.*? />    (网上流传的版本太糟糕，上面这个也仅仅能部分，对于复杂的嵌套标记依旧无能为力)
29 首尾空白字符的正则表达式：^\s*|\s*$或(^\s*)|(\s*$)    (可以用来删除行首行尾的空白字符(包括空格、制表符、换页符等等)，非常有用的表达式)
30 腾讯QQ号：[1-9][0-9]{4,}    (腾讯QQ号从10000开始)
31 中国邮政编码：[1-9]\d{5}(?!\d)    (中国邮政编码为6位数字)
32 IP地址：\d+\.\d+\.\d+\.\d+    (提取IP地址时有用)
33 IP地址：((?:(?:25[0-5]|2[0-4]\\d|[01]?\\d?\\d)\\.){3}(?:25[0-5]|2[0-4]\\d|[01]?\\d?\\d)) 
"""
# 获取数组|列表中最大|最小的N个数及索引
"""
# 原理基于堆的，也就是二叉树
a = [1, 2, 3, 4, 5]
rel = heapq.nlargest(3, a)  # 求最大的三个元素
rel2 = map(a.index, rel)  # 求最大的三个元素索引
print(rel)
print(list(rel2))
rem = heapq.nsmallest(3, a)
rem2 = map(a.index, rem)
print(rem)
print(list(rem2))
# 配合lambda表达式
portfolio = [{'name': 'IBM', 'shares': 100, 'price': 91.1},
             {'name': 'AAPL', 'shares': 50, 'price': 543.22},
             {'name': 'FB', 'shares': 200, 'price': 21.09},
             {'name': 'HPQ', 'shares': 35, 'price': 31.75},
             {'name': 'YHOO', 'shares': 45, 'price': 16.35},
             {'name': 'ACME', 'shares': 75, 'price': 115.65}
]
cheap = heapq.nsmallest(3, portfolio, key=lambda s: s['price'])
print(cheap)
expensive = heapq.nlargest(3, portfolio, key=lambda s: s['price'])
print(expensive)
"""
# 每次获取数组|列表最小-并删除弹出
"""
nums = [12, -9, -3, 32, 9, 56, 23, 0, 11, 34]
heapq.heapify(nums)  # 此时集合第一个元素即为最小
print(nums)
while nums:
    # heappop方法: 把第一个元素（最小的）给弹出来，然后第二小的元素会自动补位
    print(heapq.heappop(nums))  # 不可缺少heapq.heapify(nums)这一步
"""
# 数据增广的八种常用方式
"""
# -*- coding:utf-8 -*-
# 旋转、剪切、改变图像色差、扭曲图像特征、改变图像尺寸、增加图像噪声（高斯噪声、盐胶噪声）
'''数据增强
   1. 翻转变换 flip
   2. 随机修剪 random crop
   3. 色彩抖动 color jittering
   4. 平移变换 shift
   5. 尺度变换 scale
   6. 对比度变换 contrast
   7. 噪声扰动 noise
   8. 旋转变换/反射变换 Rotation/reflection
   author: XiJun.Gong
   date:2016-11-29
'''
 
from PIL import Image, ImageEnhance, ImageOps, ImageFile
import numpy as np
import random
import threading, os, time
import logging
 
logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True
 
 
class DataAugmentation:
    '''
    包含数据增强的八种方式
    '''
 
 
    def __init__(self):
        pass
 
    @staticmethod
    def openImage(image):
        return Image.open(image, mode="r")
 
    @staticmethod
    def randomRotation(image, mode=Image.BICUBIC):
        '''
         对图像进行随机任意角度(0~360度)旋转
        :param mode 邻近插值,双线性插值,双三次B样条插值(default)
        :param image PIL的图像image
        :return: 旋转转之后的图像
        '''
        random_angle = np.random.randint(1, 360)
        return image.rotate(random_angle, mode)
 
    @staticmethod
    def randomCrop(image):
        '''
        对图像随意剪切,考虑到图像大小范围(68,68),使用一个一个大于(36*36)的窗口进行截图
        :param image: PIL的图像image
        :return: 剪切之后的图像
        '''
        image_width = image.size[0]
        image_height = image.size[1]
        crop_win_size = np.random.randint(40, 68)
        random_region = (
            (image_width - crop_win_size) >> 1, (image_height - crop_win_size) >> 1, (image_width + crop_win_size) >> 1,
            (image_height + crop_win_size) >> 1)
        return image.crop(random_region)
 
    @staticmethod
    def randomColor(image):
        '''
        对图像进行颜色抖动
        :param image: PIL的图像image
        :return: 有颜色色差的图像image
        '''
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因子
        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因1子
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度
 
    @staticmethod
    def randomGaussian(image, mean=0.2, sigma=0.3):
        '''
         对图像进行高斯噪声处理
        :param image:
        :return:
        '''
 
        def gaussianNoisy(im, mean=0.2, sigma=0.3):
            '''
            对图像做高斯噪音处理
            :param im: 单通道图像
            :param mean: 偏移量
            :param sigma: 标准差
            :return:
            '''
            for _i in range(len(im)):
                im[_i] += random.gauss(mean, sigma)
            return im
 
        # 将图像转化成数组
        img = np.asarray(image)
        img.flags.writeable = True  # 将数组改为读写模式
        width, height = img.shape[:2]
        img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
        img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
        img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
        img[:, :, 0] = img_r.reshape([width, height])
        img[:, :, 1] = img_g.reshape([width, height])
        img[:, :, 2] = img_b.reshape([width, height])
        return Image.fromarray(np.uint8(img))
 
    @staticmethod
    def saveImage(image, path):
        image.save(path)
 
 
def makeDir(path):
    try:
        if not os.path.exists(path):
            if not os.path.isfile(path):
                # os.mkdir(path)
                os.makedirs(path)
            return 0
        else:
            return 1
    except Exception, e:
        print str(e)
        return -2
 
 
def imageOps(func_name, image, des_path, file_name, times=5):
    funcMap = {"randomRotation": DataAugmentation.randomRotation,
               "randomCrop": DataAugmentation.randomCrop,
               "randomColor": DataAugmentation.randomColor,
               "randomGaussian": DataAugmentation.randomGaussian
               }
    if funcMap.get(func_name) is None:
        logger.error("%s is not exist", func_name)
        return -1
 
    for _i in range(0, times, 1):
        new_image = funcMap[func_name](image)
        DataAugmentation.saveImage(new_image, os.path.join(des_path, func_name + str(_i) + file_name))
 
 
opsList = {"randomRotation", "randomCrop", "randomColor", "randomGaussian"}
 
 
def threadOPS(path, new_path):
    '''
    多线程处理事务
    :param src_path: 资源文件
    :param des_path: 目的地文件
    :return:
    '''
    if os.path.isdir(path):
        img_names = os.listdir(path)
    else:
        img_names = [path]
    for img_name in img_names:
        print img_name
        tmp_img_name = os.path.join(path, img_name)
        if os.path.isdir(tmp_img_name):
            if makeDir(os.path.join(new_path, img_name)) != -1:
                threadOPS(tmp_img_name, os.path.join(new_path, img_name))
            else:
                print 'create new dir failure'
                return -1
                # os.removedirs(tmp_img_name)
        elif tmp_img_name.split('.')[1] != "DS_Store":
            # 读取文件并进行操作
            image = DataAugmentation.openImage(tmp_img_name)
            threadImage = [0] * 5
            _index = 0
            for ops_name in opsList:
                threadImage[_index] = threading.Thread(target=imageOps,
                                                       args=(ops_name, image, new_path, img_name,))
                threadImage[_index].start()
                _index += 1
                time.sleep(0.2)
 
 
if __name__ == '__main__':
    threadOPS("/home/pic-image/train/12306train",
              "/home/pic-image/train/12306train3")
"""
# 获取等间隔的数组实例
"""
# np.linspace(start, stop, num, endpoint, retstep, dtype)
# start和stop为起始和终止位置，均为标量
# num为包括start和stop的间隔点总数，默认为50
# endpoint为bool值，为False时将会去掉最后一个点计算间隔
# restep为bool值，为True时会同时返回数据列表和间隔值
# dtype默认为输入变量的类型，给定类型后将会把生成的数组类型转为目标类型
"""
# 对excel操作使用
'''
1、wlrd 读取excel表中的数据
2、xlwt 创建一个全新的excel文件,然后对这个文件进行写入内容以及保存。
3、xlutils 读入一个excel文件，然后进行修改或追加，不能操作xlsx，只能操作xls。
'''
# xlrd - 读取excel数据
"""
# 读excel表
data = xlrd.open_workbook(r"..\..\数据\CSDN\test.xls")
# xlsx打开错误 -- xlrd更新到了2.0.1版本，只支持.xls文件
# 安装旧版 -- pip uninstall xlrd pip install xlrd==1.2.0
# 若只为了打开 -- df=pandas.read_excel("data.xlsx", engine="openpyxl")
table1 = data.sheets()[0]               # 通过索引顺序获取
table2 = data.sheet_by_index(0)         # 通过索引顺序获取
table3 = data.sheet_by_name(u'Sheet1')  # 通过名称获取
print(table1)
print(table2)
print(table3)
# 获取行数列数
nrows = table1.nrows
ncols = table1.ncols
print("行数:%d, 列数:%d" % (nrows, ncols))
# 获取整行和整列的值，以列表形式返回
rows = table1.row_values(0)
cols = table1.col_values(0)
print("rows:%s \n cols:%s" % (rows, cols))
cell_A1 = table1.cell_value(0, 0)
cell_C4 = table1.cell_value(3, 2)
print("A1:%s, C4:%s" % (cell_A1, cell_C4))
"""
# xlwt - 写入excel数据
"""
# encoding:设置字符编码，一般要这样设置：w = Workbook(encoding=’utf-8’)，就可以在excel中输出中文了。默认是ascii
# style_compression:表示是否压缩，不常用。
workbook = xlwt.Workbook(encoding="utf-8", style_compression=0)
# 创建一个sheet对象，一个sheet对象对应Excel文件中的一张表格
# cell_overwrite_ok，表示是否可以覆盖单元格，其实是Worksheet实例化的一个参数，默认值是False
sheet = workbook.add_sheet('test', cell_overwrite_ok=True)
# 向表中添加数据
sheet.write(0, 0, 'EnglishName')  # 其中的'0-行, 0-列'指定表中的单元，'EnglishName'是向该单元写入的内容
sheet.write(1, 0, 'Marcovaldo')
txt1 = '中文名字'
sheet.write(0, 1, txt1)
txt2 = '马可瓦多'
sheet.write(1, 1, txt2)
# 保存
workbook.save(r"..\..\数据\CSDN\test1.xls")
"""
# xlutils - excel追加数据
"""
data = xlrd.open_workbook(r"..\..\数据\CSDN\test1.xls")
ws = xlutils.copy.copy(data)
table = ws.get_sheet(0)
table.write(0, 3, 'D1')
ws.save(r"..\..\数据\CSDN\test1.xls")
"""
# 读写txt文本
"""
'''
读写模式:
r:     读取文件，若文件不存在则会报错
w:     写入文件，若文件不存在则会先创建再写入，会覆盖原文件
a:     写入文件，若文件不存在则会先创建再写入，但不会覆盖原文件，而是追加在文件末尾
rb,wb: 分别于r,w类似，但是用于读写二进制文件
r+:    可读、可写，文件不存在也会报错，写操作时会覆盖
w+:    可读，可写，文件不存在先创建，会覆盖
a+:    可读、可写，文件不存在先创建，不会覆盖，追加在末尾
'''
with open(r"..\..\数据\CSDN\test.txt", "r", encoding="utf-8") as f:
    data = f.read()  # 一次性读取全部内容
    print(data)
with open(r"..\..\数据\CSDN\test.txt", "r", encoding="utf-8") as f:
    data = f.readline()  # 读取一行
    print(data)
with open(r"..\..\数据\CSDN\test.txt", "r", encoding="utf-8") as f:
    data = f.readlines()  # 读取全部内容, 且列表按行形式返回, 但是会读取\n换行符等符号
    print(data)
with open(r"..\..\数据\CSDN\test1.txt", "w", encoding="utf-8") as f:
    f.write("我想测试测试")
"""
# 错误类型
"""
 错误类型——说明
 * ZeroDivisionError——除(或取模)零 (所有数据类型) 
 * ValueError——传入无效的参数
 * AssertionError——断言语句失败 
 * StopIteration——迭代器没有更多的值 
 * IndexError——序列中没有此索引(index) 
 * IndentationError——缩进错误 
 * OSError——输入/输出操作失败 
 * ImportError——导入模块/对象失败 
 * NameError——未声明/初始化对象 (没有属性) 
 * AttributeError——对象没有这个属性
 * GeneratorExit——生成器(generator)发生异常来通知退出 
 * TypeError——对类型无效的操作 
 * KeyboardInterrupt——用户中断执行(通常是输入^C) 
 * OverflowError——数值运算超出最大限制 
 * FloatingPointError——浮点计算错误 
 * BaseException——所有异常的基类 
 * SystemExit——解释器请求退出 
 * Exception——常规错误的基类 
 * StandardError——所有的内建标准异常的基类 
 * ArithmeticError——所有数值计算错误的基类 
 * EOFError——没有内建输入,到达EOF 标记 
 * EnvironmentError——操作系统错误的基类 
 * WindowsError——系统调用失败 
 * LookupError——无效数据查询的基类 
 * KeyError——映射中没有这个键 
 * MemoryError——内存溢出错误(对于Python 解释器不是致命的) 
 * UnboundLocalError——访问未初始化的本地变量 
 * ReferenceError——弱引用(Weak reference)试图访问已经垃圾回收了的对象 
 * RuntimeError——一般的运行时错误 
 * NotImplementedError——尚未实现的方法 
 * SyntaxError Python——语法错误 
 * TabError——Tab 和空格混用 
 * SystemError——一般的解释器系统错误 
 * UnicodeError——Unicode 相关的错误 
 * UnicodeDecodeError——Unicode 解码时的错误 
 * UnicodeEncodeError——Unicode 编码时错误 
 * UnicodeTranslateError——Unicode 转换时错误
 
 以下为警告类型 
 * Warning——警告的基类 
 * DeprecationWarning——关于被弃用的特征的警告 
 * FutureWarning——关于构造将来语义会有改变的警告 
 * OverflowWarning——旧的关于自动提升为长整型(long)的警告 
 * PendingDeprecationWarning——关于特性将会被废弃的警告 
 * RuntimeWarning——可疑的运行时行为(runtime behavior)的警告 
 * SyntaxWarning——可疑的语法的警告 
 * UserWarning——用户代码生成的警告
"""
# 读写csv文件
"""
# 整体写入
df1 = pd.DataFrame(np.arange(12).reshape((3, 4)), columns=['a', 'b', 'c', 'd'])
df1.to_csv(r"..\..\数据\CSDN\test.csv", encoding='utf-8')
# 以不断添加行的形式写入
# python写csv文件会遇到两个问题，一个是csv文件每写入一行会自动空一行，一个是中文遇到的编码问题。
li1 = [1, 2, 3, 4, 5]
with open(r"..\..\数据\CSDN\test1.csv", 'a+', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(li1)
    writer.writerow(li1)
# 或者
# f = open(r"..\..\数据\CSDN\test1.csv", 'a+', newline='')
# 中文编码问题
'''
# python3.6（以及相关版本，例3.5等）写入以编码为'utf-8'中文时，
# 虽然读的时候用'utf-8'打开不影响中文编码，但用excel打开csv文件时，
# 会出现中文乱码问题，因此采用编码为'utf-8-sig'的方式写入，
# 读文件时可用'utf-8'打开，也可用'utf-8-sig'打开。
'''
# csv读取
df = pd.read_csv(r"..\..\数据\CSDN\test1.csv", encoding="utf-8")
print(df)
"""
# python设置字典-值为列表|字典
"""
# 内置方法
dic = {}
dic.setdefault('a', []).append(1)
dic.setdefault('a', []).append(2)
print(dic)
dic1 = {}
dic1.setdefault('b', {})['f'] = 1
dic1.setdefault('b', {})['h'] = 1
dic1.setdefault('b', {})["i"] = 1
print(dic1)
# collections包的defaultdict方法
dic2 = defaultdict(list)
dic2["b"].append(4)
dic2["c"].append(3)
print(dict(dic2))
dic3 = defaultdict(dict)
dic3["b"]["a"] = 1
dic3["b"]["b"] = 2
dic3["c"]["a"] = 3
print(dict(dic3))
"""
# NetworkX-画图
"""
G = nx.Graph()
nx.draw(G, pos=nx.random_layout(G), node_color="b",
        edge_color='r', with_labels=True,
        font_size=18, node_size=20)
# pos 指的是布局 主要有spring_layout , random_layout,circle_layout,shell_layout。
# node_color指节点颜色，有rbykw ,同理edge_color.
# with_labels指节点是否显示名字,size表示大小，font_color表示字的颜色。
nx.draw_networkx_edges(G, pos=nx.random_layout(G), edgelist=None, width=1.0,
                       edge_color='k', style='solid', alpha=1.0, edge_cmap=None,
                       edge_vmin=None, edge_vmax=None, ax=None, arrows=True, label=None)
# G：图表 一个networkx图
# pos：dictionary 将节点作为键和位置作为值的字典。位置应该是长度为2的序列。
# edgelist：边缘元组的集合 只绘制指定的边（默认 = G.edges（））
# width：float或float数组 边线宽度（默认值 = 1.0）
# edge_color：颜色字符串或浮点数组边缘颜色。可以是单颜色格式字符串（default = 'r'）,或者具有与edgelist相同长度的颜色序列。
# 如果指定了数值, 它们将被映射到, 颜色使用edge_cmap和edge_vmin，edge_vmax参数。
# style：string 边线样式（默认 = 'solid'）（实线 | 虚线 | 点线，dashdot）
# alpha：float 边缘透明度（默认值 = 1.0）
# edge_cmap：Matplotlib色彩映射 用于映射边缘强度的色彩映射（默认值 = 无）
# edge_vmin，edge_vmax：float 边缘色图缩放的最小值和最大值（默认值 = 无）
# ax：Matplotlib Axes对象，可选 在指定的Matplotlib轴中绘制图形。
# arrows：bool，optional（default = True） 对于有向图，如果为真，则绘制箭头。
# label：图例的标签
nx.draw_networkx_nodes(G, pos=nx.random_layout(G), nodelist=None, node_size=300,
                       node_color='r', node_shape='o', alpha=1.0, cmap=None,
                       vmin=None, vmax=None, ax=None, linewidths=None, label=None)
# pos：dictionary 将节点作为键和位置作为值的字典。 位置应该是长度为2的序列。
# ax：Matplotlib Axes对象，可选 在指定的Matplotlib轴中绘制图形。
# nodelist：list，可选 只绘制指定的节点（默认G.nodes（））
# node_size：标量或数组 节点大小（默认值= 300）。如果指定了数组，它必须是与点头长度相同。
# node_color：颜色字符串或浮点数组节点颜色。可以是单颜色格式字符串（default ='r'）, 或者具有与点头相同长度的颜色序列。
# 如果指定了数值，它们将被映射到颜色使用cmap和vmin，vmax参数。看到matplotlib.scatter更多详细信息。
# node_shape：string 节点的形状。规格为matplotlib.scatter 标记，'so ^> v <dph8'（默认='o'）之一。
# alpha：float 节点透明度（默认值= 1.0）
# cmap：Matplotlib色图 色彩映射节点的强度（默认=无）
# vmin，vmax：float 节点色彩映射缩放的最小值和最大值（默认值=无）
# 线宽：[无|标量|序列] 符号边框的线宽（默认值= 1.0）
# label：[无|串] 图例的标签
"""
# 随机生成均匀分布在单位圆内的点
"""
'''
原理: 随机函数可以产生[0,1)区间内的随机数，但是如果我们想生成随机分布在单位圆上的，
那么我们可以首先生成随机分布在单位圆边上的点，然后随机调整每个点距离原点的距离，
但是我们发现这个距离不是均匀分布于[0,1]的，而是与扇形的面积相关的

我们使用另外的随机函数生成从[0,1)的随机数r，我们发现r<s0的概率为s0，
显而易见，如果r为0，那么对应的距离应该为0，如果是1，对应的距离自然也应该是1，
假设我们产生了m个随机数，那么小于s0的随机数应该为s0*m左右，
而且这些应该对应于扇形面积的s0倍处即图2的小扇形区域，落在这一区域的点应该为s0*m，
此时扇形边长为s0^0.5,因此s0对应的距离应该为s0^0.5，因此我们得到的映射函数为y=x^0.5

因此我们对于每个顶点的边长便是产生随机数的算术平方根的大小
'''
samples_num = 800
t = np.random.random(size=samples_num) * 2 * np.pi - np.pi
x = np.cos(t)
y = np.sin(t)
i_set = np.arange(0, samples_num, 1)
for i in i_set:
    len_set = np.sqrt(np.random.random())
    x[i] = x[i] * len_set
    y[i] = y[i] * len_set
plt.figure(figsize=(10, 10.1), dpi=125)
plt.plot(x, y, 'ro')
_t = np.arange(0, 7, 0.1)
_x = np.cos(_t)
_y = np.sin(_t)
plt.plot(_x, _y, 'g-')
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Random Scatter')
plt.grid(True)
plt.savefig(r"..\..\数据\CSDN\imag.png")
plt.show()
"""
