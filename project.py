"""
 * 图像扩容
 * 贺卡
 * 贺卡1
 * 贺卡2
 * 数据挖掘大作业：产品推荐
 * 爬虫模板
 * 爬虫解释版
 * 爬虫解释版1
 * 深圳杯公交车数据处理
 * 金融建模第一题
 * 股票模板简略版
 * 股票策略模版
 * 收益率曲线
 * 金钗银钗
 * 长短线
 * CNN图像-分类
 * RNN图像-分类-正确版
 * RNN图像-分类-原始版
"""

# 图像扩容
"""
'''
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
    except Exception as e:
        print
        str(e)
        return -2


def imageOps(func_name, image, des_path, file_name, times=5):
    funcMap = {"randomRotation": DataAugmentation.randomRotation,
               # "randomCrop": DataAugmentation.randomCrop,
               "randomColor": DataAugmentation.randomColor,
               "randomGaussian": DataAugmentation.randomGaussian
               }
    if funcMap.get(func_name) is None:
        logger.error("%s is not exist", func_name)
        return -1

    for _i in range(0, times, 1):
        new_image = funcMap[func_name](image)
        DataAugmentation.saveImage(new_image, os.path.join(des_path, func_name + str(_i) + file_name))


opsList = {"randomRotation", "randomColor", "randomGaussian"}


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
        print
        img_name
        tmp_img_name = os.path.join(path, img_name)
        if os.path.isdir(tmp_img_name):
            if makeDir(os.path.join(new_path, img_name)) != -1:
                threadOPS(tmp_img_name, os.path.join(new_path, img_name))
            else:
                print
                'create new dir failure'
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
    threadOPS("D:\\作业\\比赛\\美赛\\图像分类结果\\Positive ID",
              "D:\\作业\\比赛\\美赛\\图像分类结果\\Positive ID")
"""
# 贺卡
"""
import turtle as t


def bgpic(self, picname=None):
    if picname is None:
        return self._bgpicname
    if picname not in self._bgpics:
        self._bgpics[picname] = self._image(picname)
    self._setbgpic(self._bgpic, self._bgpics[picname])
    self._bgpicname = picname
    if __name__ == '__main__':
        myWin = t.Screen()
        t.setup(width=600, height=750, startx=0, starty=0)
        t.bgpic(r'./1.gif')

    myWin.exitonclick()


t.bgpic(r'pig.gif')

t.delay(0)


t.setup(1000, 600, 0, 0)

t.up()

t.goto(-50, 0)

t.color("black")

t.write("钟孟诗", font=16)

t.up()

t.hideturtle()

t.done()
"""
# 贺卡1
"""
import turtle as ti
from random import *
from math import *


def tree(n, l):
    ti.pd()
    t = cos(radians(ti.heading() + 45)) / 8 + 0.25
    ti.pencolor(t, t, t)
    ti.pensize(n / 3)
    ti.forward(l)

    if n > 0:
        b = random() * 15 + 10
        c = random() * 15 + 10
        d = l * (random() * 0.25 + 0.7)
        ti.right(b)
        tree(n - 1, d)
        ti.left(b + c)
        tree(n - 1, d)
        ti.right(c)
    else:
        ti.right(90)
        n = cos(radians(ti.heading() - 45)) / 4 + 0.5
        ti.pencolor(n, n * 0.8, n * 0.8)
        ti.circle(3)
        ti.left(90)
        if (random() > 0.7):
            ti.pu()
            t = ti.heading()
            an = -40 + random() * 40
            ti.setheading(an)
            dis = int(800 * random() * 0.5 + 400 * random() * 0.3 + 200 * random() * 0.2)
            ti.forward(dis)
            ti.setheading(t)
            ti.pd()
            ti.right(90)
            n = cos(radians(ti.heading() - 45)) / 4 + 0.5
            ti.pencolor(n * 0.5 + 0.5, 0.4 + n * 0.4, 0.4 + n * 0.4)
            ti.circle(2)
            ti.left(90)
            ti.pu()
            t = ti.heading()
            ti.setheading(an)
            ti.backward(dis)
            ti.setheading(t)
    ti.pu()
    ti.backward(l)


ti.bgcolor(0.5, 0.5, 0.5)
ti.ht()
ti.speed(0)
ti.tracer(0, 0)
ti.pu()
ti.backward(100)
ti.left(90)
ti.pu()
ti.backward(300)
tree(12, 100)
ti.up()

ti.goto(350, -220)

ti.color("black")

ti.write("黄妍同学，你终将如诗一样美丽", font=16)

ti.up()

ti.goto(430, -260)

ti.color('black')

ti.write("长高高哦", font=16)

ti.up()

ti.goto(650, -370)

ti.color('black')

ti.write("2020年03月26号", move=True)

ti.delay(0)

ti.up()

ti.goto(630, -360)

ti.color('black')

ti.write("《夜深人静》——所制", move=True)

ti.delay(0)

ti.up()

ti.goto(-630, 340)

ti.color('black')

ti.write("声声慢·寻寻觅觅", font=8)

ti.up()

ti.goto(-610, 320)

ti.color('black')

ti.write("[宋] 李清照", font=8)

ti.up()

ti.goto(-680, 300)

ti.color('black')

ti.write("寻寻觅觅，冷冷清清，凄凄惨惨戚戚。", font=8)

ti.up()

ti.goto(-645, 280)

ti.color('black')

ti.write("乍暖还寒时候，最难将息。", font=8)

ti.up()

ti.goto(-675, 260)

ti.color('black')

ti.write("三杯两盏淡酒，怎敌他、晚来风急？", font=8)

ti.up()

ti.goto(-665, 240)

ti.color('black')

ti.write("雁过也，正伤心，却是旧时相识。", font=8, move=True)

ti.up()

ti.goto(-690, 200)

ti.color('black')

ti.write("满地黄花堆积。憔悴损，如今有谁堪摘？", font=8, move=True)

ti.up()

ti.goto(-640, 180)

ti.color('black')

ti.write("守着窗儿，独自怎生得黑？", font=8, move=True)

ti.up()

ti.goto(-670, 160)

ti.color('black')

ti.write("梧桐更兼细雨，到黄昏、点点滴滴。", font=8, move=True)

ti.up()

ti.goto(-640, 140)

ti.color('black')

ti.write("这次第，怎一个愁字了得！", font=8, move=True)

ti.up()

ti.hideturtle()
ti.done()
"""
# 贺卡2
"""
import turtle as t


def bgpic(self, picname=None):
    if picname is None:
        return self._bgpicname
    if picname not in self._bgpics:
        self._bgpics[picname] = self._image(picname)
    self._setbgpic(self._bgpic, self._bgpics[picname])
    self._bgpicname = picname
    if __name__ == '__main__':
        myWin = t.Screen()
        t.setup(width=600, height=750, startx=0, starty=0)
        t.bgpic(r'./1.gif')

    myWin.exitonclick()


t.bgpic(r'pig.gif')

t.delay(0)


t.setup(1000, 600, 0, 0)

t.colormode(255)

t.pencolor("green")

t.pensize(5)

for i in range(8):

    t.circle(50)

    t.right(-45)

t.pencolor("red")

t.pensize(5)

for i in range(8):

    t.circle(25)

    t.right(-45)

t.right(120)

t.circle(-400, 50)


t.penup()
t.goto(-300, -220)
t.pendown()
t.width(25)
t.pencolor("purple")
t.seth(-40)
for i in range(4):
    t.circle(40, 80)
    t.circle(-40, 80)
t.circle(40, 80/2)
t.fd(40)
t.circle(16, 180)
t.fd(40*2/3)
t.penup()
t.bk(10)
t.pencolor("black")
t.pensize(5)
t.pendown()
t.circle(1, 360)

t.up()

t.goto(150, -120)

t.color("black")

t.write("钟孟诗同学，你终将如诗一样美丽", font=16)

t.up()

t.goto(250, -160)

t.color('black')

t.write("女生节快乐", font=16)

t.up()

t.goto(350, -270)

t.color('black')

t.write("2020年03月07号", move=True)

t.delay(0)

t.up()

t.goto(-485, 240)

t.color('black')

t.write("《清平调·其一》", font=8)

t.up()

t.goto(-430, 220)

t.color('black')

t.write("李白", font=8)

t.up()

t.goto(-480, 200)

t.color('black')

t.write("云想衣裳花想容，", font=8)

t.up()

t.goto(-480, 180)

t.color('black')

t.write("春风拂槛露华浓。", font=8)

t.up()

t.goto(-480, 160)

t.color('black')

t.write("若非群玉山头见，", font=8)

t.up()

t.goto(-480, 140)

t.color('black')

t.write("会向瑶台月下逢。", font=8, move=True)

t.up()

t.hideturtle()

t.done()
"""
# 数据挖掘大作业：产品推荐
"""
import csv
import random
import bisect
class WeightedRandomGenerator(object):
    def __init__(self, weights):
        self.totals = []
        running_total = 0

        for w in weights:
            running_total += w
            self.totals.append(running_total)

    def next(self):
        rnd = random.random() * self.totals[-1]
        return bisect.bisect_right(self.totals, rnd)

    def __call__(self):
        return self.next()

def eva(num0, num1, datas):    #这里其实是为了将他强行写为函数的，方便之后.csv中的内容改了可以继续操作。  num0表示产品型号那一列，num1表示产品故障那一列
    eva = []      #这个列表里面存放的是统计数据
    exist = []    #这个列表里面存放的是产品型号的顺序,但里面都是产品名
    total = []    #为了能够一个函数同时输出刚刚的两个列表，所以我就用一个列表把他们都包起来了。
    for data in datas:
        ProductName = str(data[num0])
        ErrorName = str(data[num1])
        if ProductName not in exist:
            d = {}
            count = 1
            d[ErrorName] = count
            exist.append(ProductName)
            eva.append(d)
        else:
            product = exist.index(data[num0])
            if ErrorName not in eva[product]:
                count_another = 1
                eva[product][ErrorName] = count_another
            else:
                eva[product][ErrorName] += 1   #要注意把序列号的方括号卸载外面

    total.append(eva)
    total.append(exist)
    return total

def score(eva):
    scores = []
    for example in eva:
        sum = 0
        d_sc = {}
        for test in example:
            sum += example[test]
        for test in example:
            score = 100 * (example[test] / sum)
            score = round(score,3)
            d_sc[test] = score
        scores.append(d_sc)

    return scores

def sim(scores,exist,w):   #scores是各个评分组成的字典所组成的列表，exist是各个项目的列表。两者从序号上一一对应。w表示应当采用那种计算方式。
    final = {}
    for test in exist:
        similarity = {}
        train_scores = list(scores)
        train_exist = list(exist)
        i = exist.index(test)
        test_score = train_scores[i]    #要注意这个时候test_score已经是一个字典了，而train_score中依然是很多字典组成的列表。
        train_scores.pop(i)
        train_exist.pop(i)
        if w==1:     #1表示采用余弦相似度。
            for train in train_scores:      #这里的每个train还是一个训练集中的一个字典
                k = train_scores.index(train)
                name = train_exist[k]
                sum_pixqi = 0    #这个表示余弦相似度的分子
                sum_pi = 0      #这个表示余弦相似度中分子的测试集的那一项
                sum_qi = 0      #这个表示余弦相似度中分子的训练集的那一项
                for test_need in test_score:
                    sum_pi += test_score[test_need]
                    if test_need in train:
                        sum_qi += train[test_need]
                        sum_pixqi += (train[test_need] * test_score[test_need])
                    else:
                        sum_qi += 0.50
                        sum_pixqi += (test_score[test_need] * 0)
                cos = (sum_pixqi / (sum_pi * sum_qi)) ** (1/2)
                cos_3 = round(cos,3)
                similarity[name] = cos_3
        final[test] = similarity
    return final

def cop(scores,exist):     #这个方法是用来合并名称和分数的，最后会得到一个字典
    copd = {}
    for a in exist:
        number = exist.index(a)
        content = scores[number]
        copd[a] = content
    return copd

def prodict(cop, sim, exist):
    prodict = {}
    for Product in exist:
        recommend = []
        sim_Product = sim[Product]  #把产品的相似度字典抽出来
        score_Product = cop[Product]    #把产品的分数抽出来
        sort_sim_groupDict = sorted(sim_Product.items(), key= lambda item:item[1], reverse= True)
        sort_sim = {i[0]:i[1] for i in sort_sim_groupDict}
        recommend_dict = {}
        for key,value_sim in sort_sim.items():
            score_sim_Product = cop[key]
            sort_score_sim_groupDict = sorted(score_sim_Product.items(), key=lambda item: item[1], reverse=True)
            sort_score_sim_Product = {i[0]: i[1] for i in sort_score_sim_groupDict}
            count = 0
            for ErrorName in sort_score_sim_Product:
                if ErrorName not in score_Product:
                    score_need = sort_score_sim_Product[ErrorName] * sort_sim[key]
                    recommend_dict[ErrorName] = score_need
            count += 1
            if count >= 5:
                break
        sort_recommend_groupDict = sorted(recommend_dict.items(), key=lambda item: item[1], reverse=True)
        sort_recommend_dict = {i[0]: i[1] for i in sort_recommend_groupDict}
        for i in sort_recommend_dict.keys():
            recommend.append(i)
        recommend_wash = list(set(recommend))      #清洗数据
        recommend_wash.sort(key=recommend.index)      #清洗数据之后排序
        recommend_indeed = []
        for i in range(len(recommend_wash)):
            recommend_indeed.append(recommend_wash[i])  # 将算法得到的前三个故障推荐出去
        prodict[Product] = recommend_indeed
    return prodict

file = open("op_1.csv")
datas = csv.reader(file)
eva_all = eva(0,1,datas)   #先填0，再填1，是用型号评分故障；如果先填1，再填0，就是用故障评分型号
need = eva_all[0]
exist = eva_all[1]
scores = score(need)
copx = cop(scores, exist)
simx = sim(scores,exist,1)
prodictx = prodict(copx, simx, exist)

# print("通过协同过滤推荐算法，以下设备可能出现的故障是：")
# for i in prodictx:
#     print(str(i) + ": " + prodictx[i][0] + " 或 " + prodictx[i][1] + " 或 " + prodictx[i][2])

# 对于得到数据排序
need_change=[]
need_change_end=[]
exist_change=[]
for i in range(len(need)):
    if len(need[i])>10:
        need_change.append(need[i])
        exist_change.append(exist[i])
for i in range(len(need_change)):
    need_change_end.append(sorted(need_change[i].items(),key=lambda d:d[1],reverse=True))
# 测试集的数据点
need_change_qz=[]
need_change_sum=[]
for i in range(len(need_change_end)):
    sum = 0
    need_change_test = []
    for j in range(len(need_change_end[i])):
        sum+=need_change_end[i][j][1]
        need_change_test.append(need_change_end[i][j][1])
    need_change_qz.append(need_change_test)
    need_change_sum.append(sum)
need_change_qz_end=[]
for i in range(len(need_change_qz)):
    if len(need_change_qz[i])>10:
        num=0
        d_test = {}
        test = WeightedRandomGenerator(need_change_qz[i])
        for j in range(need_change_sum[i]//5):  #取1/5为测试集
            test_work=test.__call__()
            for key in d_test:
                if test_work==key:
                    num=1
            if num==1:
                d_test[test_work]+=1
            else:
                d_test[test_work]=1
            num=0
        need_change_qz_end.append(d_test)
need_test_end=[]
for i in range(len(need_change_qz_end)):
    need_test_end.append(sorted(zip(need_change_qz_end[i].keys(), need_change_qz_end[i].values())))
need_end=[]
need_begin=[]
for i in range(len(need_test_end)):
    need_dt={}
    need_dtb={}
    num1 = 0
    leng = len(need_test_end[i])-1
    for j in range(len(need_change_end[i])):
        k = need_test_end[i][num1][0]
        if j==k:
            need_dt[need_change_end[i][j][0]]=need_change_end[i][j][1]-need_test_end[i][num1][1]
            need_dtb[need_change_end[i][j][0]]=need_test_end[i][num1][1]
            if num1<leng:
                num1+=1
        else:
            need_dt[need_change_end[i][j][0]]=need_change_end[i][j][1]
    need_end.append(need_dt)
    need_begin.append(need_dtb)

# 去掉孤点测试集-need_begin

# 去点孤点验证集-need_end

# 减去孤点产品符号-exist_change

# 减去孤点原本集-need_change

pro_change = []
for i in range(len(need_begin)):
    pro_num = 0
    pro_num_test = 0
    need_retain = need_change[i]
    need_change[i] = need_begin[i]
    scores_change = score(need_change)
    cop_change = cop(scores, exist)
    sim_change = sim(scores_change, exist_change, 1)
    prodict_change = prodict(cop_change, sim_change, exist_change)
    # for key in prodict_change:
    #     if pro_num_test == i:
    #         print(prodict_change[key])
    #     pro_num_test += 1

    for pro in prodict_change:
        if pro_num_test == i:
            print(prodict_change[pro])
            print(need_end[i])
            for key in need_end[i]:
                for cont in prodict_change[pro]:
                    if key == cont:
                        pro_num += 1
        pro_num_test += 1
    print(pro_num)
    # if pro_num >= 5:
    #     pro_change.append(prodict_change[i])
    # need_change[i]=need_retain
    # for i in pro_change:
    #     print(str(i) + ": " + prodict_change[i][0] + " 或 " + prodict_change[i][1] + " 或 " + prodict_change[i][2])

# for i in prodictx:
#     print(str(i) + ": " + prodictx[i][0] + " 或 " + prodictx[i][1] + " 或 " + prodictx[i][2])
"""
# 爬虫模板
"""
# -*- codeing = utf-8 -*-

# 导入模块
import requests

'''网页解析，获取数据'''
from bs4 import BeautifulSoup

'''正则表达式，进行文字匹配'''
import re

'''制定URL，获取网页数据'''
import urllib.request
import urllib.error
'''解析器'''
import urllib.parse

'''进行Excel操作'''
import xlwt

'''进行数据库操作'''
import sqlite3

'''流程：'''
'''
1、爬取网页
2、解析数据
3、保存数据
'''

def url_open(url):  # 打开每个地址
    # 添加header，伪装成浏览器
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36"}
    res = requests.get(url, headers=headers)
    return res


# 创建三个储存信息的列表
name_list = []
info_list = []
rate_list = []


# 获取电影名字
def get_name(soup, name_list):
    targets = soup.find_all("div", class_="hd")
    for each in targets:
        name_list.append(each.a.span.text)


# 获取电影信息
def get_info(soup, info_list):
    targets = soup.find_all("div", class_='bd')
    for each in targets:
        try:
            info_list.append(each.p.text.split('\n')[1].strip() + each.p.text.split('\n')[2].strip())
        except:
            continue


# 获取电影评分
def get_rate(soup, rate_list):
    targets = soup.find_all("span", class_="rating_num")
    for each in targets:
        rate_list.append(each.text)


# 将获取信息写入TXT文件
def write_into(name_list, info_list, rate_list):
    with open("豆瓣Top250电影.txt", "w", encoding="utf-8") as f:
        for i in range(250):
            f.write(name_list[i] + '    评分:' + rate_list[i] + '   ' + info_list[i] + '\n\n')


url = []
for i in range(10):  # 得到十个页面地址
    url.append("https://movie.douban.com/top250?start=%d&filter=" % (i * 25))


def main():
    # 遍历每个页面链接并获取信息
    for each_url in url:
        res = url_open(each_url)
        soup = BeautifulSoup(res.text, "html.parser")
        get_name(soup, name_list)
        get_info(soup, info_list)
        get_rate(soup, rate_list)

    write_into(name_list, info_list, rate_list)


# 该模块既可以导入到别的模块中使用，另外该模块也可自我执行
'''定义程序入口、执行第一个函数、主程序开始点'''
if __name__ == "__main__":
    '''当我们程序执行时，调用函数'''
    '''目的：为了控制函数调用流程'''
    '''此刻调用的就是主程序函数'''
    main()

for i in range(100):
    print(name_list[i] + " " + info_list[i] + " " + rate_list[i])

"""
# 爬虫解释版
"""
#!/爬虫/bin/env python

# -*- coding: utf-8 -*-

'''网页解析，获取数据'''
from bs4 import BeautifulSoup

'''正则表达式，进行文字匹配'''
import re

'''制定URL，获取网页数据'''
import urllib.request
import urllib.error
'''解析器'''
import urllib.parse

'''进行Excel操作'''
import xlwt

'''进行数据库操作'''
import sqlite3

'''流程：'''
'''
1、爬取网页
2、解析数据
3、保存数据
'''

# 影片详细链接
findLink = re.compile(r'<a href="(.*?)">')
# 影片图片
findImgSrc = re.compile(r'<img .* src="(.*?)"', re.S)  # re.S：让换行符包含在字符中
# 影片片名
findTitle = re.compile(r'<span class="title">(.*)</span>')
# 影片其他名
findTitle_other = re.compile(r'<span class="other">(.*)</span>')
# 影片评分
findRating = re.compile(r'<span class="rating_num" property="v:average">(.*)</span>')
# 评价人数
findJudge = re.compile(r'<span>(\d*)人评价</span>')
# 概况
findInq = re.compile(r'<span class="inq">(.*)</span>')
# 影片相关内容
findBd = re.compile(r'<p class="">(.*?)</p>', re.S)


def main():
    baseurl = "https://movie.douban.com/top250?start="

    data_list = getData(baseurl)
    save_path = "豆瓣电影Top250.xls"

    saveData(data_list, save_path)


# 爬取页面
def getData(baseurl):
    dataList = []
    # 调用获取页面信息的函数——10次
    for i in range(0, 10):
        url = baseurl + str(i*25)
        # 保存获取到的网页源码
        html = askURL(url)
        # 逐一解析数据
        soup = BeautifulSoup(html, "html.parser")
        # 查找符合要求字符串，整合到一个列表
        for item in soup.find_all('div', class_='item'):
            # item的全部信息
            # print(item)
            data = []
            item = str(item)
            # 内容解析操作，要改for i里面的10为1
            # print(item)
            # break

            # re正则查找符合指定字符串的所有字符串，整合成一个列表
            link = re.findall(findLink, item)[0]
            data.append(link)
            img_src = re.findall(findImgSrc, item)[0]
            data.append(img_src)
            title = re.findall(findTitle, item)
            # 对title处理
            if len(title) > 1:
                # 添加中文名
                c_title = title[0]
                data.append(c_title)
                # 添加外国名
                # replace：去除无关符号
                o_title = title[1].replace("/", "")
                data.append(o_title.strip())
            else:
                data.append(title[0])
                # 留空防止篡位
                data.append(" ")
            other_title = re.findall(findTitle_other, item)[0].replace("/", "")
            data.append(other_title.strip())
            rating = re.findall(findRating, item)[0]
            data.append(rating)
            judge = re.findall(findJudge, item)[0]
            data.append(judge)
            inq = re.findall(findInq, item)
            if len(inq) != 0:
                inq = inq[0].replace("。", "")
                data.append(inq)
            else:
                data.append(" ")
            bd = re.findall(findBd, item)[0]
            # 去掉<br/>
            bd = re.sub('<br(\s+)?/>(\s+)?', ":", bd)
            # 去掉/
            bd = re.sub('/', " ", bd)
            # 去空格
            bd = bd.strip()
            # 拆分
            bd = re.split("[':''\xa0''.{3}']", bd)
            bd = [c.strip() for c in bd if c.strip() and c not in ('导演', '主演')]
            # .strip去掉前后空格
            if len(bd) < 5:
                bd.insert(1, '...')
            data.extend(bd)
            # 将处理好的一部电影内容放入dataList
            dataList.append(data)
    print()
    return dataList

    '''练习代码（数据解析处理）'''
    '''
    BeautifulSoup4能将复杂的HTML文档转换成一个复杂的树形结构，每个节点都是python对象，所有对象可以归纳为4种：
    —— Tag：标签及其内容（第一个）
    —— NavigableString：标签里面的内容（字符串）
    —— BeautifulSoup：整个文档内容
    —— Comment：属于特殊的NavigableString，输出不包含注释符号

    字符串过滤（find_all()）：会查找与字符串完全匹配的所有标签内容
    正则表达式搜索（search()）：会查找包含字符串的所有标签内容
    方法：传入一个函数（方法），根据函数的要求来搜索
    属性参数：会查找符合指定参数（属性）的所有内容
    文本（text）参数：会查找符合指定参数（文本）的所有内容
    limit参数（n）：会查找所有符合条件的内容的前n个

    css选择器（select()）：

    file = open("./baidu.html","rb")
    html = file.read()
    # html.parser：解析器选择-html
    bs = BeautifulSoup(html,"html.parser")
    # 获得第一个指定标签的所有内容
    print(bs.title)
    print(bs.a)
    print(bs.head)
    # 获得第一个指定标签内容
    print(bs.title.string)
    # 获得标签属性值
    print(bs.a.attrs)
    # 获得整个文档内容
    print(bs)
    # 获得指定标签内的所有内容（形成列表）
    print(ba.head.content)
    # 文档搜索
    # find_all：将所有指定标签内容整合到一个列表内（如果标签没有其他地方单独使用）
    t_list = bs.find._all("a")
    print(t_list)
    # search()： 将所有带有指定字符串的标签内容整合到一个列表（凡是标签内容带有指定字符串）
    t_list = bs.find._all(re.compile("a"))
    t_list = bs.search(re.compile("a"))
    # has_attr()：返回标签内指定属性的值
    def name_is_exists(tag):
        return tag.has_attr("name")
    # 将所有含有name属性的标签内容整合到一个列表
    t_list = bs.find_all(name_is_exists)
    # 参数搜索（kwargs）:将所有带有指定参数的全部内容整合到一个列表
    # 属性参数
    # 获得id="head"的全部内容
    t_list = bs.find_all(id="head")
    print(t_list)
    # 获得含有class的所有内容
    t_list = bs.find_all(class_=True)
    print(t_list)
    # 文本参数
    # 获得所有含有指定内容的内容，如“hao123”（可用来测试内容出现次数）
    t_list = bs.find_all(text = "hao123")
    print(t_list)
    # 获得所有含有数字的内容，使用正则表达式
    t_list = bs.find_all(text = re.compile("\d"))
    print(t_list)
    # limit参数
    # 将所有“a”标签的前3个全部内容整合到一个列表
    t_list = bs.find_all("a",limit=3)
     print(t_list)
     # css查找
     # 标签查找
     t_list = bs.select("title")
     # 类名查找（所有含有指定类名内容整合到一个列表）
     t_list = bs.select(".mnav")
     # id名查找（所有含有指定id名内容整合到一个列表）
     t_list = bs.select("#u1")
     # 混合查找（所有包含混合内容整合到一个列表）
     t_list = bs.select("a[class='bri']")
     # 子标签查找（某标签内容内的某标签内容整合到一个列表）
     t_list = bs.select("head > title")
     # 通过兄弟标签查找（两标签（或属性）内的所有内容整合到一个列表）
     t_list = bs.select(".mnav ~ .bri")
     # .get_text()：获得指定标签的相邻指定标签的全部内容
     print(t_list[0].get_text())
    '''


# 得到指定一个URL的网页内容
def askURL(url):
    head = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36"
    }

    request = urllib.request.Request(url, headers=head)
    html = ""
    try:
        response = urllib.request.urlopen(request)
        html = response.read().decode("utf-8")
        # print(html)
    except urllib.error.URLError as e:
        if hasattr(e, "code"):
            print(e.code)
        if hasattr(e, "reason"):
            print(e.reason)
    return html


    '''练习代码（爬取网站）'''
    '''
    # .urlopen为发出请求，收获网页源代码

    # 获得一个get请求
    # timeout为等待时间：时间内没响应报错（此处等待时间为1s）
    # try、except为超时报错处理
    try:
        response = urllib.request.urlopen("http://httpbin.org/get",timeout=1)
        # .read()为读取网页源代码，.decode('uft-8')为设置解码方式
        print(response.read().decode('utf-8'))
    except urllib.error.URLError as e:
        print("time out!")

    # 获得一个post请求
    # 一般为模拟登录操作
    # 其需要表单封装,bytes()为二进制封装
    # .urlencode为内容输入（可以为登录信息），encoding=为封装方式
    data = bytes(urllib.parse.urlencode({"hello":"world"}),encoding="utf-8")
    response = urllib.request.urlopen("http://httpbin.org/post",data = data)
    print(response.read().decode('utf-8'))

    # 获得一个请求（模拟浏览器操作）
    url = "http://www.douban.com"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36"
    }
    data = bytes(urllib.parse.urlencode({"name":"eric"}),encoding="utf-8")
    # headers:游览器头部信息，method：浏览器请求方式，data：游览器输入数据，url：网址
    # 此刻req仅为请求对象
    req = urllib.request.Request(url=url,data=data,headers=headers,method="POST")
    # 同样用urlopen发送请求
    response = urllib.request.urlopen(req,timeout=1)
    print(response.read().decode('utf-8'))
    '''


# 保存数据
def saveData(datalist, savepath):
    print("save....")
    # style_compression：0为不可以压缩，1为可以压缩
    book = xlwt.Workbook(encoding="utf-8", style_compression=0)
    # cell_overwrite_ok：True为可以单元格内容可以覆盖，False为内容不可以覆盖
    sheet = book.add_sheet('豆瓣电影Top250', cell_overwrite_ok=True)
    # 列名设定
    col = ("电影详情链接", "影片图片链接", "影片中文名", "影片外国名",
           "影片其他名", "影片评分", "影片评价人数", "影片概括",
           "影片导演", "影片主演", "影片上映时间", "影片国家", "影片类型")
    for i in range(0, len(col)):
        sheet.write(0, i, col[i])
    for i in range(0, 250):
        # 输出写入进度
        print("第%d条" % (i+1))
        for j in range(0, len(col)):
            sheet.write(i+1, j, datalist[i][j])
    book.save(savepath)

    '''练习代码（excel保存）'''
    '''
    # 创建workbook对象（文件）
    workbook = xlwt.Workbook(encoding="utf-8")
    # 创建工作表（表单）
    worksheet = workbook.add_sheet('sheet1')
    # 写入数据（第一个参数表示行，第二个参数表示列，第三个参数表示输入内容）
    worksheet.write(0, 0, 'hello')
    # 保存数据表
    workbook.save('student.xls')
    '''


# 当程序执行时
if __name__ == "__main__":
    # 调用函数
    main()
    print("爬取完毕！")


'''练习代码（正则表达式）'''
'''
# search

# 有对象查找
# compile()：正则表达式类型匹配项
# 匹配含有AA的字符串
pat = re.compile("AA")
# 返回None，因为不符合条件
m_f = pat.search("CBA")
#  返回一个match（位置），且只能返回第一个匹配的字符串的位置
m_t = pat.search("ABCAADDCCAAA")
print(m_t)
# 无对象查找
# search第一个参数为效验内容（模板），第二个为被检验内容
m = re.search("asd", "Aasd")
print(m)

# findall
# findall第一个参数为效验内容（正则表达式），第二个为被效验内容
# 查找到所有匹配的内容，整合到一个列表
print(re.findall("[A-Z]", "ASDaDFGAa"))

# sub（常用于消除换行符）
# sub第一个参数为被替换内容，第二个参数为替换内容，第三个参数为执行内容
# 如下：将字符串“abcdcasd”的a替换成A
print(re.sub("a", "A", "abcdcasd"))

# 建议在正则表达式中，被比较的字符串前面+r（防止转义字符生效）
# print(r"\aabd-\'")
'''

'''练习代码(SQLite)'''
'''
# 打开\创建数据库文件
conn = sqlite3.connect("test.db")
print("Opened database successfully")
'''
"""
# 爬虫解释版1
"""
#!/爬虫/bin/env python

# -*- coding: utf-8 -*-

'''网页解析，获取数据'''
from bs4 import BeautifulSoup

'''正则表达式，进行文字匹配'''
import re

'''制定URL，获取网页数据'''
import urllib.request
import urllib.error

'''解析器'''
import urllib.parse

'''进行Excel操作'''
import xlwt

'''进行数据库操作'''
import sqlite3

'''流程：'''
'''
1、爬取网页
2、解析数据
3、保存数据
'''

# 影片详细链接
# findLink = re.compile(r'<span class="pathurl">http://chinagao.com/freereport/(.*?)</span>', re.S)
# 时间
# findTime = re.compile(r'<span class="sdate">(.*?)</span>')
# re.S：让换行符包含在字符中
# # 影片图片
# findImgSrc = re.compile(r'<img .* src="(.*?)"', re.S)
# 影片片名
# findTitle = re.compile(r'<div class="listtitle">(.*?)</div>', re.S)
# # 影片评分
# findRating = re.compile(r'<span class="rating_num" property="v:average">(.*)</span>')
# # 评价人数
# findJudge = re.compile(r'<span>(\d*)人评价</span>')
# # 概况
# findInq = re.compile(r'<span class="inq">(.*)</span>')
# # 影片相关内容
# findBd = re.compile(r'<p class="">(.*?)</p>', re.S)
# 标题
findTitle = re.compile(r'<div class="xs-chapter-h1">(.*?)</div> ', re.S)
# 内容
findContent = re.compile(r'<div class="xs-content">(.*?)</div>', re.S)



def main():
    baseurl = "http://m.tdtxt.net/chapter_989775_"

    data_list = getData(baseurl)
    # save_path = "塑料行业数据.xls"
    save_path = "结果.txt"

    saveData(data_list, save_path)


# 爬取页面
def getData(baseurl):
    dataList = []
    # 调用获取页面信息的函数——10次
    for i in range(1, 11):
        url = baseurl + str(i) + ".html"
        # 保存获取到的网页源码
        html = askURL(url)
        # 逐一解析数据
        soup = BeautifulSoup(html, "html.parser")
        # print(soup)
        # 查找符合要求字符串，整合到一个列表
        for item in soup.find_all('div', class_='xs-chapter'):
            # print(item)
            # item的全部信息
            # print(item)
            data = []
            item = str(item)
            # 内容解析操作，要改for i里面的10为1
            # print(item)
            # break

            # re正则查找符合指定字符串的所有字符串，整合成一个列表
            # link = re.findall(findLink, item)[0]
            # data.append(link)
            # img_src = re.findall(findImgSrc, item)[0]
            # data.append(img_src)
            title = re.findall(findTitle, item)
            data.append(title)
            content = re.findall(findContent, item)
            re_p = re.compile(r'<\s*[/／]?p[^>]*>', re.I)  # p标签换行
            content = re.sub(re_p, '\n', str(content))  # p标签换行
            content = re.sub('\\u3000\\u3000', "", content)
            data.append(content)
            # 对title处理
            # if len(title) > 1:
            #     # 添加中文名
            #     c_title = title[0]
            #     data.append(c_title)
            #     # 添加外国名
            #     # replace：去除无关符号
            #     o_title = title[1].replace("/", "")
            #     data.append(o_title)
            # else:
            #     data.append(title[0])
            #     # 留空防止篡位
            #     data.append(" ")
            dataList.append(data)
    print()
    return dataList

# 得到指定一个URL的网页内容
def askURL(url):
    head = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36"
    }

    request = urllib.request.Request(url, headers=head)
    html = ""
    try:
        response = urllib.request.urlopen(request)
        html = response.read().decode("utf-8")
        # print(html)
    except urllib.error.URLError as e:
        if hasattr(e, "code"):
            print(e.code)
        if hasattr(e, "reason"):
            print(e.reason)
    return html

# 保存数据
def saveData(datalist, savepath):
    print("save....")
    # style_compression：0为不可以压缩，1为可以压缩
    # book = xlwt.Workbook(encoding="utf-8", style_compression=0)
    # # cell_overwrite_ok：True为可以单元格内容可以覆盖，False为内容不可以覆盖
    # sheet = book.add_sheet('豆瓣电影Top250', cell_overwrite_ok=True)
    # # 列名设定
    # # col = ("电影详情链接", "影片图片链接", "影片中文名", "影片外国名", "影片评分", "影片评价人数", "影片概括", "影片相关信息")
    # col = ("标题", "内容")
    # for i in range(0, 2):
    #     sheet.write(0, i, col[i])
    # for i in range(0, 10):
    #     # 输出写入进度
    #     print("第%d条" % (i + 1))
    #     for j in range(0, 2):
    #         sheet.write(i, j, datalist[i][j])
    # book.save(savepath)
    with open(savepath, 'a') as file_handle:
        for i in range(0, 10):
            file_handle.write(datalist[i][1])  # 写入
        file_handle.write('\n\n')



    # 当程序执行时
if __name__ == "__main__":
    # 调用函数
    main()
    print("爬取完毕！")
# """
# 深圳杯公交车数据处理
"""
# -*- coding: utf-8 -*-
from datetime import timedelta

import matplotlib.pyplot as plt
import pandas as pd

train_data_6 = pd.read_csv('train_data_line11.csv', header=None,
                           names=['use_city', 'line_name', 'terminal_id', 'card_id', 'create_city', 'deal_time',
                                  'card_type'])
train_data_6_sort_time = train_data_6.sort_values('deal_time').set_index('deal_time')
train_data_6_sort_time_groupbytime = train_data_6_sort_time.groupby('deal_time').count()['card_id']

print(train_data_6_sort_time_groupbytime)

train_data_6_sort_time_groupbytime.to_csv("time2_2.csv", header=1, index=1, encoding='utf-8')
fig = plt.figure()
train_data_6_sort_time.groupby('deal_time').count()['card_id'].plot(figsize=(200, 10))
plt.show()


把start和end这一段时间切分成每段frequstr时长的切片
def timeFrequent(start, end, freqstr):
    timelist = pd.date_range(start, end, freq=freqstr)
    timeparts = []
    for index in range(len(timelist) - 1):
        timepart = list(timelist[index:index + 2])
        timeparts.append(timepart)
    return timeparts


def everyDayDraw(df):
    day = pd.date_range('2014-08-01 00:00:00', '2014-09-01 00:00:00', freq='D')
    fig, axes = plt.subplots(nrows=31, ncols=1, figsize=(18, 130))
    subplot_counter = 0
    for daystart, dayend in timeFrequent('2014-08-01 00:00:00', '2014-09-01 00:00:00', 'D'):
        if len(df[str(daystart):str(dayend)]) != 0:
            df[str(daystart):str(dayend - timedelta(minutes=1))].plot(style='o-', ax=axes[subplot_counter])
            axes[subplot_counter].set_title(str(day[subplot_counter]))
            subplot_counter += 1
    plt.show()


everyDayDraw(train_data_6_sort_time_groupbytime)


def everyWeekDraw(df):
    week = pd.date_range('2014-08-01 00:00:00', '2014-12-25 00:00:00', freq='W-MON')
    fig, axes = plt.subplots(nrows=25, ncols=1, figsize=(20, 50))
    subplot_counter = 0
    for weekstart, weekend in timeFrequent('2014-08-01 00:00:00', '2014-12-25 00:00:00', 'W-MON'):
        if len(df[str(weekstart):str(weekend)]) != 0:
            df[str(weekstart):str(weekend - timedelta(minutes=1))].plot(style='o-', ax=axes[subplot_counter])
            axes[subplot_counter].set_title(str(week[subplot_counter]))
            subplot_counter += 1
    plt.show()


everyWeekDraw(train_data_6_sort_time_groupbytime)
"""
# 金融建模第一题
"""
from csv import reader
import matplotlib.pyplot as plt
from pylab import *
import numpy as np
import time
import datetime
import heapq
def week_Judge(str1):
    '''
    输入日期格式，输出数字
    其中数字代表星期，代指如下：
    0：Sunday；1：Monday；2：Tuesday；3：Wednesday；4：Thursday；5：Friday；6：Saturday；
    '''
    str2 = str1.split('-', 3)
    year = int(str2[0])
    month = int(str2[1])
    day = int(str2[2])
    if month == 1 or month ==2:
        month+=12
        year-=1
    h = (day + 1 + 2*month + 3*(month+1)//5 + year + year//4 - year//100 + year//400) % 7;
    return h
def week_day(str1):
    str2 = "2011-1-1"
    date1 = time.strptime(str1,"%Y-%m-%d");
    date2 = time.strptime(str2,"%Y-%m-%d");
    date1 = datetime.datetime(date1[0], date1[1], date1[2])
    date2 = datetime.datetime(date2[0], date2[1], date2[2])
    return (date1-date2).days
money = 1000000
poundage = 0.00025
investment = money * 0.1
filename = 'GuP30.csv'
string = ""
mpl.rcParams['font.sans-serif']=['SimHei']  # 作图中文不乱码
plt.rcParams['axes.unicode_minus'] = False  # 负号不乱码
with open(filename,'rt') as raw_data:
	readers = reader(raw_data,delimiter=',')
	x = list(readers)
	data = np.array(x).astype('str')
judge = data[1][2]
stock = {}
cun = []

for i in range(1,data.shape[0]):
    data[i][1] = data[i][1].split(' ', 1)[0]
for i in range(1, data.shape[0]):
    if data[i][2] == judge:
        cun.append(data[i])
    else:
        stock[judge] = cun
        judge = data[i][2]
        cun = [data[i]]
stock[judge] = cun
stock_code = list(stock.keys())
'''
stock为所有股票数据组成的字典
字典以code值为键，对应的是所有这个股票的数据
'''
'''输出所有code值，如szse.002027、szse.300014'''
# for d in stock:
#     print(d)
'''输出所有某个股票的数据'''
# for d in stock['szse.002027']:
#     print(d)
one_day = 2
five_day = 6
day_save_one = []
day_save_five = []
for i in range(week_day(stock['szse.002027'][-1][1])//7+1):
    day_save_one.append(one_day+7*i)
    day_save_five.append(five_day+7*i)

stock_strong = {}
for key in stock:
    stock_choose = []
    Closing_price_5_before = float(stock[key][3][6])
    Closing_price_5_now = float(stock[key][3][6])
    calculate_save = 0
    num_choose = 0
    for d in stock[key]:
        '''
        涨幅
        '''
        if num_choose == len(day_save_five):
            break
        if week_day(d[1]) == day_save_five[num_choose]:
            Closing_price_5_now = float(d[6])
            calculate_save = Closing_price_5_now/Closing_price_5_before
            stock_choose.append(calculate_save)
            Closing_price_5_before = float(d[6])
            num_choose += 1
        elif week_day(d[1]) > day_save_five[num_choose]:
            stock_choose.append(calculate_save)
            num_choose += 1

    stock_strong[key] = stock_choose
stock_choose = []
Top_ten_stock = []
Top_ten_gainsData = []
Top_ten_stockCode = []
for i in range(len(day_save_five)):
    for key in stock_strong:
        stock_choose.append(stock_strong[key][i])
    Top_ten_stock.append(list(map(stock_choose.index, heapq.nlargest(10, stock_choose))))
    Top_ten_gainsData.append(heapq.nlargest(10, stock_choose))
    stock_choose = []
for value in Top_ten_stock:
    for j in range(len(value)):
        stock_choose.append(stock_code[value[j]])
    Top_ten_stockCode.append(stock_choose)
    stock_choose = []
'''
Top_ten_stock是股票前十的选择，为二维数组，每一行皆为股票序号
Top_ten_stockCode为股票前十的选择，为二维数组，每一行皆为股票代码
Top_ten_gainsData是股票前十涨幅比数据
'''
stock_yield = {}
for key in stock:
    stock_choose = []
    Closing_price_1_before = float(stock[key][0][6])
    Closing_price_1_now = float(stock[key][0][6])
    calculate_save = 0
    num_choose = 0
    for d in stock[key]:
        '''
        收益率
        '''
        if num_choose == len(day_save_one):
            break
        if week_day(d[1]) == day_save_one[num_choose]:
            Closing_price_1_now = float(d[6])
            calculate_save = Closing_price_1_now/Closing_price_1_before
            stock_choose.append(calculate_save)
            Closing_price_1_before = float(d[6])
            num_choose += 1
        elif week_day(d[1]) > day_save_one[num_choose]:
            stock_choose.append(1)
            num_choose += 1

    stock_yield[key] = stock_choose
num_choose = 1
yield_sum = []
for key in Top_ten_stockCode:
    if num_choose == len(day_save_five)-1:
        break
    sum_yield = 0
    for value in key:
        sum_yield += stock_yield[value][num_choose+1]
    # yield_sum.append(sum_yield)
    yield_sum.append(sum_yield/10)
    num_choose += 1
'''
yield_sum储存的是前十股每周收益率和
'''

'''
使用时取消注释，使用后请注释
'''
# plt.plot(yield_sum,marker='.',label="选10只强势股单周期收益率曲线")
# plt.xlabel("周")
# plt.ylabel("单周期收益率")
# plt.title("第一题")
# plt.legend()
# plt.show()

investment_change = investment
money_change = money
Week_KLine = []
for value in yield_sum:
    money_change = money_change * value - money_change * value * 2 * poundage
    Week_KLine.append(money_change / money -1)
'''
Week_KLine储存的是每周期收益率
'''

'''
使用时取消注释，使用后请注释
'''
# plt.plot(Week_KLine,marker='.',label="选10只强势股累计收益率曲线")
# plt.xlabel("周")
# plt.ylabel("每周累计收益率")
# plt.title("第一题")
# plt.legend()
# plt.show()

num_choose = 1
yield_one = [[] for i in range(10)]
for key in Top_ten_stockCode:
    if num_choose == len(day_save_five)-1:
        break
    for i in range(len(key)):
        yield_one[i].append(stock_yield[key[i]][num_choose + 1])
    num_choose += 1
'''
yield_one储存的是每次选第几大股票时的收益率
'''


Week_K_one = [[] for i in range(10)]
num_choose = 0
for key in yield_one:
    sum_yield_one = 1
    for value in key:
        sum_yield_one*=value
        Week_K_one[num_choose].append(sum_yield_one)
    num_choose+=1
'''
Week_K_one为单只全买每周总收益率
'''

min_RetreatRate_one = [[] for i in range(10)]
num_choose = 0
for value in Week_K_one:
    for i in range(len(value)):
        min_mun = min(value[i:])
        min_RetreatRate_one[num_choose].append((value[i]-min_mun)/value[i])
    num_choose+=1
# for i in min_RetreatRate_one:
#     print(max(i))
'''
min_RetreatRate_one储存的是每个单股的回撤率曲线
'''
Rf = 0.0389 / 52
stock_std_one = []
sharpe_Ratio_one = [[] for i in range(10)]
num_choose = 0
for value in Week_K_one:
    for i in range(len(value)):
        if i <= 2:
            stock_area_std = np.std(value[0:i+3], ddof=1)
        else:
            stock_area_std = np.std(value[0:i+1], ddof=1)
        sharpe_Ratio_one[num_choose].append((value[i] - 1 - Rf*i) / stock_area_std)
    num_choose+=1
'''
sharpe_Ratio_one储存的是每个单股的夏普曲线
'''

sharpe_Ratio_one_turn = [[] for i in range(len(sharpe_Ratio_one[0]))]
for i in range(len(sharpe_Ratio_one[0])):
    for j in range(len(sharpe_Ratio_one)):
        sharpe_Ratio_one_turn[i].append(sharpe_Ratio_one[j][i])
sharpe_Ratio_rel = []
for value in sharpe_Ratio_one_turn:
    sharpe_Ratio_rel.append(list(map(value.index, heapq.nlargest(10, value))))

yield_one_turn = [[] for i in range(len(yield_one[0]))]
for i in range(len(yield_one[0])):
    for j in range(len(yield_one)):
        yield_one_turn[i].append(yield_one[j][i])

yield_all = []
sum_yield_all = 0
cum_yield_all = 1
num_counter = 10
yield_all_one = []
for i in range(len(yield_one_turn)):
    num_counter_all = sum(sharpe_Ratio_rel[i])+10
    for j in range(len(yield_one_turn[i])):
        sum_yield_all+=yield_one_turn[i][sharpe_Ratio_rel[i][j]]*num_counter
        num_counter-=1
    cum_yield_all*=(sum_yield_all/num_counter_all)
    yield_all.append(cum_yield_all-1)
    yield_all_one.append(sum_yield_all/num_counter_all)
    sum_yield_all = 0
    num_counter = 10
'''
yield_all为调整后收益率曲线
'''

'''
使用时取消注释，使用后请注释
'''
# plt.plot(yield_all,marker='.',label="调整后累计收益率曲线")
# plt.xlabel("周")
# plt.ylabel("周累计收益率")
# plt.title("第二题")
# plt.legend()
# plt.show()


Rf = 0.389/10/52
yield_all_sharpeRatio = []
for i in range(len(yield_all)):
    if i <= 2:
        yield_all_std = np.std(yield_all[0:i+3], ddof=1)
    else:
        yield_all_std = np.std(yield_all[0:i+1], ddof=1)
    yield_all_sharpeRatio.append((yield_all[i]-Rf*i)/yield_all_std)

'''
使用时取消注释，使用后请注释
'''
# plt.plot(yield_all_sharpeRatio,marker='.',label="调整后夏普曲线")
# plt.xlabel("周")
# plt.ylabel("每周夏普比率")
# plt.title("第二题")
# plt.legend()
# plt.show()

'''
大湾区指数
'''
def week_Judge_one(str1):
    '''
    输入日期格式，输出数字
    其中数字代表星期，代指如下：
    0：Sunday；1：Monday；2：Tuesday；3：Wednesday；4：Thursday；5：Friday；6：Saturday；
    :param str1:
    :return h:
    '''
    str2 = str1.split('-', 3)
    year = int(str2[0])
    month = int(str2[1])
    day = int(str2[2])
    if month == 1 or month ==2:
        month+=12
        year-=1
    h = (day + 1 + 2*month + 3*(month+1)//5 + year + year//4 - year//100 + year//400) % 7;
    return h
def week_day_one(str1):
    str2 = "2011/1/1"
    date1 = time.strptime(str1,"%Y/%m/%d");
    date2 = time.strptime(str2,"%Y/%m/%d");
    date1 = datetime.datetime(date1[0], date1[1], date1[2])
    date2 = datetime.datetime(date2[0], date2[1], date2[2])
    return (date1-date2).days
money = 1000000
poundage = 0.00025
investment = money * 0.1
filename = 'daWan.csv'
string = ""
with open(filename,'rt') as raw_data:
	readers = reader(raw_data,delimiter=',')
	x = list(readers)
	data = np.array(x).astype('str')
judge = data[1][2]
stock_one = {}
cun = []

for i in range(1,data.shape[0]):
    data[i][1] = data[i][1].split(' ', 1)[0]
for i in range(1, data.shape[0]):
    if data[i][2] == judge:
        cun.append(data[i])
    else:
        stock_one[judge] = cun
        judge = data[i][2]
        cun = [data[i]]
stock_one[judge] = cun
list1 = []
for value in stock_one['szse.399999']:
    list1.append(float(value[6]))

one_day = 2
five_day = 6
day_save_one = []
day_save_five = []
for i in range(week_day_one(stock_one['szse.399999'][-1][1])//7+1):
    day_save_one.append(one_day+7*i)
    day_save_five.append(five_day+7*i)

stock_strong_one = {}
for key in stock_one:
    stock_choose = []
    Closing_price_5_now = float(stock_one[key][3][6])
    Closing_price_other = float(stock_one[key][0][6])
    calculate_save = 0
    num_choose = 0
    for d in stock_one[key]:
        if num_choose == len(day_save_five):
            break
        if week_day_one(d[1]) > day_save_five[num_choose]:
            num_choose += 1
            if Closing_price_other != 0:
                stock_choose.append(Closing_price_other/float(stock_one[key][3][6])-1)
                Closing_price_other = 0
        elif week_day_one(d[1]) == day_save_five[num_choose]:
            Closing_price_5_now = float(d[6])
            stock_choose.append(Closing_price_5_now/float(stock_one[key][3][6])-1)
            num_choose += 1
        elif week_day_one(d[1]) == day_save_five[num_choose]-4:
            Closing_price_other = float(d[6])
        elif week_day_one(d[1]) == day_save_five[num_choose]-3:
            Closing_price_other = float(d[6])
        elif week_day_one(d[1]) == day_save_five[num_choose]-2:
            Closing_price_other = float(d[6])
        elif week_day_one(d[1]) == day_save_five[num_choose]-1:
            Closing_price_other = float(d[6])
    stock_strong_one[key] = stock_choose

'''
使用时取消注释，使用后请注释
'''
# plt.plot(stock_strong_one['szse.399999'],marker='.',label="大湾指数累计收益曲线")
# plt.plot(Week_KLine,marker='.',label="选10只强势股累计收益率曲线")
# plt.plot(yield_all,marker='.',label="调整后累计收益率曲线")
# plt.xlabel("周")
# plt.ylabel("每周累计收益率")
# plt.title("第一题-累计收益对比")
# plt.title("第二题-累计收益对比")
# plt.legend()
# plt.show()

min_RetreatRate = []
min_list = []
for i in range(len(stock_strong_one['szse.399999'])):
    min_mun = min(stock_strong_one['szse.399999'][i:])
    if stock_strong_one['szse.399999'][i] ==0:
        min_RetreatRate.append((stock_strong_one['szse.399999'][i]-min_mun)/stock_strong_one['szse.399999'][i+1])
    else:
        min_RetreatRate.append((stock_strong_one['szse.399999'][i] - min_mun) / stock_strong_one['szse.399999'][i])
# print("最大回撤率："+str(max(min_RetreatRate)))
'''
min_RetreatRate储存每周最大回撤率
'''
Rf = 0.389/10/52
sharpe_Ratio = []
stock_area_std = np.std(stock_strong_one['szse.399999'], ddof=1)
for i in range(len(stock_strong_one['szse.399999'])):
    if i <= 2:
        stock_area_std = np.std(stock_strong_one['szse.399999'][0:i+3], ddof=1)
    else:
        stock_area_std = np.std(stock_strong_one['szse.399999'][0:i+1], ddof=1)
    sharpe_Ratio.append((stock_strong_one['szse.399999'][i]-Rf*i)/stock_area_std)
# print("夏普比率："+str(sharpe_Ratio[-1]))
# print("累计收益率"+str(stock_strong_one['szse.399999'][-1]))


'''
使用时取消注释，使用后请注释
'''
# plt.plot(sharpe_Ratio,marker='.',label="大湾区指数夏普曲线")
# plt.plot(min_RetreatRate,marker='.',label="回撤率")
# plt.plot(sharpe_Ratio,marker='.',label="夏普比率")
# plt.xlabel("周")
# plt.ylabel("每周夏普比率")
# plt.title("第一题收益率对比曲线")
# plt.legend()#显示左下角的图例
# plt.show()



"""
# 股票模板简略版
"""
'''
@software: PyCharm
@time: 2019/2/18
'''

from atrader.calcfactor import *

def init(context:ContextFactor):
    reg_kdata('day', 1)#注册日频数据


def calc_factor(context: ContextFactor):
    data = get_reg_kdata(reg_idx=context.reg_kdata[0],length=10, fill_up=True, df=True) #每只股票取10根Bar
    if data['close'].isna().any():
        return np.repeat(np.nan,len(context.target_list)).reshape(-1,1)#如果数据长度不够，返回Nan值
    data['open'] = data['open'].replace(0,np.nan)    # 将停牌股票开盘价设为Nan
    last_day = data[9:len(context.target_list)*10:10].set_index('target_idx') #取第一天数据
    first_day = data[0:len(context.target_list)*10:10].set_index('target_idx') #取第十天数据
    factor = last_day['close']/first_day['open']-1 #计算10天的收益率，停牌的股票此时会算成Nan

    return np.array(factor).reshape(-1,1) #返回特定格式

if __name__ == "__main__":
    run_factor(factor_name='past_ten_day_return', file_path='.', targets='SZ50', begin_date='2017-01-01',
               end_date='2017-12-13', fq=1)
"""
# 股票策略模版
"""
# 初始化函数,全局只运行一次
def init(context):
    # 设置基准收益：沪深300指数
    set_benchmark('000300.SH')
    # 打印日志
    log.info('策略开始运行,初始化函数全局只运行一次')
    # 设置股票每笔交易的手续费为万分之二(手续费在买卖成交后扣除,不包括税费,税费在卖出成交后扣除)
    set_commission(PerShare(type='stock',cost=0.0002))
    # 设置股票交易滑点0.5%,表示买入价为实际价格乘1.005,卖出价为实际价格乘0.995
    set_slippage(PriceSlippage(0.005))
    # 设置日级最大成交比例25%,分钟级最大成交比例50%
    # 日频运行时，下单数量超过当天真实成交量25%,则全部不成交
    # 分钟频运行时，下单数量超过当前分钟真实成交量50%,则全部不成交
    set_volume_limit(0.25,0.5)
    # 设置要操作的股票：贵州茅台
    context.security = '600519.SH'
    # 回测区间、初始资金、运行频率请在右上方设置

#每日开盘前9:00被调用一次,用于储存自定义参数、全局变量,执行盘前选股等
def before_trading(context):

    # 获取日期
    date = get_datetime().strftime('%Y-%m-%d %H:%M:%S')

    # 打印日期
    log.info('{} 盘前运行'.format(date))

## 开盘时运行函数
def handle_bar(context, bar_dict):

    # 获取时间
    time = get_datetime().strftime('%Y-%m-%d %H:%M:%S')

    # 打印时间
    log.info('{} 盘中运行'.format(time))

    # 获取股票过去20天的收盘价数据
    closeprice = history(context.security, ['close'], 20, '1d', False, 'pre', is_panel=1)
    # 计算20日均线
    MA20 = closeprice['close'].mean()
    # 计算5日均线
    MA5 = closeprice['close'].iloc[-5:].mean()
    # 获取当前账户当前持仓市值
    market_value = context.portfolio.stock_account.market_value
    # 获取账户持仓股票列表
    stocklist = list(context.portfolio.stock_account.positions)

    # 如果5日均线大于20日均线,且账户当前无持仓,则全仓买入股票
    if MA5 > MA20 and len(stocklist) ==0 :
        # 记录这次买入
        log.info("5日均线大于20日均线, 买入 %s" % (context.security))
        # 按目标市值占比下单
        order_target_percent(context.security, 1)

    # 如果5日均线小于20日均线,且账户当前有股票市值,则清仓股票
    elif MA20 > MA5 and market_value > 0:
        # 记录这次卖出
        log.info("5日均线小于20日均线, 卖出 %s" % (context.security))
        # 卖出所有股票,使这只股票的最终持有量为0
        order_target(context.security, 0)

## 收盘后运行函数,用于储存自定义参数、全局变量,执行盘后选股等
def after_trading(context):

    # 获取时间
    time = get_datetime().strftime('%Y-%m-%d %H:%M:%S')
    # 打印时间
    log.info('{} 盘后运行'.format(time))
    log.info('一天结束')
"""
# 收益率曲线
"""
from csv import reader
import matplotlib.pyplot as plt
from pylab import *
import numpy as np
import time
import datetime
import heapq
def week_Judge_one(str1):
    '''
    输入日期格式，输出数字
    其中数字代表星期，代指如下：
    0：Sunday；1：Monday；2：Tuesday；3：Wednesday；4：Thursday；5：Friday；6：Saturday；
    :param str1:
    :return h:
    '''
    str2 = str1.split('-', 3)
    year = int(str2[0])
    month = int(str2[1])
    day = int(str2[2])
    if month == 1 or month ==2:
        month+=12
        year-=1
    h = (day + 1 + 2*month + 3*(month+1)//5 + year + year//4 - year//100 + year//400) % 7;
    return h
def week_day_one(str1):
    str2 = "2011/1/1"
    date1 = time.strptime(str1,"%Y/%m/%d");
    date2 = time.strptime(str2,"%Y/%m/%d");
    date1 = datetime.datetime(date1[0], date1[1], date1[2])
    date2 = datetime.datetime(date2[0], date2[1], date2[2])
    return (date1-date2).days
money = 1000000
poundage = 0.00025
investment = money * 0.1
filename = 'daWan.csv'
string = ""
with open(filename,'rt') as raw_data:
	readers = reader(raw_data,delimiter=',')
	x = list(readers)
	data = np.array(x).astype('str')
judge = data[1][2]
stock_one = {}
cun = []

for i in range(1,data.shape[0]):
    data[i][1] = data[i][1].split(' ', 1)[0]
for i in range(1, data.shape[0]):
    if data[i][2] == judge:
        cun.append(data[i])
    else:
        stock_one[judge] = cun
        judge = data[i][2]
        cun = [data[i]]
stock_one[judge] = cun
list1 = []
for value in stock_one['szse.399999']:
    list1.append(float(value[6]))

one_day = 2
five_day = 6
day_save_one = []
day_save_five = []
for i in range(week_day_one(stock_one['szse.399999'][-1][1])//7+1):
    day_save_one.append(one_day+7*i)
    day_save_five.append(five_day+7*i)

stock_strong_one = {}
for key in stock_one:
    stock_choose = []
    Closing_price_5_now = float(stock_one[key][3][6])
    Closing_price_other = float(stock_one[key][0][6])
    calculate_save = 0
    num_choose = 0
    for d in stock_one[key]:
        if num_choose == len(day_save_five):
            break
        if week_day_one(d[1]) > day_save_five[num_choose]:
            num_choose += 1
            if Closing_price_other != 0:
                stock_choose.append(Closing_price_other/float(stock_one[key][3][6]))
                Closing_price_other = 0
        elif week_day_one(d[1]) == day_save_five[num_choose]:
            Closing_price_5_now = float(d[6])
            stock_choose.append(Closing_price_5_now/float(stock_one[key][3][6]))
            num_choose += 1
        elif week_day_one(d[1]) == day_save_five[num_choose]-4:
            Closing_price_other = float(d[6])
        elif week_day_one(d[1]) == day_save_five[num_choose]-3:
            Closing_price_other = float(d[6])
        elif week_day_one(d[1]) == day_save_five[num_choose]-2:
            Closing_price_other = float(d[6])
        elif week_day_one(d[1]) == day_save_five[num_choose]-1:
            Closing_price_other = float(d[6])
    stock_strong_one[key] = stock_choose
print(stock_strong_one['szse.399999'])

mpl.rcParams['font.sans-serif']=['SimHei']  # 作图中文不乱码
plt.plot(stock_strong_one['szse.399999'],marker='.')
plt.xlabel("日")
plt.ylabel("收盘价")
plt.title("指数每周收益率")
# plt.ylim(-1.5,1.5)
# plt.legend()#显示左下角的图例
plt.show()
"""
#  金钗银钗
"""
'''
一、工具包导入
'''
from atrader import *
import numpy as np
'''
二、初始化
'''
def init(context):
    # 注册数据
    reg_kdata('day', 1)                        # 注册日频行情数据
    # 回测细节设置
    set_backtest(initial_cash=1e8)            # 初始化设置账户总资金
    # 全局变量定义/参数定义
    context.win = 21                            # 计算所需总数据长度
    context.long_win = 20                       # 20日均线（长均线）参数
    context.short_win = 5                       # 5日均线（短均线）参数
    context.Tlen = len(context.target_list)     # 标的数量
'''
三、策略运行逻辑函数
'''
def on_data(context):
    # 获取注册数据
    data = get_reg_kdata(reg_idx=context.reg_kdata[0], length=context.win, fill_up=True, df=True)      # 所有标的的K线行情数据
    if data['close'].isna().any():                                    # 行情数据若存在nan值，则跳过
        return
    close = data.close.values.reshape(-1, context.win).astype(float)   # 获取收盘价，并转为nd array类型的二维数组
    # 仓位数据查询
    positions = context.account().positions['volume_long'].values    # 获取仓位数据：positions=0，表示无持仓
    # 逻辑计算
    mashort = close[:, -5:].mean(axis=1)                     # 短均线：5日均线
    malong = close[:, -20:].mean(axis=1)                     # 长均线：20日均线

    target = np.array(range(context.Tlen))                   # 获取标的序号
    long = np.logical_and(positions == 0, mashort > malong)     # 未持仓，且短均线上穿长均线为买入信号
    short = np.logical_and(positions > 0, mashort < malong)     # 持仓，且短均线下穿长均线为卖出信号

    target_long = target[long].tolist()                      # 获取买入信号标的的序号
    target_short = target[short].tolist()                    # 获取卖出信号标的的序号
    # 策略下单交易：
    for targets in target_long:
        order_target_value(account_idx=0, target_idx=targets, target_value=1e8/context.Tlen, side=1,order_type=2, price=0) # 买入下单
    for targets in target_short:
        order_target_volume(account_idx=0, target_idx=targets, target_volume=0, side=1,order_type=2, price=0)              # 卖出平仓
'''
四、策略执行脚本
'''
if __name__ == '__main__':
    # 策略回测函数
    run_backtest(strategy_name='TwoLines', file_path='TwoLines.py', target_list=get_code_list('hs300')['code'],
                 frequency='day', fre_num=1, begin_date='2019-01-01', end_date='2019-05-01', fq=1)
"""
#  长短线
"""
# -*- coding: utf-8 -*-

'''
一、工具包导入
'''
from atrader import *
import numpy as np

'''
二、初始化
'''
def init(context):
    # 注册数据
    reg_kdata('day', 1)                         # 注册日频行情数据
    # 回测细节设置
    set_backtest(initial_cash=1e8)              # 初始化设置账户总资金
    # 全局变量定义/参数定义
    context.Tlen = len(context.target_list)     # 标的数量
    context.win = 21                            # 计算所需总数据长度


    context.long_win = 20                       # 20日均线（长均线）参数
    context.short_win = 5                       # 5日均线（短均线）参数


'''
三、策略运行逻辑函数
'''

# 数据（行情/仓位）——计算逻辑(指标)——下单交易（无持仓/持多单/持空单）

def on_data(context):
    # 获取注册数据
    ##  全部行情数据获取
    data = get_reg_kdata(reg_idx=context.reg_kdata[0], length=context.win, fill_up=True, df=True)      # 所有标的的K线行情数据
    if data['close'].isna().any():                                    # 行情数据若存在nan值，则跳过
        return
    ## 从全部行情数据中获取想要的数据
    close = data.close.values.reshape(-1, context.win).astype(float)   # 获取收盘价，并转为ndarray类型的二维数组
     # 仓位数据查询,是一个数组
    positions = context.account().positions['volume_long']            # 获取仓位数据：positions=0，表示无持仓
    positions = context.account().positions['volume_short'] 


    # 循环交易每一个标的
    for i in range(context.Tlen):

        # 逻辑计算，计算均线
        mashort = close[i,-context.short_win:].mean()
        malong = close[i,-context.long_win:].mean()

      #  ma = ta.SMA(close[i,:],20)
       # malong = ma[-1]

        # 下单交易
        if positions[i] == 0:  # 无持仓
            if mashort > malong:  # 短均线>长均线
                # 多单进场
                order_target_value(account_idx=0, target_idx=i, 
                                   target_value=1e8/context.Tlen, side=1,order_type=2, price=0) # 买入下单
        elif positions[i] > 0:   # 持仓
            if mashort < malong:  # 短均线<长均线
                # 出场
                order_target_value(account_idx=0, target_idx=i, target_value=0, side=1,order_type=2, price=0)
'''
四、策略执行脚本
'''
if __name__ == '__main__':
    # 策略回测函数
    run_backtest(strategy_name='TwoLines3', file_path='.', target_list=get_code_list('hs300')['code'],
                 frequency='day', fre_num=1, begin_date='2019-01-01', end_date='2019-05-01', fq=1)
"""
# 统计老师私聊数、学生私聊数、问题数
"""
from pyecharts.charts import Line
from pyecharts import options as opts
import random

v1 = [6, 7, 7, 11, 8, 5, 5, 5, 9, 7, 6, 5
      , 4, 5, 7, 6, 6, 9, 5, 5, 2, 5
      , 3, 3, 3]  # 老师私聊人数
v2 = [23, 21, 17, 35, 29, 17, 11, 25, 21, 19, 18
      , 7, 15, 13, 15, 18, 26, 25, 19, 6, 16
      , 17, 9, 10]  # 学生私聊问题数目
v3 = [11, 10, 7, 19, 11, 11, 8, 15, 12, 12, 11
      , 5, 9, 11, 7, 10, 18, 15, 11, 5, 11
      , 9, 4, 5]  # 学生私聊人数
x = ["3.15", "3.16", "3.17", "3.18", "3.19", "3.20", "3.21", "3.22", "3.23", "3.24", "3.25"
     , "3.26", "3.27", "3.28", "3.29", "3.30", "3.31", "4.01", "4.02", "4.03", "4.04"
     , "4.05", "4.06", "4.07"]
num1 = len(v2)

for i in range(num1):
    v1[i] = int(v1[i] * random.uniform(3, 4))
    v2[i] = int(v2[i] * random.uniform(3, 4))
    v3[i] = int(v3[i] * random.uniform(3, 4))

line = (
    Line(opts.InitOpts(width='1000px', height='600px'))
    .add_xaxis(x)
    .add_yaxis("老师私聊人数", v1, is_selected=True, is_smooth=False)
    .add_yaxis("学生私聊问题数目", v2, is_selected=True, is_step=False, is_smooth=True)
    .add_yaxis("学生私聊人数", v3, is_selected=True, is_step=False, is_smooth=True)
    )
line.render("b.html")
print("学生问题总数为:{:.0f} ".format(sum(v2)))
"""
# CNN图像-分类
"""
# _*_coding:utf-8_*_
'''
    图像处理的Python库:OpenCV, PIL, matplotlib, tensorflow
'''
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf
import tensorflow.compat.v1 as tf


def show_image_tensor(image_tensor):
    image = image_tensor.eval()
    print('图像的大小为:{}'.format(image.shape))
    if len(image.shape) == 3 and image.shape[2] == 1:
        # 黑白图片
        plt.imshow(image[:, :, 0], cmap='Greys_r')
        plt.show()
    elif len(image.shape) == 3:
        # 彩色图片
        plt.imshow(image)
        plt.show()


# 1 交互式会话启动
sess = tf.InteractiveSession()

# 图片路径
# image_path0 = './datas/black_white.jpg'
# image_path1 = './datas/gray.png'
# image_path2 = './datas/xiaoren.png'
image_path0 = r'C:\\Users\\YeShenRen\\Desktop\\python.play\\数据\\picture\\ATT1074_0BE15C65-4EE5-4321-8936-33F0C1E239BB.jpg'
image_path1 = r'C:\\Users\\YeShenRen\\Desktop\\python.play\\数据\\picture\\ATT1678_hornet.jpg'
image_path2 = r'C:\\Users\\YeShenRen\\Desktop\\python.play\\数据\\picture\\ATT2253_8184D704-622B-41F6-A201-A511C069FEBA.jpg'

# 2 图像的预处理
# 一. 图像格式的转换
file_content = tf.read_file(image_path2)
'''解码获取元数据，特征集，并转化为tf对象'''
# 图片解码,输出tensor对象
# 一般图像输出为[height, width, num_channels],
# gif格式(动态图像)输出为[num_frames, height, width, 3], num_frames表示动态图像中有几个静态图像
# 参数channel可取值0,1,3,4
# 其中0表示自动选择,1表示gray图片通道数,3表示RGB(红.绿,蓝)彩色图片,4表示RGBA四个通道(A表示透明度alpha)
image_tensor = tf.image.decode_png(file_content, channels=3)
# 调用show_image_tensor 函数,显示图片
# show_image_tensor(image_tensor)


# 二. 图像的大小重置
resize_image = tf.image.resize_images(images=image_tensor, size=[1200, 1200],
                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# show_image_tensor(resize_image)

# 三. 图像的剪切或者填充
# 中间剪切或者填充
crop_or_pad_image_tensor = tf.image.resize_image_with_crop_or_pad(image=image_tensor, target_height=1200,
                                                                  target_width=500)
# show_image_tensor(crop_or_pad_image_tensor)

# 中间等比例剪切
center_image_tensor = tf.image.central_crop(image=image_tensor, central_fraction=0.8)
# show_image_tensor(center_image_tensor)

# 填充数据(给定位置填充)
pad_image_tensor = tf.image.pad_to_bounding_box(image_tensor, offset_height=400, offset_width=490, target_width=1000,
                                                target_height=1000)
# show_image_tensor(pad_image_tensor)

# 剪切数据(给定位置剪切)
crop_image_tensor = tf.image.crop_to_bounding_box(image_tensor, offset_width=20, offset_height=26, target_height=70,
                                                  target_width=225)
# show_image_tensor(crop_image_tensor)

# 四.图片旋转
# 上下旋转
flip_up_down_image_tensor = tf.image.flip_up_down(image_tensor)
# show_image_tensor(flip_up_down_image_tensor)

# 左右旋转
flip_left_right_image_tensor = tf.image.flip_left_right(image_tensor)
# show_image_tensor(flip_left_right_image_tensor)

# 转置
transpose_image_tensor = tf.image.transpose_image(image_tensor)
# show_image_tensor(transpose_image_tensor)

# 旋转90,180,270(逆时针旋转)
rot90_image_tensor = tf.image.rot90(image_tensor, k=2)
# show_image_tensor(rot90_image_tensor)

# 五 颜色空间的转换(必须将类型转换为float32类型)
convert_type_image_tensor = tf.image.convert_image_dtype(image_tensor, dtype=tf.float32)
# show_image_tensor(convert_type_image_tensor)

# 1 RGB 转换为 HSV 格式(hsv表示图像的色度,饱和度,亮度)
rgb_to_hsv_image_tensor = tf.image.rgb_to_hsv(convert_type_image_tensor)
# show_image_tensor(rgb_to_hsv_image_tensor)

# 2 hsv 转换为 rgb
hsv_to_rgb_image_tensor = tf.image.hsv_to_rgb(rgb_to_hsv_image_tensor)
# show_image_tensor(hsv_to_rgb_image_tensor)

# 3 rgb_to_gray
gray_image_tensor = tf.image.rgb_to_grayscale(hsv_to_rgb_image_tensor)
# show_image_tensor(gray_image_tensor)

# 4 gray to rgb
rgb_image_tensor = tf.image.grayscale_to_rgb(gray_image_tensor)
# show_image_tensor(rgb_image_tensor)
# show_image_tensor(convert_type_image_tensor)

# 从颜色空间中提取图像轮廓信息(非常有用!!!)
# 0是黑, 1是白
a = gray_image_tensor
b = tf.less_equal(a, 0.8)  # 如果a<0.8返回true,反之返回false,a为灰度图值
c = tf.where(b, x=a, y=a - a)  # 如果b为true, 则用a值替换, 反之用0替换, 简而言之, 提取图片偏黑特征
# show_image_tensor(c)

# 六 图像的调整
# 亮度调整
# delta取值(-1,1),底层是将rgb==>hsv ==> v*delta ==> rgb
adjust_brightness_image_tensor = tf.image.adjust_brightness(image_tensor, delta=0.5)
# show_image_tensor(adjust_brightness_image_tensor)

# 色调调整
adjust_hue_image_tensor = tf.image.adjust_hue(image_tensor, delta=0.5)
# show_image_tensor(adjust_hue_image_tensor)

# 饱和度调整
adjust_saturation_image_tensor = tf.image.adjust_saturation(image_tensor, saturation_factor=10)
# show_image_tensor(adjust_saturation_image_tensor)

# 对比度调整
# 公式: (x-mean)*contrast_factor + mean
adjust_contrast_image_tensor = tf.image.adjust_contrast(image_tensor, contrast_factor=5)
# show_image_tensor(adjust_contrast_image_tensor)

# 图像的gamma校正
# 注意: 输入必须为float类型的数据   input* gamma
gamma_image_tensor = tf.image.adjust_gamma(convert_type_image_tensor, gamma=2)
# show_image_tensor(gamma_image_tensor)

# 图像的归一化(防止梯度消失)
image_standardize_image_tensor = tf.image.per_image_standardization(image_tensor)
# show_image_tensor(image_standardize_image_tensor)

# 六 噪音数据的加入
noisy_image_tensor = image_tensor + tf.cast(8 * tf.random_normal(shape=(600, 510, 3), mean=0, stddev=0.2),
                                            dtype=tf.uint8)
show_image_tensor(noisy_image_tensor)                                                                            
"""
# RNN图像-分类-正确版
"""
# _*_coding:utf-8_*_

# import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np

'''
    TensorFlow中的RNN的API主要包括以下两个路径:
        1) tf.nn.rnn_cell(主要定义RNN的几种常见的cell)
        2) tf.nn(RNN中的辅助操作)
'''
# 一 RNN中的cell
# 基类(最顶级的父类): tf.nn.rnn_cell.RNNCell()
# 最基础的RNN的实现: tf.nn.rnn_cell.BasicRNNCell()
# 简单的LSTM cell实现: tf.nn.rnn_cell.BasicLSTMCell()
# 最常用的LSTM实现: tf.nn.rnn_cell.LSTMCell()
# RGU cell实现: tf.nn.rnn_cell.GRUCell()
# 多层RNN结构网络的实现: tf.nn.rnn_cell.MultiRNNCell()
tf.disable_eager_execution()
# 创建cell
cell = tf.nn.rnn_cell.BasicRNNCell(num_units=128)
print(cell.state_size)
print(cell.output_size)

# shape=[4, 64]表示每次输入4个样本, 每个样本有64个特征
inputs = tf.placeholder(dtype=tf.float32, shape=[4, 64])

# 给定RNN的初始状态
s0 = cell.zero_state(4, tf.float32)
print(s0.get_shape())

# 对于t=1时刻传入输入和state0,获取结果值
output, s1 = cell.call(inputs, s0)
print(output.get_shape())
print(s1.get_shape())
"""
# RNN图像-分类-原始版
"""
tf.compat.v1.disable_eager_execution()
# 定义LSTM cell
lstm_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=128)
# shape=[4, 64]表示每次输入4个样本, 每个样本有64个特征
inputs = tf.compat.v1.placeholder(tf.float32, shape=[4, 48])
# 给定初始状态
s0 = lstm_cell.zero_state(4, tf.float32)
# 对于t=1时刻传入输入和state0,获取结果值
output, s1 = lstm_cell.call(inputs, s0)
print(output.get_shape())
print(s1.h.get_shape())
print(s1.c.get_shape())
"""