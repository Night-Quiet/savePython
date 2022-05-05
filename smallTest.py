"""
 目录 ------------------------------------------------------
 * 超级马里奥
 * 超级马里奥1
 * 过河卒
 * 求最小的既是质数又是回文数的数
 * 求36按某种速率翻一倍所需时间
 * 画一条蛇
 * 画一条蛇1
 * 简单输入输出
 * 简单的类型转换、计算
 * 数组
 * 画叠加正方形
 * 画叠加正六边形
 * 判断输入
 * 绩点判断是否需要补考
 * 水果判断是否有货
 * 无限输入单词
 * 车票收费统计
 * 判断字符串是否处理数组中
 * 单词本 简单输入输出
 * 字典的增删查改
 * 单词中英文翻译
 * 画小猪佩奇
 * 无限输入，直到quit停止
 * 无限输入，直到quit停止
 * 画同点圆
 * 处理，储存一组数据的大于0的奇数、偶数
 * 格式化输出
 * 数组验证其内容
 * 删除数组全部指定字符串
 * 单词本插入
 * 函数
 * 函数引用
 * 数组排序
 * 二分图
 * 类
 * 文件操作
 * 异常处理
 * 数据库处理
 * 折线图至html
 * 打印每天信息
 * pyecharts画热力图
 * pyecharts画地图
 * 使用pandas读取excel文件
 * pyecharts画词云
 * pyecharts分析《围城》词云
 * 螺旋输出顺序数字
 * 
 * 
 * 
"""

# 超级马里奥
"""
print("                ********")
print("               ************")
print("               ####....#.")
print("             #..###.....##....")
print("             ###.......######              ###            ###")
print("                ...........               #...#          #...#")
print("               ##*#######                 #.#.#          #.#.#")
print("            ####*******######             #.#.#          #.#.#")
print("           ...#***.****.*###....          #...#          #...#")
print("           ....**********##.....           ###            ###")
print("           ....****    *****....")
print("             ####        ####")
print("           ######        ######")
print("##############################################################")
print("#...#......#.##...#......#.##...#......#.##------------------#")
print("###########################################------------------#")
print("#..#....#....##..#....#....##..#....#....#####################")
print("##########################################    #----------#")
print("#.....#......##.....#......##.....#......#    #----------#")
print("##########################################    #----------#")
print("#.#..#....#..##.#..#....#..##.#..#....#..#    #----------#")
print("##########################################    ############")
"""
# 超级马里奥1
"""
print(
                 ********
               ************
               ####....#.
             #..###.....##....
             ###.......######              ###            ###
                ...........               #...#          #...#
               ##*#######                 #.#.#          #.#.#
            ####*******######             #.#.#          #.#.#
           ...#***.****.*###....          #...#          #...#
           ....**********##.....           ###            ###
           ....****    *****....
             ####        ####
           ######        ######
##############################################################
#...#......#.##...#......#.##...#......#.##------------------#
###########################################------------------#
#..#....#....##..#....#....##..#....#....#####################
##########################################    #----------#
#.....#......##.....#......##.....#......#    #----------#
##########################################    #----------#
#.#..#....#..##.#..#....#..##.#..#....#..#    #----------#
##########################################    ############
)
"""
# 过河卒
"""
a, b, c, d = input().split()
ha = True
a = int(a)
b = int(b)
c = int(c)
d = int(d)
e = f = n = 0
HaHa = True
list1 = []
list1.append((c, d))
list1.append((c+1, d+2))
list1.append((c+2, d+1))
list1.append((c-1, d-2))
list1.append((c-2, d-1))
list1.append((c-1, d+2))
list1.append((c-2, d+1))
list1.append((c+2, d-1))
list1.append((c+1, d-2))
n, m, j, k = input().split(" ")
zuo = False
you = False
while HaHa:
    for i in range(int(n)+int(m)):
        if (e+1, f) not in list1 and e != int(n):
            zou = True
        if (e, f+1) not in list1 and f != int(m):
            you = True
        if zuo and True:
            if i == 11:
                n += 1
e = f = 0
"""
# 求最小的既是质数又是回文数的数
"""
import math
def prime(x):
    dx = int(math.sqrt(x))
    for k in range(2, dx+1):
        if x % k == 0:
            return 0
    return 1
def palindrome(y):
    y = str(y)
    num = len(y)
    for j in range(num//2):
        if y[j] != y[num-j-1]:
            return 0
    return 1
a, b = input().split(" ")
for i in range(int(a), int(b)):
    if prime(i) == 1 and palindrome(i) == 1:
        print(i)
"""
# 求36按某种速率翻一倍所需时间
"""
import math
t = 0
y = x1 = 36
r = 0.021
num = 72
while y < num:
    t += 1
    y = x1*math.exp(r*t)
print(t)
"""
# 画一条蛇
"""
import turtle

turtle.setup(650, 350, 200, 200)
turtle.penup()
turtle.fd(-250)
turtle.pendown()
turtle.pensize(25)
turtle.pencolor("purple")
turtle.seth(-40)
for i in range(4):
    turtle.circle(40, 80)
    turtle.circle(-40, 80)
turtle.circle(40, 80 / 2)
turtle.fd(40)
turtle.circle(16, 180)
turtle.fd(40 * 2 / 3)
turtle.done()
import turtle as t

t.left(45)
t.fd(150)
t.right(135)
t.fd(300)
t.left(135)
t.fd(200)
t.circle(80, -360)
t.seth(-135)
t.fd(300)
t.goto(-100, 100)
"""
# 画一条蛇1
"""
import turtle as t
t.setup(650,350,200,200)
t.penup()
t.goto(-300,-100)
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
t.done()
"""
# 简单输入输出
"""
'''
# 输出 hello world!
message = 'hello world!'
print(message)
'''
'''
# 输出 你好
message = '你好'
print(message)
'''
'''
# 输出 圆周长
pi = 3.1415926
meter = 2*pi*5.5
print("圆的周长：", meter)
'''
'''
# 输入你的姓名 输出 问候
name = input("输入你的姓名：")
print(name,"，你好！")
'''
'''
# 输出hello的开头大写、全大写、全小写
name = "HeLlo"
print(name.title())
print(name.upper())
print(name.lower())
'''
'''
# 去点前面空格、去掉后面空格、去掉两边空格
name = "HeLlo"
names = "\t"+name+name.title()+"\t"
print(names.lstrip())
print(names.rstrip())
print(names.strip())
'''
"""
# 简单的类型转换、计算
"""
'''
# 数字转换字符串
ha = 3.21312
str(ha)
print(ha)
'''
'''
# 连接输出字符串
print("9/2=",9/2)
print("9//2=",9//2)
print("9**2=",9**2)
'''
'''
# 计算
a = 3
b = 4
r = a*a+b*b
print(r)
'''
'''
# 输出小数
a=float(0.2)
b=float(0.1)
print("a + b = ",float(a+b))
'''
'''
# 输入带说明
name = input("请你输入你的大名（当然，我不建议你说自己是垃圾：")
print(name)
# a = int(input("请输入第一个整数垃圾："))
# b = int(input("请输入第二个整数垃圾："))
# print(a + b)
'''
'''
# 格式输出带数字字符串
a = 1233.1415926
print("{:f}".format(a))
'''
'''
# 字符串转换数字
number1 = int(input("混蛋，你的第一个数字是啥，快说："))
number2 = int(input("垃圾，你的第二个数字是啥，快讲："))
print("垃圾，这是两个数字之和"+str(number1+number2))
print("垃圾，这是两个数字之差"+str(number1-number2))
print("垃圾，这是两个数字之积"+str(number1*number2))
print("垃圾，这是两个数字之商"+str(number1/number2))
'''
"""
# 数组
"""
'''
# 数组的建立、输出
fruits = ['apple','pear','banana','orange']
print(fruits[0])
scores = [99.5,100,97.5]
print(scores)
ns = ['python',10000,77.5,'加油']
print(ns)
bicycles = ['trek', 'cannonade', 'recline', '捷安特', '凤凰', '永久' ]
print("倒数的索引：")
print(bicycles[-1])
print(bicycles[-3])
print(fruits)
name = []
for i in range(3):
    name.append(input())
name.sort()
for i in range(3):
    print(name[i])
squares = []
for value in range(1,11):
    square = value**2
    squares.append(square)
print(squares)
square = [valuer**2 for valuer in range(1,11)]
print(square)
numbers = range(1,6)
print("numbers:", numbers)
num = list(range(20,11,-2))
print("num:",num)
numbers = list(range(10))
print("sum(numbers):", sum(numbers))
print("max(numbers):", max(numbers))
print("min(numbers):", min(numbers))
for ma in mag:
    print(ma.title()+',that was a great trick')
    print("I can't wait to see your next trick,"+ma.title()+".")
print("Thank you everyone, that was a great magic show!")
print("type(numbers):", type(numbers))
print("numbers内的列表：", list(numbers))
imx = [
    [1,2,3],
    [4,5,6]
]
print(imx)
players = ['charles', 'martina', 'michael', 'florence', 'eli']
print(players[1:4])
players = ['charles', 'martina', 'michael', 'florence', 'eli']
print("Here are the first three players on my team:")
for player in players[:3]:
    print(player.title())
players = ['charles', 'martina', 'michael', '马龙', '孙杨']
# -1是最后一个，-2是倒数第二，-3是倒数第三，...
print("[-2:-1]:", players[-2:-1])
print("[-3:  ]:", players[-3:])
print(players[:-2])
'''
'''
# 数组的增删查改排
fruits[1] = 'pineapple'
print(fruits)
fruits.append('watermelon')
print(fruits)
fruits.insert(0,'grapes')
print(fruits)
del fruits[0]
print(fruits)
fruits = fruits.pop()
print(fruits)
fruits = fruits.pop(1)
print(fruits)
fruits.sort()
print(fruits)
print(sorted(fruits))
fruits.sort(reverse=True)
print(fruits)
fruits.reverse()
print(fruits)
print(len(fruits))
# numbers = []
# for i in range(10):
#     number = int(input())
#     numbers.append(number)
# for i in range(3):
#     print(max(numbers))
#     numbers.remove(max(numbers))
# friends = ['das','dsa','eye','adv','gsd']
# print(friends[-3:])
# my_foods = ['pizza', 'falasha', 'carrot cake']
# friend_foods = my_foods[:]
# my_foods.append('cannily')
# friend_foods.append('ice cream')
# print("My favorite foods are:")
# print(my_foods)
# print("\nMy friend's favorite foods are:")
# print(friend_foods)
# players = ['charles', 'martina', 'michael', '马龙', '孙杨']
# print("1. players=", players)
# plist = players
# plist.append('姚明')
# print("2. players=", players)
# players = ['charles', 'martina', 'michael', '马龙', '孙杨']
# print("3. players=", players)
# plist = players[:]
# plist.append('姚明')
# print("4. players=", players)
# print("5. plist=", plist)
# my_foods = ['pizza', 'rice', 'milk']
# your_foods = ['口味虾', '红烧肉', '馒头']
# foods = my_foods + your_foods
# print("6. foods=", foods)
# your_foods[0] = '剁椒鱼头'
# print("7. foods=", foods)
# dimensions = [200, 50]
# print("Original dimensions:")
# for dimension in dimensions:
#     print(dimension)
# friends = ['罗曼迪康蒂','加里奥','亚索','龙瞎','东方耀','何以琛','徐建国','没毛病','胡打样']
# classmates = friends[:]
# classmates.remove('亚索')
# print(friends)
# print(classmates)
num = (200, 50, 300, 400)
print(num[0])
print(num[1])
cars = ['audi','bmw','subaru','ban']
for car in cars:
    if car == 'bmw':
        print(car.upper())
    else:
        print(car.title())
'''
"""
# 画叠加正方形
"""
import turtle as t
t.setup(800, 800)
for i in range(1, 200):
    t.forward(2*i)
    t.left(90)
for i in range(1, 21, 1):
    print(i)
le = [nu for nu in range(3, 31, 3)]
print(le)
ha = [na**3 for na in range(1, 11, 1)]
print(ha)
"""
# 画叠加正六边形
"""
import turtle as t
t.setup(800, 800)
for i in range(1, 201, 1):
    t.forward(2*i)
    t.left(60)
"""
# 判断输入
"""
# end = input("请输入一个字符串：")
# if end == 'end':
#     print("输出的是end。")
# else:
#     print("输出的不是end。")
# if end > 'end':
#     print("输入的字符串大于end。")
# if end < 'end':
#     print("输入的字符串小于end")
"""
# 绩点判断是否需要补考
"""
# score = input("请输入你的绩点：")
# class = input("请输入你的成绩：")
# if float(score) > 3 and float(class) >400:
#     print("你被录取了。")
# else:
#     print("对不起，你没有被录取。")
# physical = int(input("请输入你的物理成绩："))
# chemical = int(input("请输入你的化学成绩："))
# if physical >= 60 and chemical >=60:
#     print("你不需要补考。")
# else:
#     print("你需要补考。")
"""
# 水果判断是否有货
"""
# fruits = ["apple","banana","pear"]
# fruit = input("顾客提问水果名字是：")
# if fruit in fruits:
#     print("有货。")
# if fruit not in fruits:
#     print("没货。")
"""
# 无限输入单词
"""
# words = []
# for i in range(1,101):
#     word = input("输入的单词是：")
#     if word == "停止":
#         break
#     if word not in words:
#         words.append(word)
# print("单词清单：",words)
"""
# 车票收费统计
"""
# sum = 0
# num = int(input("请输入你们的人数："))
# for i in range(1, num+1,1):
#     age = int(input("请输入您的年龄："))
#     if age == 0:
#         break
#     elif age >=4 and age <18:
#         sum+=5
#     elif age >=18:
#         sum+=10
# print(sum)
"""
# 判断字符串是否处理数组中
"""
available_toppings = ['mushrooms', 'olives', 'green peppers', 'pepperoni', 'pineapple', 'extra cheese']
requested_toppings = ['mushrooms', 'french fries', 'extra cheese']

for requested_topping in requested_toppings:
    if requested_topping in available_toppings:
        print("Adding " + requested_topping + ".")
    else:
        print("Sorry, we don't have " + requested_topping + ".")
print("\nFinished making your pizza!")
"""
# 单词本 简单输入输出
"""
# words = []
# for i in range(1,5,1):
#     word = input("请输入单词：")
#     if word not in words:
#         words.append(word)
# ban = words[:]
# for i in ban:
#     print(i)
#     print("请问你记住了吗？")
#     answer = input()
#     if answer.lower() == "y":
#         words.remove(i)
# print(words)
"""
# 字典的增删查改
"""
word_dict = {
    'name':'名字',
    'python':'蟒蛇',
    'dictionary':'字典',
    'list':'列表',
    'variable':'变量',
    'class':'类',
    'object':'对象'
}
contacts = {
    '马云':  '13309283335',
    '赵龙':  '18989227822',
    '张敏':  '13382398921',
    '乔治':  '19833824743',
    '乔丹':  '18807317878',
    '库里':  '15093488129',
    '韦德':  '19282937665'
}
name = input()
print(contacts[name])
contacts['耶稣'] = '12345678910'
del contacts['马云']
for name,phone in contacts.items():
    print(name + ":" + phone)
for name in contacts.keys():
    print(name)
for phone in contacts.values():
    print(phone)
contacts = {
    '马云':  {'phone':'13309283335','address':'南校'},
    '赵龙':  {'phone':'18989227822','address':'北校'}
}
print(contacts['马云']['address'])
contacts = {
    '马云':  ['13309283335', '13863381383'],
    '赵龙':  ['18989227822']
}
print(contacts['马云'][1])
words = {
    'apple':['苹果','牛气'],
    'banana':'香蕉',
    'pear':'梨'
}
"""
# 单词中英文翻译
"""
for i in range(1,101,1):
    e = input('请输入需要增加的英文：')
    if e == 'end':
        break
    c = input('请输入对应的中文：')
    words[e] = c
words1 = words.copy()
for e,c in words1.items():
    print(e + ":" + c)
    answer = input("请问你记住了吗？（回复y/n）：")
    if answer == "y":
        del words[e]
print(words)
for i in range(1,101,1):
    answer = input("请输入你想要查找的英文翻译：")
    if answer == "end":
        break
    if answer not in words1.keys():
        print("单词表没有这个单词。")
    else:
        print(answer + "该单词的中文翻译是：" + words1[answer])
for i in range(1,101,1):
    sum=0
    c = input("请输入你想要翻译的中文：" )
    if c == "停止":
        break
    else:
        for eh,ch in words.items():
            if len(ch) == 1:
                if ch == c:
                    print(eh)
                    sum=1
            elif len(ch) > 1:
                if c in ch:
                    print(eh)
                    sum=1
    if sum==0:
         print("对不起，没有这组单词。" )
"""
# 画小猪佩奇
"""
# # coding:utf-8
# import turtle as t
# # 绘制小猪佩奇
# # =======================================
#
# t.pensize(4)
# t.hideturtle()
# t.colormode(255)
# t.color((255, 155, 192), "pink")
# t.setup(840, 500)
# t.speed(50)
#
# # 鼻子
# t.pu()
# t.goto(-100,100)
# t.pd()
# t.seth(-30)
# t.begin_fill()
# a = 0.4
# for i in range(120):
#     if 0 <= i < 30 or 60 <= i < 90:
#         a = a+0.08
#         t.lt(3)  # 向左转3度
#         t.fd(a)  # 向前走a的步长
#     else:
#         a = a-0.08
#         t.lt(3)
#         t.fd(a)
#         t.end_fill()
#
# t.pu()
# t.seth(90)
# t.fd(25)
# t.seth(0)
# t.fd(10)
# t.pd()
# t.pencolor(255, 155, 192)
# t.seth(10)
# t.begin_fill()
# t.circle(5)
# t.color(160, 82, 45)
# t.end_fill()
#
# t.pu()
# t.seth(0)
# t.fd(20)
# t.pd()
# t.pencolor(255, 155, 192)
# t.seth(10)
# t.begin_fill()
# t.circle(5)
# t.color(160, 82, 45)
# t.end_fill()
#
# # 头
# t.color((255, 155, 192), "pink")
# t.pu()
# t.seth(90)
# t.fd(41)
# t.seth(0)
# t.fd(0)
# t.pd()
# t.begin_fill()
# t.seth(180)
# t.circle(300, -30)
# t.circle(100, -60)
# t.circle(80, -100)
# t.circle(150, -20)
# t.circle(60, -95)
# t.seth(161)
# t.circle(-300, 15)
# t.pu()
# t.goto(-100, 100)
# t.pd()
# t.seth(-30)
# a = 0.4
# for i in range(60):
#     if 0 <= i < 30 or 60 <= i <90:
#         a = a+0.08
#         t.lt(3)  # 向左转3度
#         t.fd(a)  # 向前走a的步长
#     else:
#         a = a-0.08
#         t.lt(3)
#         t.fd(a)
#         t.end_fill()
#
# # 耳朵
# t.color((255, 155, 192), "pink")
# t.pu()
# t.seth(90)
# t.fd(-7)
# t.seth(0)
# t.fd(70)
# t.pd()
# t.begin_fill()
# t.seth(100)
# t.circle(-50, 50)
# t.circle(-10, 120)
# t.circle(-50, 54)
# t.end_fill()
#
# t.pu()
# t.seth(90)
# t.fd(-12)
# t.seth(0)
# t.fd(30)
# t.pd()
# t.begin_fill()
# t.seth(100)
# t.circle(-50, 50)
# t.circle(-10, 120)
# t.circle(-50, 56)
# t.end_fill()
#
# #眼睛
# t.color((255, 155, 192), "white")
# t.pu()
# t.seth(90)
# t.fd(-20)
# t.seth(0)
# t.fd(-95)
# t.pd()
# t.begin_fill()
# t.circle(15)
# t.end_fill()
#
# t.color("black")
# t.pu()
# t.seth(90)
# t.fd(12)
# t.seth(0)
# t.fd(-3)
# t.pd()
# t.begin_fill()
# t.circle(3)
# t.end_fill()
#
# t.color((255, 155, 192), "white")
# t.pu()
# t.seth(90)
# t.fd(-25)
# t.seth(0)
# t.fd(40)
# t.pd()
# t.begin_fill()
# t.circle(15)
# t.end_fill()
#
# t.color("black")
# t.pu()
# t.seth(90)
# t.fd(12)
# t.seth(0)
# t.fd(-3)
# t.pd()
# t.begin_fill()
# t.circle(3)
# t.end_fill()
#
# # 腮
# t.color((255, 155, 192))
# t.pu()
# t.seth(90)
# t.fd(-95)
# t.seth(0)
# t.fd(65)
# t.pd()
# t.begin_fill()
# t.circle(30)
# t.end_fill()
#
# # 嘴
# t.color(239, 69, 19)
# t.pu()
# t.seth(90)
# t.fd(15)
# t.seth(0)
# t.fd(-100)
# t.pd()
# t.seth(-80)
# t.circle(30, 40)
# t.circle(40, 80)
#
# # 身体
# t.color("red", (255, 99, 71))
# t.pu()
# t.seth(90)
# t.fd(-20)
# t.seth(0)
# t.fd(-78)
# t.pd()
# t.begin_fill()
# t.seth(-130)
# t.circle(100,10)
# t.circle(300,30)
# t.seth(0)
# t.fd(230)
# t.seth(90)
# t.circle(300,30)
# t.circle(100,3)
# t.color((255,155,192),(255,100,100))
# t.seth(-135)
# t.circle(-80,63)
# t.circle(-150,24)
# t.end_fill()
#
# # 手
# t.color((255,155,192))
# t.pu()
# t.seth(90)
# t.fd(-40)
# t.seth(0)
# t.fd(-27)
# t.pd()
# t.seth(-160)
# t.circle(300,15)
# t.pu()
# t.seth(90)
# t.fd(15)
# t.seth(0)
# t.fd(0)
# t.pd()
# t.seth(-10)
# t.circle(-20,90)
#
# t.pu()
# t.seth(90)
# t.fd(30)
# t.seth(0)
# t.fd(237)
# t.pd()
# t.seth(-20)
# t.circle(-300,15)
# t.pu()
# t.seth(90)
# t.fd(20)
# t.seth(0)
# t.fd(0)
# t.pd()
# t.seth(-170)
# t.circle(20,90)
#
# # 脚
# t.pensize(10)
# t.color((240,128,128))
# t.pu()
# t.seth(90)
# t.fd(-75)
# t.seth(0)
# t.fd(-180)
# t.pd()
# t.seth(-90)
# t.fd(40)
# t.seth(-180)
# t.color("black")
# t.pensize(15)
# t.fd(20)
#
# t.pensize(10)
# t.color((240, 128, 128))
# t.pu()
# t.seth(90)
# t.fd(40)
# t.seth(0)
# t.fd(90)
# t.pd()
# t.seth(-90)
# t.fd(40)
# t.seth(-180)
# t.color("black")
# t.pensize(15)
# t.fd(20)
#
# # 尾巴
# t.pensize(4)
# t.color((255, 155, 192))
# t.pu()
# t.seth(90)
# t.fd(70)
# t.seth(0)
# t.fd(95)
# t.pd()
# t.seth(0)
# t.circle(70, 20)
# t.circle(10, 330)
# t.circle(70, 30)
# t.done()
"""
# 无限输入，直到quit停止
"""
# message = input("-->")
# while message != "quit":
#     print('a str')
#     message = input("-->")
# print("bye")
"""
# 无限输入，直到quit停止
"""
# while True:
#     message = input("-->")
#     if message == "quit":break
#     print(message)
# print("bye")
"""
# 画同点圆
"""
import turtle as turtle
turtle.setup(300, 500)
i = 1
step = 5
draw = True
while draw:
    turtle.circle(step*i)
    i = i + 1
    if i > 100:
        draw = False
    if step*i > 30:
        draw = False
"""
# 处理，储存一组数据的大于0的奇数、偶数
"""
ss = ['2','3','4','5','-3','-5','6','23']
evens = []
odds = []
for s in ss:
    n = int(s)
    if n < 0:
        continue
    if n % 2 == 0:
        evens.append(n)
    else:
        odds.append(n)
print(evens)
print(odds)
print(ss)
"""
# 格式化输出
"""
i = 0
while i < 5:
    j = 0
    while j < 5:
        print("*", end="")
        j = j + 1
    i = i + 1
    print()
"""
# 数组验证其内容
"""
unconfirmed_users = ['alice', 'brian', 'candace']
confirmed_users = []
while unconfirmed_users:
    current_user = unconfirmed_users.pop()  # pop删除尾部元素
    print("正在验证用户: " + current_user.title())
    confirmed_users.append(current_user)
print("\n以下用户验证通过:")
for confirmed_user in confirmed_users:
    print(confirmed_user.title())
"""
# 删除数组全部指定字符串
"""
# pets = ['dog', 'cat', 'dog', 'goldfish', 'cat', 'rabbit', 'cat']
# print(pets)
# while 'cat' in pets:
#     pets.remove('cat')
# print(pets)
"""
# 单词本插入
"""
words = []
dui = True
while dui:
    word = input("请输入新单词：")
    if word == "#":
        dui = False
        continue
    else:
        if word not in words:
            words.append(word)
print(words)
for word in words:
    print(word)
    answer = input("是否下一个（输入y/n）：")
    if answer == 'y':
        continue
    else:
        print(word)
answer = input("输入你需要查找的单词：")
if answer in words:
    print("在这里。")
"""
# 函数
"""
# 简化函数
'''
f = lambda x, y, z: x+y+z
f(1, 2, 3)
'''
''' 
# 问候语
def greet_user(username):
    # 显示简单的问候语
    print("hello," + username.title() + "!")
    print("nice to meet you.")
greet_user("HuangMou")
'''
# 批量问候
'''
def greet_users(names):
    # 向列表中的每位用户都发出简单的问候
    for name in names:
        msg = "Hello, " + name.title() + "!"
        print(msg)
userNames = ['zha', 'li', 'wang']
greet_users(userNames)
'''
# 动物叫
'''
def animal(name):
    if name == 'duck':
        print('GaGa')
    elif name == 'cow':
        print('MowMow')
animal('duck')
animal('cow')
'''
# 数学计算
'''
def f(x):
    y = x**2+3*x+5
    print(y)
x = int(input())
f(x)
'''
# 稍复杂数学计算
'''
def f(x):
    return x**2 + 3*x + 5
def f(x,y):
    return x//y,x%y
a,b = f(9,4)
print("9/4=",a,b)
def f(x,y,z):
    return x**3+y**3+z**3
print(f(4,5,6))
'''
# 可选择返回
'''
def pic(size,num1,num2,num3):
    if size == "green":
        return num1*10+num2*20+num3*10
    elif size == "red":
        return num1*15+num2*30+num3*15
print(pic("red",3,5,6))
'''
# 带默认参数
'''
def make_shirt(size = "大码",word = "I Love Python"):
    print(word + ", this is " + size)
make_shirt()
make_shirt("中码")
make_shirt(word = 'I love Java')
'''
# 带输入
'''
def read_command():
    # 接收命令
    line = input("-->")   # -->是命令提示符
    command = line.split()
    for i in range(0, len(command)):
        command[i] = command[i].strip()
    return command
print(read_command())
'''
# 简化交换值
'''
def exchange(x, y):
    return y, x
a = 3
b = 4
print('调用exchange(a, b)前，a=', a, ', b=', b)
a, b = exchange(a, b)   #完成a,b的交换
print("调用exchange(a, b)后，a=", a, ", b=", b)
'''
# 交换值
'''
def swap(x, y):
    t = x
    x = y
    y = t
a = 1
b = 2
print("调用swap(a, b)前，a=", a, ", b=", b)
swap(a, b)
print("调用swap(a, b)后，a=", a, ", b=", b)
'''
# 建立人信息+年龄参数
'''
def build_person(first_name, last_name, age=''):
    # 返回一个字典
    person = {'first': first_name, 'last': last_name}
    if age:  #age参数不为空
        person['age'] = age
    return person
musician = build_person('亮', '诸葛', age=27)
print(musician)
'''
# 二维字典
'''
def make_album(name = None,music = None,num = None):
    back = {"name":name,"music":music}
    if num:
        back["number"]=num
    return back
print(make_album("apple","kale"))
print(make_album("banana","genius"))
print(make_album("cat","paint",4))
'''
# 数组插入
'''
def insert(name, name_list):
    if name not in name_list:
        name_list.append(name)
name_list = []
insert('ye', name_list)
insert('he', name_list)
insert('ye', name_list)
insert('guo', name_list)
print(name_list)
name = "guo"
if name not in name_list[:]:
    name_list[:].append(name)
print(name_list)
'''
# 输出制作披萨语句
'''
def make_pizza(size, *toppings):
    print("\nMaking a " + str(size) +
          "-inch pizza with the following toppings:")
    for topping in toppings:
        print("- " + topping)
    print(toppings)

make_pizza(16, 'pepperoni')
make_pizza(12, 'mushrooms', 'green peppers', 'extra cheese')
'''
# 多参数
'''
def build_profile(first, last, **user_info):
    profile = {}
    profile['first_name'] = first
    profile['last_name'] = last
    for key, value in user_info.items():
        profile[key] = value
    print(user_info)
    return profile
user_profile = build_profile('albert', 'einstein', location='princeton',
                             field='physics')
print(user_profile)
'''
# 无限添加
'''
def add(**food):
    print("this is have:",end="")
    for key,value in food.items():
        print(" "+key+":"+str(value),end="")
    print(".that is all!")
add(gar=3,gen=4,gla=5)
add(salad = 5,solo = 8,kale = 1)
'''
# 车字典数据
'''
def make_car(human = None,size = None,**sear):
    car = {}
    car["name"]=human
    car["size"]=size
    for key,value in sear.items():
        car[key]=value
    return car
car = make_car("subaru",'outback',color = 'blue',tow_package=True)
print(car)
'''
# 递归函数
'''
def f(n):
    if n == 0:
        return 1
    elif n > 0:
        return n*f(n-1)
print(f(5))
'''
"""
# 函数引用
"""
import pizza
pizza.make_pizza(16, 'pepperoni')
pizza.make_pizza(12, 'mushrooms', 'green peppers', 'extra cheese')
from pizza import make_pizza
make_pizza(16, 'pepperoni')
make_pizza(12, 'mushrooms', 'green peppers', 'extra cheese')
from pizza import make_pizza  as mp
mp(16, 'pepperoni')
mp(12, 'mushrooms', 'green peppers', 'extra cheese')
import pizza as p
p.make_pizza(16, 'pepperoni')
p.make_pizza(12, 'mushrooms', 'green peppers', 'extra cheese')
from pizza import *
make_pizza(16, 'pepperoni')
make_pizza(12, 'mushrooms', 'green peppers', 'extra cheese')
"""
# 数组排序
"""
# grade = [(90,80,60),(70,87,65),(45,43,56)]
# print(sorted(grade, key=lambda g: g[0]+g[1]+g[2]))
# print(sorted(grade, key=lambda g: sum(g)))
# ge=grade[0]+grade[1]+grade[2]
# print(ge)
"""
# 二分图
"""
import turtle
def koch(size, n):
    print(size, n)
    if n == 0:
        turtle.fd(size)
    else:
        for angle in [0, 60, -120, 60]:
           turtle.left(angle)
           koch(size/3, n-1)
def main():
    turtle.setup(800,400)
    turtle.speed(0)  #控制绘制速度
    turtle.penup()
    turtle.goto(-300, -50)
    turtle.pendown()
    turtle.pensize(2)
    koch(600,2)
    turtle.hideturtle()
    input()
main()
"""
# 类
"""
# 狗描述
'''
class Dog():
    def __init__(self,name,age):
        # 初始化属性name和age
        self.name = name
        self.age = age
    def sit(self):
        print(self.name.title()+"is now sitting.")
    def roll_over(self,n):
        print(self.name.title() + "rolled over!")
        for i in range(1,n+1):
            print("#" + str(i))
my_dog = Dog("willie",6)
my_dog.sit()
my_dog.roll_over(3)
'''
# 车描述
'''
class Car():
    def __init__(self,make,model,year):
        self.make = make
        self.model = model
        self.year = year
    def get_descriptive_name(self):
        long_name = str(self.year) + " " + self.make + " " + self.model
        return long_name
    def long_descriptive(self,n):
        long = "this car run more than "+str(n)
        return long
    def read_odometer(self):
        return self.odometer
    def set_odometer(self,meter):
        self.odometer = meter
    def add_odometer(self,new):
        if new > 0:
            self.odometer += new
my_new_car = Car("audi","a4",2016)
print(my_new_car.get_descriptive_name())
print(my_new_car.long_descriptive(100))
my_new_car.set_odometer(5000)
print("已行驶的里程：",my_new_car.read_odometer())
'''
# 电动车描述
'''
class ElectricCar(Car):
    # 电动车的独特之处
    def __init__(self,make,model,year):
        # 初始化父类的属性
        super().__init__(make,model,year)
        self.battery_size = 70
    def get_nattily_size(self):
        self.battery_size = 70
        return self.battery_size
my_tesla = ElectricCar("tesla","model",2016)
print(my_tesla.get_descriptive_name())
print("电池容量: ",my_tesla.get_nattily_size())
'''
# 动物描述
'''
class Animal():
    def crow(self):
        pass
class Dog(Animal):
    def crow(self):
        print("WangWang")
class Cat(Animal):
    def crow(self):
        print("MaoMao")
dog = Dog()
dog.crow()
cat = Cat()
cat.crow()
'''
# 用户操作
'''
class User():
    def login(self):
        self.name = input("请输入你的用户名:")
        self.over = input("请输入密码:")
    def showPrivilege(self,odd):
        if odd == "say":
            say = input("请输入你想说的话:")
            print(self.name + ":" + say)
        elif odd == "in":
            print("欢迎来到这里.")
            print("希望你会开心.")
            print("虽然这并没有什么用.")
class Get_user(User):
    def level(self,level):
        print("your level is :" + str(level))
    def showPrivilege(self,odd):
        super().showPrivilege(odd)
use1 = Get_user()
use1.login()
say = input("请输入特权:")
use1.showPrivilege(say)
use1.level(5)
'''
"""
# 文件操作
"""
# 读取每行最后一个单词
'''
with open(r"D:\\ZuoYe\\python1\\pi_digits.txt") as datafile:
    lines = datafile.read()
    print(lines)
    print("line[-1]:",lines[-1])
'''
# 逐行读取
'''
filename = 'pi_digits.txt'
with open(filename) as datafile:
    for line in datafile:  #逐行读取
        print(line.rstrip())
    lines = datafile.readlines()
    print(lines)
'''
# 读行去两边空格
'''
filename = "pi_digits.txt"
with open(filename) as file_object:
    lines = file_object.readlines()
pi_string = ""
for line in lines:
    pi_string +=line.strip()
print(pi_string)
print(len(pi_string))
'''
# 写文件
'''
# filename = "hello.txt"
# with open(filename,"w") as wfile:
#     wfile.write("I love python too.\n")
#     wfile.write("Wow,Python is a kind of snake\n")
#     wfile.write("you are so gar".title())
# print("that is ok".title())
# with open(filename,"a") as afile:
#     afile.write("\n死亡如风,常伴吾身.\n")
#     afile.write("我于杀戮之中绽放,亦如黎明中的花朵.\n")
# print("你的文件已经写完,我的主人.")
"""
# 异常处理
"""
# 除法异常
'''
a = int(input("你输入的第一个数字:"))
b = int(input("你输入的第二个数字:"))
try:
    r = a/b
except ZeroDivisionError:
    print("You can't davidde by 0!")
else:
    print(r)
'''
# 文件存在异常
'''
# filename = 'alice.txt'
# try:
#     with open(filename, encoding='utf-8') as f_obj:
#         contents = f_obj.read()
# except FileNotFoundError as e:
#     msg = "Sorry, the file " + filename + " does not exist."
#     print(msg)
# else:
#     # Count the approximate number of words in the file.
#     words = contents.split()
#     num_words = len(words)
#     print(words)
#     print("The file " + filename + " has about " + str(num_words) + " words.")
'''
"""
# 数据库处理
"""
!/usr/bin/env python
coding: utf-8
import pymysql
conn = pymysql.connect(
    host="10.188.2.14",  # 数据库ip地址，如果是本机填127.0.0.1
    port=4406,  # 数据库访问端口，默认4406
    user="root",  # 用户名
    password="xxx",  # 口令
    database="LPOJ",  # 访问的数据库名
    charset='utf8',  # 数据库编码
    cursorClass=pymysql.cursors.DictCursor)  # 创建什么样的游标访问数据。字典游标表示每条记录映射为字典

cursor = conn.cursor()  # 获取游标
sql = "select username, ip from user_userlogindata where logintime between '2019-11-21 14:00' and '2019-11-21 17:30'"
cursor.execute(sql)  # 执行sql语句
for rec in cursor:  # 通过游标依次访问返回的每条记录，记录格式为字典
    print(rec)
"""
# 折线图至html
"""
from pyecharts.charts import Line
from pyecharts import options as opts
v1 = [5, 20, 36, 10, 75, 90]
v2 = [10, 25, 8, 60, 20, 80]
x = ["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"]
line = (
    Line(opts.InitOpts(width = '500px',height = '300px'))
    .add_xaxis(x)
    .add_yaxis("商家A", v1,is_selected = True,is_smooth=False)
    .add_yaxis("商家B", v2,is_selected = True,is_step=False,
               is_smooth=True)
    )
line.render("a.html")
"""
# 打印每天信息
"""
import json
import matplotlib.pyplot as plt
from datetime import datetime
# 将数据加载到一个列表中
filename = 'ecdict.txt'
with open(filename) as f:
    ecdict = json.load(f)
# 打印每一天的信息
dates = []
prices = []
for btc_dict in ecdict:
        current_date = datetime.strptime(btc_dict["date"], "%Y-%m-%d")
        dates.append(current_date)
        prices.append(float(btc_dict["close"]))

fig = plt.figure(dpi=128, figsize=(10, 6))
fig.autofmt_xdate(rotation=60)
plt.plot(dates, prices)
plt.show()
"""
# pyecharts画热力图
"""
from pyecharts.charts import Geo
from pyecharts.globals import ChartType
from pyecharts import options as opts
keys = ['上海', '北京市', '合肥市', '哈尔滨市', '广州市', '成都市',
        '无锡市', '杭州市', '武汉市', '深圳市', '西安市', '郑州市',
        '重庆市', '长沙市', '贵阳市', '乌鲁木齐市']
values = [4.07, 1.85, 4.38, 2.21, 3.53, 4.37, 1.38, 4.29, 4.1,
          1.31, 3.92, 4.47, 2.40, 3.60, 1.2, 3.7]
geo = Geo()
geo.add_schema(maptype="china")
geo.set_global_opts(
    visualmap_opts=opts.VisualMapOpts(is_piecewise=False,
                                      max_=10, min_=0),
    title_opts=opts.TitleOpts(title="空气质量热力图"),
    toolbox_opts=opts.ToolboxOpts(is_show=False),
)
geo.add("空气质量", [list(z) for z in zip(keys, values)],
        type_=ChartType.EFFECT_SCATTER)
geo.set_series_opts(label_opts=opts.LabelOpts(is_show=True))
geo.render("d.html")
"""
# pyecharts画地图
"""
import csv
from pyecharts.charts import Map
from pyecharts import options as opts
with open(r'人口普查.csv', newline='') as f:
    reader = csv.reader(f)

    keys = []
    values = []
    next(f)
    for line in reader:
        keys.append(line[0].replace(" ", ""))
        values.append(line[4])
print(keys)
map= (
        Map()
        .add("人口数量", [list(z) for z in zip(keys,values)], "china")
        .set_global_opts(
            title_opts=opts.TitleOpts(title="中国人口数量"),
            visualmap_opts=opts.VisualMapOpts(max_=10000000),
        )
    )
map.render("e.html")
"""
# 使用pandas读取excel文件
"""
import csv
from pyecharts.charts import Map
from pyecharts import options as opts
import pandas as pd
df = pd.read_excel(r'人口普查.xls', skiprows=8, header=None, usecols='A, E', names=["city","num"])
keys = [ x.replace(" ", "") for x in df['city'] ]
map= (
        Map()
        .add("人口数量", [list(z) for z in zip(keys, df['num'])], "china")
        .set_global_opts(
            title_opts=opts.TitleOpts(title="中国人口数量"),
            visualmap_opts=opts.VisualMapOpts(max_=10000000),
        )
    )
map.render_notebook()
"""
# pyecharts画词云
"""
from pyecharts.charts import WordCloud
name =['Sam S Club', 'Macys', 'Amy Schumer', 'Jurassic World',
       'Charter Communications', 'Chick Fil A', 'Planet Fitness',
       'Pitch Perfect', 'Express', 'Home', 'Johnny Depp',
       'Lena Dunham', 'Lewis Hamilton', 'KXAN', 'Mary Ellen Mark',
       'Farrah Abraham', 'Rita Ora', 'Serena Williams',
       'NCAA baseball tournament', 'Point Break']
value =[10000, 6181, 4386, 4055, 2467, 2244, 1898, 1484, 1112,
        965, 847, 582, 555, 550, 462, 366, 360, 282, 273, 265]
wordcloud = WordCloud()
wordcloud.add("", [ list(z) for z in zip(name, value) ])
wordcloud.render("c.html")
"""
# pyecharts分析《围城》词云
"""
from pyecharts.charts import WordCloud
import jieba
stop_words = []
with open("中文停用词表.txt", encoding='utf8') as f:
       for line in f:
              stop_words.append(line.strip())
with open("围城.txt", encoding='utf8') as f:
       lines = f.readlines()
word_freq = {}
count = 0
for line in lines:
       words = jieba.cut(line.strip(), cut_all=False)
       # print([w for w in words])
       for word in words:
              if word in stop_words: continue
              if word in word_freq:
                     word_freq[word] = word_freq[word] + 1
              else:
                     word_freq[word] = 1
word_freq = sorted(word_freq.items(), key=lambda x: x[1],
                   reverse=True)
wordcloud = WordCloud()
wordcloud.add("", [list(z) for z in word_freq[:100]])
wordcloud.render("a.html")
"""
# 螺旋输出顺序数字
"""
n = int(input())
b = [[0, 1], [1, 0], [0, -1], [-1, 0]]
h = 0
x = 0
y = 0
put = [[0 for i in range(n)] for j in range(n)]
for i in range(1, n * n + 1):
    put[x][y] = i
    x += b[h][0]  # 坐标轴y的加减
    y += b[h][1]  # 坐标轴x的加减
    if x > n-1 or y > n-1 or x < 0 or y < 0 or put[x][y] != 0:
        x -= b[h][0]
        y -= b[h][1]
        h += 1
        h %= 4
        x += b[h][0]
        y += b[h][1]
for k in range(0, n):
    for l in range(0, n-1):
        print("%3d" % put[k][l], end=" ")
    print("%2d" % put[k][n-1])
"""