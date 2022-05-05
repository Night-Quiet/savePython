import json
import operator
import os
import copy
import csv
import sqlite3
from pathlib import Path
import re
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize.casual import casual_tokenize
from nltk.stem.porter import PorterStemmer
import nltk
import jieba
import jieba.posseg as psg
import pandas as pd
from collections import Counter
import stopwords
import math
import naive_stopwords as sw
import sklearn.feature_extraction.text as sfe
import numpy as np
import time
from textrank4zh import TextRank4Keyword as textKeyWord

"""
文件读写
"""


def json_read(path):
    """
    :param path: json文件地址(相对|绝对)  此处相对地址
    :return: json对应数据集(maybe list dict)  此处得到list
    :describe: 读取并返回json内容
    """
    f_read = open(path, 'r', encoding='utf-8')
    ts = f_read.read()
    list_data = json.loads(ts)
    f_read.close()

    # print("...............")
    # print('记录数为', str(len(dic)))
    # print("...............")
    # print('')
    # print(dic[0])
    return list_data


def json_save(path, dict_data):
    """
    :param path: 储存地址
    :param dict_data: 储存内容
    :return: None
    :describe: 储存数据
    """
    f_write = open(path, 'w', encoding='utf-8')
    json.dump(dict_data, f_write)
    f_write.close()


def txt_save(path, data, describe):
    """
    :param path: file address
    :param data: list, dict - save data
    :param describe: 描述内容
    :return: None
    :describe: 将数据储存为txt, 一般循环解读储存
    """
    with open(path, 'a') as file_object:
        file_object.write(describe + ": \n")
        if isinstance(data, list):
            for dirP in data:
                file_object.write(dirP + "\n")
            file_object.write("----------------------------\n")
        elif isinstance(data, dict):
            for key, value in data.items():
                file_object.write(str(key) + ":  " + str(value) + "\n")
            file_object.write("----------------------------\n")
    file_object.close()


def DataFrame_save(path, data):
    """
    :param path: file address
    :param data: save data
    :return: None
    :describe: 专门对DataFrame数据储存
    """
    data.to_csv(path, sep='\t', index=False)


def sql_save(path, list_data, table_name):
    """
    :param path: file address
    :param list_data: list - 数据元胞列表
    :param table_name: list - 建立表的列名
    :return: None
    :describe: 对数据储存入数据库
    """
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute('drop table if exists paper')
    tableName = "CREATE TABLE paper("
    for i in range(len(table_name)):
        if i == len(table_name) - 1:
            tableName += table_name[i] + " text)"
        elif i == 0:
            tableName += table_name[i] + " int primary key,"
        else:
            tableName += table_name[i] + " text,"
    c.execute(tableName)
    question_num = "INSERT INTO paper VALUES ("
    for i in range(len(table_name)):
        if i == len(table_name) - 1:
            question_num += "?)"
        else:
            question_num += "?, "
    c.executemany(question_num, list_data)

    conn.commit()
    conn.close()


def txt_read(path):
    """
    :param path: str - 文件路径
    :return: list_read: list - 读取后数据 [[1, 2], ... ]
    """
    if os.path.exists(path):
        list_read = list()
        with open(path, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                line = line.strip(" ")
                if len(line) != 0:
                    line = line.split(",", 1)
                    list_read.append(line)
        return list_read


"""
str - 字符串处理
"""


def filterHtmlTag(htmlStr):
    """
    :param htmlStr: str - 带html标签的字符串
    :return: str - 去除html标签的字符串
    :describe: 过滤html中的标签
    """
    # 兼容换行
    s = htmlStr.replace('\r', '\n')

    # 规则
    re_script = re.compile(r'<\s*/?script[^>]*>', re.I)  # script
    re_style = re.compile(r'<\s*/?style[^>]*>', re.I)  # style
    re_br = re.compile(r'<\s*br\s*/?\s*>', re.I)  # br标签换行
    re_p = re.compile(r'<\s*[/／]?p[^>]*>', re.I)  # p标签换行
    re_h = re.compile(r'<\s*[\!|/|／]?\w+[^>]*>', re.I)  # other HTML标签
    re_comment = re.compile(r'<!--[^>]*-->')  # HTML注释
    re_hendstr = re.compile(r'^\s*|\s*$')  # 头尾空白字符
    re_lineblank = re.compile(r'[\t\f\n\v]*')  # 空白字符
    re_sql = re.compile(r'<sql>\w*</sql>', re.I)  # sql奇怪符号
    re_error = re.compile(r'<\s*[\!|/|／]?\w+[^>]*>?', re.I)  # 错误的html字符

    # 处理
    # 转义字符
    while re.search('&amp', s) is not None:
        s = re.sub('&amp;', '&', s)
    s = re.sub('&quot;', '"', s)
    s = re.sub('&lt;', '<', s)
    s = re.sub('&gt;', '>', s)
    s = re.sub('&nbsp;', ' ', s)
    s = re.sub('&#39', "'", s)

    s = re.sub(re_sql, ' ', s)  # sql去除
    s = re.sub(re_script, '', s)  # 去script
    s = re.sub(re_style, '', s)  # 去style
    s = re.sub(re_br, '', s)  # br标签换行
    s = re.sub(re_p, '', s)  # p标签换行
    s = re.sub(re_h, '', s)  # 去HTML标签
    s = re.sub(re_comment, '', s)  # 去HTML注释
    s = re.sub(re_lineblank, '', s)  # 去空白字符
    s = re.sub(re_hendstr, '', s)  # 去头尾空白字符
    s = re.sub(re_error, '', s)  # 去错误html标签
    return s


def cut_vocabulary(content, way):
    """
    :param content: str - 分词内容
    :param way: str - 分词方式
    :way - example: re - re分词;
    :return: list - 分词完成内容
    :describe: 对文本内容分词, 得到分词结果
    """
    # re分词 正常操作
    if way == "re":
        re_split = re.compile(r'[-\s.,;!?]+', re.I)
        tokens = re.split(re_split, content)
        res = [x for x in tokens if x and x not in '- \t\n.,;!?']  # 将空白符和标点符号过滤掉
        return res
    # nltk分词
    elif way == "nltk":
        tokenizer = RegexpTokenizer(r'\w+|$[0-9.]+|\S+')
        token = tokenizer.tokenize(content)
        return token
    # nltk分词-解决n't问题
    elif way == "nltk-nt":
        tokenizer = TreebankWordTokenizer()
        token = tokenizer.tokenize(content)
        return token
    # nltk分词-去重
    elif way == "nltk-dw":
        res = casual_tokenize(content)
        # 去重
        res = casual_tokenize(content, reduce_len=True, strip_handles=True)
        return res
    # jieba分词-全模式
    elif way == "jieba":
        wordlist1 = jieba.cut(content, cut_all=True)
        wd1 = "|".join(wordlist1)
        return wd1
    # jieba分词-精确模式
    elif way == "jieba_et":
        wordlist2 = jieba.cut(content)  # cut_all=False
        wd2 = "|".join(wordlist2)
        return wd2
    # 搜索引擎模式
    elif way == "jieba_sc":
        wordlist3 = jieba.cut_for_search(content)
        wd3 = "|".join(wordlist3)
        return wd3


def str_deal(content):
    """
    :param content: str
    :return: str
    :describe: 去停用词, 词干还原
    """

    def cut_stopNote(content_in):
        """
        :param content_in: str - 欲被去停用词字符串内容
        :return: str - 去停用词后的字符串内容
        :describe: 去停用词
        """
        cachedStopWords = stopwords.get_stopwords("english")
        text_in = ' '.join([word for word in content_in.split() if word not in cachedStopWords])
        return text_in

    def stem_reduction(content_in):
        """
        :param content_in: str - 欲还原内容
        :return: str - 还原后内容
        :describe: 词干还原
        """
        stm = PorterStemmer()
        res_in = ' '.join([stm.stem(w).strip("'") for w in content_in.split()])
        return res_in

    text = cut_stopNote(content)
    text = stem_reduction(text)
    return text


def note_voc(content_cut):
    """
    :param content_cut: list - 已分词列表
    :return: dict - 带标签的分词字典
    :describe: 对分词进行标签化
    """
    word_tag = nltk.pos_tag(content_cut)
    return word_tag


def csv_one_hot(path):
    """
    :param path: str - file address
    :return: DataFrame - one-hot矩阵
    :describe: 对字典数据进行one-hot操作
    """
    num = 0
    deal = dict()
    with open(str(path)) as f:
        reader = csv.reader(f)
        for line in reader:
            if num == 0:
                num += 1
                continue
            wordList = dict(Counter(filterHtmlTag(line[4])))
            deal['sent{}'.format(num)] = wordList
            num += 1
    print(deal)
    d_f = pd.DataFrame(deal, dtype=int).fillna(0).astype(int).T
    print(d_f)
    return d_f


def csv_counter(path):
    """
        :param path: str - file address
        :return: deal: dict - 简单的分词字典
        :describe: 对字典数据进行one-hot操作
        """
    num = 0
    deal = dict()
    with open(str(path)) as f:
        reader = csv.reader(f)
        for line in reader:
            if num == 0:
                num += 1
                continue
            wordList = dict(Counter(cut_vocabulary(filterHtmlTag(line[4]), "jieba").split("|")))
            deal['sent{}'.format(num)] = wordList
            num += 1
    return deal


def csv_tf(path):
    """
    :param path: str - file address
    :return: dict - 词频字典
    :describe: 对csv某列内容整合成字符串, 并计算词频
    """
    num = 0
    deal_tf = dict()
    tf_str = ""
    with open(str(path)) as f:
        reader = csv.reader(f)
        for line in reader:
            if num == 0:
                num += 1
                continue
            tf_str += filterHtmlTag(line[4])
            num += 1
    deal_tf_temp = dict(Counter(tf_str))
    tf_length = len(deal_tf_temp)
    for key, value in deal_tf_temp.items():
        deal_tf[key] = value / tf_length
    return deal_tf


def corpus_built(data):
    """
    :param data: list - 数据列表
    :return: ids: 数据id; corpus: 预处理后数据
    """
    # 选择
    jieba_save = ['n', 'nz', 'v', 'vd', 'vn', 'l', 'a', 'd']
    # 停用词
    stop_class = sw.Stopwords()
    stop_class.add(' ')
    # corpus创建
    corpus = []
    ids = []
    for val1, val2 in data:
        temp = psg.cut(val2)
        temp_str = ""
        for val in temp:
            if stop_class.contains(val.word) is not True and val.flag in jieba_save:
                temp_str += val.word + " "
        if len(temp_str) > 0:
            ids.append(val1)
            temp_str = temp_str[:-1]
            corpus.append(temp_str)
    return ids, corpus


def corpus_built_list(data):
    """
        :param data: list - 数据列表
        :return: ids: 数据id; corpus: 预处理后数据
        """
    # 选择
    jieba_save = ['n', 'nz', 'v', 'vd', 'vn', 'l', 'a', 'd']
    # 停用词
    stop_class = sw.Stopwords()
    stop_class.add(' ')
    # corpus创建
    corpus = []
    ids = []
    for val1, val2 in data:
        temp = psg.cut(val2)
        temp_str = []
        for val in temp:
            if stop_class.contains(val.word) is not True and val.flag in jieba_save:
                temp_str.append(val.word)
        if len(temp_str) > 0:
            ids.append(val1)
            temp_str = temp_str
            corpus.append(temp_str)
    return ids, corpus


"""
文件夹统计
"""


def folder_struct(path):
    """
    :param path: file address
    :return: list1, list2
    :describe: 将文件夹的结构以列表形式返还
    """
    file_list = []
    dir_list = []

    for root, dirs, files in os.walk(path):
        if files:
            for file in files:
                file_list.append(str(os.path.join(root, str(file))))
        if dirs:
            for dir in dirs:
                dir_list.append(str(os.path.join(root, str(dir))))
    return file_list, dir_list


"""
题目专门函数, 不可复用
"""


def data_paper_part(list_data):
    """
    :param list_data: list
    :return: dict
    :template: {作者: [论文数量], [论文原始序号], [论文ID], [论文标题], [论文所属期刊], [论文署名次序], {期刊:发表数}}
    :describe: 对数据处理, 得到以author为keys的相关内容字典, 以论文数排序
    """
    # 获取各类数据
    id = {}
    title = {}
    author = {}
    source = {}
    num = 1
    for item in list_data:
        if item.get('author') is not None:
            id[num] = item.get('id')
            title[num] = item.get('title')
            author[num] = item.get('author')
            source[num] = item.get('source')
            num = num + 1

    authors = {}

    # 巨大数据字典: key为作者名; value为对应数据列表
    for i in author.keys():
        num_sign = 1
        for authorItem in author[i]:
            authorItem = re.split('[&;]', authorItem)  # 检测出部分作者名不规范, 存在两个名字用[& ;]等符号连接, 故在此进行分割
            for authorItem_split in authorItem:
                if authorItem_split != '':  # 分割异常产生空名, 略去
                    if authorItem_split not in authors.keys():
                        authors[authorItem_split] = [[], [], [], [], [], [], {}]
                        authors[authorItem_split][0].append(1)  # 论文数量
                        authors[authorItem_split][1].append(i)  # 论文原始序号
                        authors[authorItem_split][2].append(id[i])  # 论文ID
                        authors[authorItem_split][3].append(title[i])  # 论文标题
                        authors[authorItem_split][4].append(source[i])  # 论文所属期刊
                        authors[authorItem_split][5].append(num_sign)  # 论文署名次序
                        if source[i] in authors[authorItem_split][6].keys():  # 期刊对应发表数
                            authors[authorItem_split][6][source[i]] += 1
                        else:
                            authors[authorItem_split][6][source[i]] = 1
                    else:
                        authors[authorItem_split][0][0] += 1
                        authors[authorItem_split][1].append(i)
                        authors[authorItem_split][2].append(id[i])
                        authors[authorItem_split][3].append(title[i])
                        authors[authorItem_split][4].append(source[i])
                        authors[authorItem_split][5].append(num_sign)
                        if source[i] in authors[authorItem_split][6].keys():
                            authors[authorItem_split][6][source[i]] += 1
                        else:
                            authors[authorItem_split][6][source[i]] = 1
                    num_sign += 1

    # 按照作者发布论文总数量进行排序, 原理未知, 异常未知
    authorsSort = dict(sorted(authors.items(), key=operator.itemgetter(1)))
    return authorsSort


def simple_output(dict_data):
    """
    :param dict_data: dict
    :return: None
    :describe: 简单输出前10个数据
    """
    judge = 1
    for authorsSortItem, paper_sign in dict_data.items():
        if judge > 10:
            break
        print("作者: " + authorsSortItem)
        for paper_num in range(paper_sign[0][0]):
            print("论文原始序号: " + str(paper_sign[1][paper_num]))
            print("论文ID: " + paper_sign[2][paper_num])
            print("论文标题: " + paper_sign[3][paper_num])
            print("作论文所属期刊: " + paper_sign[4][paper_num])
            print("论文署名次序: " + str(paper_sign[5][paper_num]))
            print("**********************************")
        print("论文数量: " + str(paper_sign[0][0]))
        for journal, jouNum in paper_sign[6].items():
            print("作者发布期刊: " + journal)
            print("对应期刊论文发布数量: " + str(jouNum))
        print("----------------------------------")
        print("----------------------------------")
        judge += 1


def data_paper_all(list_data):
    """
    :param list_data: list - 需处理数据列表
    :return: list - 元胞数组列表
    :describe: 对数据列表处理, 生成返回元胞数组列表
    """
    author = {}
    source = {}
    time = {}
    title = {}
    collaborators = {}
    abstract = {}
    num = 1

    sql_list = []

    for item in list_data:
        if item.get('author') is not None:
            author[num] = item.get('author')[0]
            source[num] = item.get('source')
            time[num] = item.get('year')
            title[num] = item.get('title')
            # 简单的获取合作者,并转化成字符串,而不是列表
            if len(item.get('author')) > 1:
                collaborators[num] = ""
                for leng in range(1, len(item.get('author'))):
                    collaborators[num] += "&" + item.get('author')[leng]
            else:
                collaborators[num] = ""
            abstract[num] = item.get('abstract')
            # 建立元胞组
            sql_list.append((num, author[num], source[num], time[num], title[num],
                             collaborators[num], abstract[num]))
            num = num + 1
    return sql_list


def data_html(path):
    """
    :param path: file address
    :return: list - 数据库元胞数组
    :describe: 对csv文件处理, 生成返回元胞数组列表
    """
    p = Path(path)
    num = 0
    sql_list = []
    with open(str(p)) as f:
        reader = csv.reader(f)
        for line in reader:
            if num == 0:
                num += 1
                continue
            sql_list.append((num, line[0], line[1], line[2], filterHtmlTag(line[3]), filterHtmlTag(line[4])))
            num += 1
    return sql_list


def is_all_chinese(content):
    """
    :param content: str - 判断内容
    :return: True | False
    :describe: 判断内容是否全中文
    """
    for _char in content:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True


def df_idf(d_f, path_save):
    """
    :param d_f: dict - 分词字典
    :param path_save: file address
    :return: None
    """
    IDF_ci = {}
    TF_ci = {}
    for value in d_f.values():
        for key, val in value.items():
            IDF_ci.setdefault(key, []).append(1)
            TF_ci.setdefault(key, []).append(val)
    IDF_ci = {key: len(value) for key, value in IDF_ci.items()}
    TF_ci = {key: sum(value) for key, value in TF_ci.items()}

    IDF = {key: math.log((1 + len(list(d_f.keys()))) / value) for key, value in IDF_ci.items()}
    TF = {key: value / sum(list(TF_ci.values())) for key, value in TF_ci.items()}
    TF_IDF = {key: IDF[key] * TF[key] for key in IDF.keys()}
    TF_IDF_list = [[key, value] for key, value in TF_IDF.items()]
    result = pd.DataFrame(TF_IDF_list)
    DataFrame_save(path_save, result)


def tf_idf(ids, corpus):
    """
    :param ids: list - 文档id
    :param corpus: list - 已预处理字符串
    :return: dict - {id: top10 tf-idf 词汇}
    """
    cv = sfe.CountVectorizer()
    tt = sfe.TfidfTransformer()
    # 词频矩阵
    X = cv.fit_transform(corpus)
    x_word = X.toarray()
    # 每个词tf-idf权值
    tfIdf = tt.fit_transform(X)
    # 关键词
    voc_word = cv.get_feature_names()
    # tf-idf矩阵
    weight_word = tfIdf.toarray()
    # 打印权重
    top10 = []
    for i in range(len(x_word)):
        df_word = []
        df_weight = []
        for j in range(len(voc_word)):
            df_word.append(voc_word[j])
            df_weight.append(weight_word[i][j])
        result = pd.DataFrame({"word": df_word, "weight": df_weight})
        result_sort = result.sort_values(by="weight", ascending=False)
        result_sort = np.array(result_sort['word'])
        top10.append(" ".join(result_sort[0:10]))
    end_result = pd.DataFrame({"id": ids, "top": top10})

    return end_result


def textRank_my(data):
    dict_tr = dict()
    score_begin_temp = []
    for val in data:
        score_begin_temp += val
        len_val = len(val) - 1
        for i in range(len_val):
            dict_tr.setdefault(val[i], []).append(val[i + 1])
            dict_tr.setdefault(val[i + 1], []).append(val[i])
    for key, value in dict_tr.items():
        dict_tr[key] = list(set(value))
    score = {key: 1 for key in dict_tr}
    score_one = {key: 1 for key in dict_tr}
    judge = True
    while judge:
        judge = False
        for key, value in dict_tr.items():
            score_one[key] = 0.15
            for k in value:
                score_one[key] += 0.85 * score_one[k] * 1 / len(dict_tr[k])
            if abs(score_one[key] - score[key]) > 0.0001:
                judge = True
        score = copy.deepcopy(score_one)
    result_all = sorted(score.items(), key=lambda x: x[1], reverse=True)
    result_six = [val[0] for val in result_all[:10]]
    return result_six


def keywords_extraction(data):
    text = ""
    for val in data:
        for val2 in val:
            text += val2
        text += "。"
    tr4w = textKeyWord(allow_speech_tags=['n', 'nz', 'v', 'vd', 'vn', 'l', 'a', 'd'])
    tr4w.analyze(text=text, window=2, lower=True, vertex_source='all_filters', edge_source='no_stop_words',
                 pagerank_config={'alpha': 0.85, })
    keywords = tr4w.get_keywords(num=10, word_min_len=1)
    result = [val["word"] for val in keywords]
    return result


def main():
    # 第一次作业第一题
    """
    path_read = r'..\数据\自然语言处理\query.json'
    path_save = r'..\数据\自然语言处理\out.json'
    json_data = json_read(path_read)
    deal_data = data_paper_part(json_data)
    simple_output(deal_data)
    json_save(path_save, deal_data)
    """
    # 第一次作业第二题
    """
    path_read = r"..\数据\自然语言处理\古汉语-demo"
    path_save = r"..\数据\自然语言处理\output.txt"
    file_list, dir_list = folder_struct(path_read)
    txt_save(path_save, file_list, "目录")
    txt_save(path_save, file_list, "文件")
    """
    # 第一次作业第三题
    """
    path_read = r"..\数据\自然语言处理\query.json"
    path_save = r"..\数据\自然语言处理\test.db"
    table_name = ["id", "author", "source", "time", "title", "collaborators", "abstract"]
    json_data = json_read(path_read)
    sql_list = data_paper_all(json_data)
    sql_save(path_save, sql_list, table_name)
    """
    # 第二次作业第一题
    """
    path_read = r"..\数据\自然语言处理\计算机基础数据.csv"
    path_save = r"..\数据\自然语言处理\html.db"
    table_name = ["id", "课程ID", "课程名", "标题", "提问", "回答"]
    sql_list = data_html(path_read)
    sql_save(path_save, sql_list, table_name)
    """
    # 第二次作业第二题
    """
    content = "Glimpse is an indexing and query system that allows for " \
              "search through a file system or document collection quickly. Glimpse " \
              "is the default search engine in a larger information retrieval system. " \
              "It has also been used as part of some web based search engines."
    stem = str_deal(content)
    cut = cut_vocabulary(str(stem), "nltk")
    voc = note_voc(cut)
    print(voc)
    """
    # 第三次作业第一题
    # """
    path_read = r"..\..\数据\自然语言处理\计算机基础数据.csv"
    path_save_df = r"..\..\数据\自然语言处理\one_hot.txt"
    path_save_txt = r"..\..\数据\自然语言处理\TF.txt"
    d_f = csv_one_hot(path_read)
    deal_tf = csv_tf(path_read)
    DataFrame_save(path_save_df, d_f)
    txt_save(path_save_txt, deal_tf, "TF词频")
    # """
    # 第四次作业第一题
    """
    path_read = r"..\数据\自然语言处理\计算机基础数据.csv"
    path_save = r"..\数据\自然语言处理\TF-IDF.txt"
    d_f = csv_counter(path_read)
    df_idf(d_f, path_save)
    """
    # 第五次作业第一题
    """
    # 文件读取
    path = r"../../数据/自然语言处理/SET2020.txt"
    txtRead = txt_read(path)
    txtRead.pop(0)

    # corpus创建
    ids, corpus = corpus_built(txtRead)
    # 计算tf-idf
    end_result = tf_idf(ids, corpus)
    # print(end_result)
    # 保存到csv
    end_result.to_csv("keys_TFIDF.csv", encoding="utf_8_sig", index=False)
    """
    # 第六次作业第一题
    """
    path = r"../../数据/自然语言处理/SET2020.txt"
    txtRead = txt_read(path)
    txtRead.pop(0)
    ids, corpus = corpus_built_list(txtRead)
    time_begin1 = time.time()
    my_textRead_result = textRank_my(corpus)
    time_end1 = time.time()
    time_begin2 = time.time()
    textRead_result = keywords_extraction(corpus)
    time_end2 = time.time()
    time_begin3 = time.time()
    text = ""
    for val in corpus:
        for val2 in val:
            text += val2 + " "
    end_result = tf_idf(["1"], [text])
    tf_idf_result = end_result["top"][0].split(" ")
    time_end3 = time.time()
    print("自写textRank代码---------------------")
    print(my_textRead_result)
    print("第三方textRank代码-------------------")
    print(textRead_result)
    print("自写TF-IDF代码-----------------------")
    print(tf_idf_result)
    print("\na: 由结果知, TEXTRANK算法和TFIDF结果并不完全相同, 但某些词的提取是相同的, 可能基于原理差异")
    print("b: 根据算法执行时间, 知: ")
    print("     tf-idf算法代码最快: 耗时: {0:.3f}".format(time_end3 - time_begin3))
    print("     自写textRank算法代码中等: 耗时: {0:.3f}".format(time_end1-time_begin1))
    print("     第三方库textRank算法代码最慢: 耗时: {0:.3f}".format(time_end2-time_begin2))
    """


if __name__ == "__main__":
    main()
