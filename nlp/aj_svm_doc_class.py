# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import jieba
import jieba.analyse
import jieba.posseg as pseg
import os
#jieba.load_userdict('wordDict.txt')
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib

rate = 0.8 #培训集比例

file_path = "D:\\DATA_SET\\SJAL\\AJ_INFO.csv"
file_dict = "D:\\DATA_SET\\SJAL\\BJLXDMB.csv"

alarm_code = pd.read_csv(file_dict, dtype=str).drop(['MS','XS','PX'], axis=1)
data = pd.read_csv(file_path, dtype=str)
data = data.dropna(axis=0)
bjlxdm = data['BJLXDM'].value_counts()[:25]
#data.loc[~(data['BJLXDM'].isin(bjlxdm.index))]
data.loc[~(data['BJLXDM'].isin(bjlxdm.index)),'BJLXMC']= '其他'
data.loc[~(data['BJLXDM'].isin(bjlxdm.index)),'BJLXDM']= '000000'
    


# 读取训练集
def readtrain():
    
    content_train = data['NR'].tolist()  #第一列为文本内容，并去除列名
    lable_train = data['BJLXDM'].tolist() #第二列为类别，并去除列名
    print ('训练集有 {} 条句子'.format(len(content_train)))
    train = [content_train, lable_train]
    return train


# 将utf8的列表转换成unicode
def changeListCode(b):
    a = []
    for i in b:
        a.append(i.decode('utf8'))
    return a


# 对列表进行分词并用空格连接
def segmentWord(cont):
    c = []
    for i in cont:
        a = list(jieba.cut(i))
        b = " ".join(a)
        c.append(b)
    return c


# corpus = ["我 来到 北京 清华大学", "他 来到 了 网易 杭研 大厦", "小明 硕士 毕业 与 中国 科学院"]
train = readtrain()
content = segmentWord(train[0])
lable = train[1]


# 划分
data_len = len(train[0])
train_len = int(rate * data_len)
test_len = data_len - train_len
print ("train_len:", train_len)
print ("test_len:", test_len)
train_content = content[:train_len]
test_content = content[train_len:]
train_lable = lable[:train_len]
test_lable = lable[train_len:]


# 计算权重
vectorizer = CountVectorizer()
tfidftransformer = TfidfTransformer()
tfidf = tfidftransformer.fit_transform(vectorizer.fit_transform(train_content))  # 先转换成词频矩阵，再计算TFIDF值
print ('tfidf.shape:',tfidf.shape)



model_file = 'svm_train_model.m'
clf = SVC()
if os.path.exists(model_file):
    clf = joblib.load(model_file)
else:
    # 训练和预测一体
    clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SVC(C=0.99, kernel = 'linear'))])
    clf = clf.fit(train_content, train_lable)
    joblib.dump(clf, model_file)

predicted = clf.predict(test_content)
print ('SVM模型准确率:',np.mean(predicted == test_lable))

code = alarm_code.loc[(alarm_code['BJLXDM'].isin(predicted))]

rs = pd.DataFrame(predicted, columns=['BJLXDM'])
rs = rs.merge(code, how='left')
rs['NR'] = test_content
rs['NR'] = rs['NR'].str.replace(' ','')
rs.columns = ['报警类别编码', '识案件分类', '报警内容']
print(rs.head(15))
#print (set(predicted))
#print metrics.confusion_matrix(test_lable,predicted) # 混淆矩阵


'''
model_file = 'svm_train_model.m'
clf = SVC()
clf = joblib.load(model_file)
docs = ["4月25日凌晨，吴某某在容县某酒吧与朋友一起喝酒，酒后离开酒吧时，"
        +"在酒吧一楼大门处趁酒意拉扯旁边一女子，该女子的同行人员就与吴"
        +"某某发生争执和肢体冲突，群众遂向“110”报警。"]
docs = segmentWord(docs)
predicted = clf.predict(docs)
alarm_code[alarm_code['BJLXDM'].isin(predicted)]
'''

# 单独预测
'''
word = vectorizer.get_feature_names()
weight = tfidf.toarray()
# 分类器
#clf = MultinomialNB().fit(tfidf, opinion)
docs = ["4月25日凌晨，吴某某在容县某酒吧与朋友一起喝酒，酒后离开酒吧时，"
        +"在酒吧一楼大门处趁酒意拉扯旁边一女子，该女子的同行人员就与吴"
        +"某某发生争执和肢体冲突，群众遂向“110”报警。"]
docs = segmentWord(docs)
predicted = clf.predict(docs)
alarm_code[alarm_code['BJLXDM'].isin(predicted)]
'''


# 循环调参
'''
parameters = {'vect__max_df': (0.4, 0.5, 0.6, 0.7),'vect__max_features': (None, 5000, 10000, 15000),
              'tfidf__use_idf': (True, False)}
grid_search = GridSearchCV(clf, parameters, n_jobs=1, verbose=1)
grid_search.fit(content, opinion)
best_parameters = dict()
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

'''