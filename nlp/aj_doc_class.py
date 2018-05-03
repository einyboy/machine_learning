import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import jieba
import jieba.analyse
import jieba.posseg as pseg
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

file_path = "D:\\DATA_SET\\SJAL\\AJ_INFO.csv"
file_dict = "D:\\DATA_SET\\SJAL\\BJLXDMB.csv"

alarm_code = pd.read_csv(file_dict, dtype=str).drop(['MS','XS','PX'], axis=1)
data = pd.read_csv(file_path, dtype=str)


data.head()
data.info(True)
data.isnull().sum()
data = data.dropna(axis=0)
bjlxdm = data['BJLXDM'].value_counts()[:25]
#data.loc[~(data['BJLXDM'].isin(bjlxdm.index))]
data.loc[~(data['BJLXDM'].isin(bjlxdm.index)),'BJLXMC']= 'ÆäËû'
data.loc[~(data['BJLXDM'].isin(bjlxdm.index)),'BJLXDM']= '000000'
dd = pd.DataFrame(bjlxdm)
dd['C1'] = dd.index
dd.columns = ['cnt','BJLXDM']
dd = dd.merge(alarm_code, how='left')
dd.plot.bar('BJLXMC','cnt')
plt.show()

'''
sentence = data.iloc[0]['NR']
text1 = [[word for word in jieba.cut(sentence)]]
key_word = jieba.analyse.extract_tags(sentence, topK=10, withWeight=False, allowPOS=())
key_word = jieba.analyse.textrank(sentence, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))
words = pseg.cut(sentence)
pos_seg = ' '.join([ word +'/' + flag  for word, flag in words])
'''

out_put = 'Train.csv'
pd_out = data[['NR','BJLXDM']]
pd_out.columns = ['content','label']
pd_out.to_csv(out_put, index=False, quoting=csv.QUOTE_ALL)