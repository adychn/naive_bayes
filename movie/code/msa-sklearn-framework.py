#!/usr/bin/env python
# coding: utf-8

# # Movie Sentiment Analysis
# https://www.kaggle.com/c/word2vec-nlp-tutorial/

#  拿到数据首先读入拿到数据

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt # 画图常用库

import pandas as pd


train = pd.read_csv('../input/labeledTrainData.tsv', delimiter="\t")
test = pd.read_csv('../input/testData.tsv', delimiter="\t")
train.head()                


# In[2]:


# test data比如train data少了label的一维
print (train.shape)
print (test.shape)


# In[3]:


'''
    清理数据，文本中包含HTML的符号比如<>，我们使用正则表达式简单地清理一下
'''
import re  #正则表达式

def review_preprocessing(review):
    #只保留英文单词
    review_text = re.sub("[^a-zA-Z]"," ", review)
    
    #变成小写
    words = review_text.lower()
    
    return(words)

# 把训练集的文本和标注分开
# 1. 把标注提取出来

# 2. 把文本提取出来
    
    
# 3. 转化成numpy数组        


# 对测试集的文本做同样的事情


# In[4]:


from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer

# 简单的计数


# 使用tf-idf


# In[5]:


# 多项式朴素贝叶斯
from sklearn.naive_bayes import MultinomialNB 


# In[7]:


# 把结果保存到csv文件中，并进行提交: https://www.kaggle.com/c/word2vec-nlp-tutorial/leaderboard
#df = pd.DataFrame({"id": test['id'],"sentiment": pred})
#df.to_csv('submission.csv',index = False, header=True)


# In[ ]:




