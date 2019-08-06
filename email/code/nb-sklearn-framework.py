#!/usr/bin/env python
# coding: utf-8

# # 垃圾邮件分类
# 
# https://www.kaggle.com/uciml/sms-spam-collection-dataset

# ## 读取数据

# In[1]:


import pandas as pd
import numpy as np 

data_dir = "../input/"

df = pd.read_csv(data_dir + '/spam.csv', encoding='latin-1')  
# 编码相关阅读http://blog.csdn.net/robertcpp/article/details/7837712 

# 查看数据
df.head()


# In[2]:


# 查看v2的样本


# In[3]:


# 查看v1的样本


# In[4]:


# 查看数据的纬度


# ## 把数据拆分成为训练集和测试集

# In[5]:


from sklearn.model_selection import train_test_split


# ## 构建模型所需要的数据格式：一个词汇表，以及训练及测试数据的计数信息：(句子id,单词id)->计数

# In[6]:


from sklearn.feature_extraction.text import CountVectorizer
# 调用库来构造分类器所需的输入数据


# 用训练集的单词来建立词汇表


# ## 图形化展示一些数据，获得更直观的理解

# In[7]:


import matplotlib.pyplot as plt # 画图常用库

#print(vectorizer.get_feature_names())
#统计每个单词出现的次数

# 排序：单词出现次数从高到低


# ## 进行模型训练以及预测

# In[8]:


from sklearn.naive_bayes import MultinomialNB


# ## 计算模型的准确率

# In[10]:


from sklearn.metrics import accuracy_score


# ## 其他常用指标: （Naive Bayes 第二节课会补充）

# In[12]:


from sklearn.metrics import classification_report,confusion_matrix


# ## 交叉验证的示范:

# In[13]:


from sklearn.model_selection import cross_val_score


# In[ ]:




