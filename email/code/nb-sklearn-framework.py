#!/usr/bin/env python
# coding: utf-8

# # 垃圾邮件分类
#
# https://www.kaggle.com/uciml/sms-spam-collection-dataset

# ## 读取数据

# In[1]:
import pandas as pd
import numpy as np

data_dir = "naive_bayes/email/input/"

df = pd.read_csv(data_dir + 'spam.csv', encoding='latin-1')
# 编码相关阅读http://blog.csdn.net/robertcpp/article/details/7837712

# 查看数据
df.head()

# In[2]:
# 查看v2的样本
df.v2.head()

# In[3]:
# 查看v1的样本
df.v1.head()

# In[4]:
# 查看数据的纬度
df.shape


# In[5]:
# ## 把数据拆分成为训练集和测试集
from sklearn.model_selection import train_test_split
data_train, data_test, labels_train, labels_test = train_test_split(df.v2, df.v1, test_size=0.2, random_state=0)

print(data_train.shape)
print(data_test.shape)

# ## 构建模型所需要的数据格式：一个词汇表，以及训练及测试数据的计数信息：(句子id,单词id)->计数
# In[6]:
from sklearn.feature_extraction.text import CountVectorizer
# 调用库来构造分类器所需的输入数据
vectorizer = CountVectorizer()

# 用训练集的单词来建立词汇表
data_train_count = vectorizer.fit_transform(data_train)
data_test_count = vectorizer.transform(data_test)

# CountVectorizer Demo
vectorizer_demo = CountVectorizer()
data_train_demo = ["We are good students students", "You are good student"]
data_train_count_demo  = vectorizer_demo.fit_transform(data_train_demo)
print (vectorizer_demo.vocabulary_)
print(data_train_count_demo.toarray())

# In[7]:
# ## 图形化展示一些数据，获得更直观的理解
import matplotlib.pyplot as plt # 画图常用库

#print(vectorizer.get_feature_names())
#统计每个单词出现的次数
occurrence = data_train_count.toarray().sum(axis = 0)
plt.plot(occurrence)
plt.show()
# 排序：单词出现次数从高到低
word_freq_df = pd.DataFrame({'term': vectorizer.get_feature_names(),
                             'occurrence': occurrence})
word_freq_df_sort = word_freq_df.sort_values(by=['occurrence'], ascending=False)
word_freq_df_sort.head()

# ## 进行模型训练以及预测
# In[8]:
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(data_train_count, labels_train)
predictions = clf.predict(data_test_count)
print(predictions)

# ## 计算模型的准确率
# In[10]:
from sklearn.metrics import accuracy_score
print(accuracy_score(labels_test, predictions))

# ## 其他常用指标: （Naive Bayes 第二节课会补充）
# In[12]:
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(labels_test, predictions))
print(confusion_matrix(labels_test, predictions))

# ## 交叉验证的示范:
# In[13]:
from sklearn.model_selection import cross_val_score
x = df.v2
y = df.v1
vect = CountVectorizer()
x_vect = vect.fit_transform(x)
cross_val = cross_val_score(MultinomialNB(), x_vect, y, cv=10, scoring='accuracy')
print(cross_val)
print(np.mean(cross_val))
