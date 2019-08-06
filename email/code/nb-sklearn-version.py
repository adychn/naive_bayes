#!/usr/bin/env python
# coding: utf-8

# # 垃圾邮件分类
# 
# https://www.kaggle.com/uciml/sms-spam-collection-dataset

# ## 读取数据

# In[2]:


import pandas as pd
import numpy as np 

data_dir = "../input/"

df = pd.read_csv(data_dir + '/spam.csv', encoding='latin-1')  
# 编码相关阅读http://blog.csdn.net/robertcpp/article/details/7837712 

# 查看数据
df.head()


# In[3]:


# 查看v2的样本
df.v2.head()


# In[4]:


# 查看v1的样本
df.v1.head()


# In[5]:


# 查看数据的纬度
df.shape


# ## 把数据拆分成为训练集和测试集

# In[6]:


from sklearn.model_selection import train_test_split

# 把数据拆分成训练集和测试集
data_train, data_test, labels_train, labels_test = train_test_split(
    df.v2,
    df.v1, 
    test_size=0.2, 
    random_state=0) 

# 查看训练集样本
print (data_train.head())
# 查看训练集标注
print (labels_train.head())
# 查看训练集的样本个数
print(data_train.shape)
# 查看测试机的样本个数
print(data_test.shape)


# ## 构建模型所需要的数据格式：一个词汇表，以及训练及测试数据的计数信息：(句子id,单词id)->计数

# In[7]:


from sklearn.feature_extraction.text import CountVectorizer
# 调用库来构造分类器所需的输入数据
vectorizer = CountVectorizer()

#fit_transform一共完成了两件事
#http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer.fit_transform
#fit: 统计单词的总个数，建成一个表，每个单词给一个标号 (这个库内部实现有一个缺陷，会把长度为1的单词给过滤掉了)
#transform:统计每句话每个单词出现的次数

# 用训练集的单词来建立词库，因为测试集的数据在现实场景中属于未知数据，且把训练集每句话词变成向量形态

# CountVectorizer Demo
# data_train_demo = ["We are good students", "You are good student"]
# data_train_count_demo  = vectorizer.transform(data_train_demo)
# print (vectorizer.vocabulary_)
# print(data_train_count.toarray())

data_train_count = vectorizer.fit_transform(data_train)
# 把测试集每句话变成向量形态
data_test_count  = vectorizer.transform(data_test)

# 看看这些数据长什么样
# 词汇表 （太长了，我这里注释掉）
# print (vectorizer.vocabulary_)
# print(data_train_count.toarray()[0:4])
# 训练数据纬度
#print (data_train_count.shape)
# 测试数据纬度
#print (data_test_count.shape)


# ## 图形化展示一些数据，获得更直观的理解

# In[20]:


import matplotlib.pyplot as plt # 画图常用库

# 我们来看看单词的分布
# 统计每个单词出现的次数
occurrence = data_train_count.toarray().sum(axis=0) #把矩阵按列求和
plt.plot(occurrence)
plt.show() # 显示图形

# 按照每个词出现的次数从高到低进行排序, get_feature_names其实就是访问vocabulary_
word_freq_df = pd.DataFrame({'term': vectorizer.get_feature_names(), 'occurrence':occurrence})
word_freq_df_sort = word_freq_df.sort_values(by=['occurrence'], ascending=False)
word_freq_df_sort.head()


# ## 进行模型训练以及预测

# In[9]:


from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()
clf.fit(data_train_count, labels_train)
predictions = clf.predict(data_test_count)
print(predictions)


# ## 计算模型的准确率

# In[10]:


from sklearn.metrics import accuracy_score

print (accuracy_score(labels_test, predictions))


# ## 其他常用指标: （Precision, Recall, F1-score, confusion_matrix）

# In[11]:


from sklearn.metrics import classification_report,confusion_matrix
print (classification_report(labels_test, predictions))
print (confusion_matrix(labels_test, predictions))


# ## 交叉验证的示范:

# In[12]:


from sklearn.model_selection import cross_val_score
# 从df获得全部邮件内容和标注
data_content = df.v2
data_label = df.v1
vect = CountVectorizer()
# 在整体数据集上构建词汇表以及转化成计数格式
data_count = vect.fit_transform(data_content)
# 交叉验证
cross_val = cross_val_score(clf, data_count, data_label, cv=20, scoring='accuracy')
# 打印每组实验测试集的准确率
print (cross_val)
# 求平均值
print (np.mean(cross_val))


# In[ ]:




