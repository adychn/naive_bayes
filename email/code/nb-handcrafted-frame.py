#!/usr/bin/env python
# coding: utf-8

# # 垃圾邮件分类
# https://www.kaggle.com/uciml/sms-spam-collection-dataset

# 拿到数据首先读入拿到数据

# In[8]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# 读取数据
data_dir = "../input/"
df = pd.read_csv(data_dir + '/spam.csv', encoding='latin-1')

# 把数据拆分成为训练集和测试集
data_train, data_test, labels_train, labels_test = train_test_split(
    df.v2,
    df.v1, 
    test_size=0.2, 
    random_state=0)  

#print ('拆分过后的每个邮件内容')
#print (data_train[:10])
#print ('拆分过后每个邮件是否是垃圾邮件')
#print (labels_train[:10])


# 建立词汇表，统计两个类目下面的共词计数

# In[2]:


'''
    用一个dictionary保存词汇，并给每个词汇赋予唯一的id
'''
def GetVocabulary(data): 
    return


# 把文章变成词向量
# 

# In[3]:


'''
    把文本变成向量的表示形式，以便进行计算
'''
def Document2Vector(vocab_dict, data):
    return


# In[4]:


# 把训练集的句子全部变成向量形式


# 做naive bayes 训练，得到训练集每个词概率

# In[5]:


'''
    在训练集计算两种概率：
        1. 词在每个分类下的概率，比如P('email'|Spam)
        2. 每个分类的概率，比如P(Spam)
        
    这里的计算实现巧妙利用了numpy的array结构：
        1. 在每个分类下创建一个与词汇量大小相等的vector(即 numpy array), 即spam_word_counter 和 ham_word_counter
        2. 在遍历每一个句子的时候，直接与句子对应的vector相加，累积每个单词出现的次数
        3. 在遍历完所有句子之后，再除以总词汇量，得到每个单词的概率
'''
def NaiveBayes_train(train_matrix,labels_train):
    return


# 进行测试集预测

# In[6]:


'''
    对测试集进行预测，按照公式计算例子在两个分类下的概率，选择概率较大者作为预测结果
'''
def Predict(test_word_vector, p_spam_vector, p_spam, p_ham_vector, p_ham):
    return


# In[7]:


# 检测模型

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score


#print (accuracy_score(labels_test, predictions))
#print (classification_report(labels_test, predictions))
#print (confusion_matrix(labels_test, predictions))


# In[ ]:




