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
data_dir = "email/input/"
df = pd.read_csv(data_dir + 'spam.csv', encoding='latin-1')
# 把数据拆分成为训练集和测试集
data_train, data_test, labels_train, labels_test = train_test_split(
    df.v2,
    df.v1,
    test_size=0.2,
    random_state=0)

#print ('拆分过后的每个邮件内容')
print (data_train[:10])
#print ('拆分过后每个邮件是否是垃圾邮件')
print (labels_train[:10])

# 建立词汇表，统计两个类目下面的共词计数
# In[2]:
'''
    用一个dictionary保存词汇，并给每个词汇赋予唯一的id
'''
def GetVocabulary(data):
    vocab_dict = {}
    wid = 0
    for document in data:
        words = document.split()
        for word in words:
            word = word.lower()
            if word not in vocab_dict:
                vocab_dict[word] = wid
                wid += 1
    return vocab_dict

vocab_dict = GetVocabulary(data_train)
print(len(vocab_dict.keys()))

# 把文章变成词向量
# In[3]:
'''
    把文本变成向量的表示形式，以便进行计算
    [1, 0, 1, 2, 4, 0, 0, 1, ..., 1]
'''
def Document2Vector(vocab_dict, document):
    word_vector = np.zeros(len(vocab_dict))
    words = document.split()
    for word in words:
        word = word.lower()
        if word in vocab_dict:
            wid = vocab_dict[word]
            word_vector[wid] += 1

    return word_vector

example = Document2Vector(vocab_dict, 'We are students')
print(example)
print(example[vocab_dict['are']])

# 把训练集的句子全部变成向量形式
# In[4]:
train_matrix = []
for document in data_train:
    word_vector = Document2Vector(vocab_dict, document)
    train_matrix.append(word_vector)


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
def NaiveBayes_train(train_matrix, labels_train):
    num_docs = len(train_matrix)
    num_words = len(train_matrix[0])

    spam_word_counter = np.ones(num_words)
    ham_word_counter = np.ones(num_words)

    spam_total_count = 0 # number of spam words count
    ham_total_count = 0

    spam_count = 0  # spam document count
    ham_count = 0

    for vector, label in zip(train_matrix, labels_train):
        # try:
        #
        # except:
        #     continue
        # if i not in labels_train: # if the label not in training set
        #     continue

        if i % 500 == 0:
            print("Training on doc id: {}".format(i))
        # print("labels_train[{}]:{}, type({})".format(i, labels_train[i], type(labels_train[i]))) # debug key error
        if label == "spam": # this is access the series using its index, because it has index, if no index then its the position.
            spam_word_counter += vector #train_matrix is a list of documents
            spam_count += 1
            spam_total_count += sum(vector)
        else:
            ham_word_counter += vector
            ham_count += 1
            ham_total_count += sum(vector)

    p_spam_vector = np.log(spam_word_counter / (spam_total_count + num_words)) # num_words is the smoothing
    p_ham_vector = np.log(ham_word_counter / (ham_total_count + num_words))
    p_spam = np.log(spam_count / num_docs)
    p_ham = np.log(ham_count / num_docs)

    return p_spam_vector, p_spam, p_ham_vector, p_ham

p_spam_vector, p_spam, p_ham_vector, p_ham = NaiveBayes_train(train_matrix, labels_train) # use labels_train to remove index

# 进行测试集预测

# In[6]:
'''
    对测试集进行预测，按照公式计算例子在两个分类下的概率，选择概率较大者作为预测结果
'''
def Predict(test_word_vector, p_spam_vector, p_spam, p_ham_vector, p_ham):
    spam = sum(test_word_vector * p_spam_vector) + p_spam # log(probabilty so plus)
    ham = sum(test_word_vector * p_ham_vector) + p_ham
    if spam > ham:
        return 'spam'
    else:
        return 'ham'

predictions = []
for document in data_test:
    test_word_vector = Document2Vector(vocab_dict, document)
    pred = Predict(test_word_vector, p_spam_vector, p_spam, p_ham_vector, p_ham)
    predictions.append(pred)

print(len(predictions))

# In[7]:


# 检测模型

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

print (accuracy_score(labels_test, predictions))
print (classification_report(labels_test, predictions))
print (confusion_matrix(labels_test, predictions))


# In[ ]:
