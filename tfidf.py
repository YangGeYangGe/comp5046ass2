
# coding: utf-8

# In[39]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfTransformer

import nltk
from nltk import word_tokenize, pos_tag    
from nltk.chunk import conlltags2tree, tree2conlltags      
from nltk.stem import WordNetLemmatizer 
import time



print("tfidf")
transformer = TfidfTransformer(smooth_idf=False)



train = pd.read_csv("topic.csv", header=0, delimiter=",")
target = train['annotation']
vectorizer = CountVectorizer(analyzer = "word",
                             tokenizer = None,
                             preprocessor = None,
                             stop_words = None, 
                             )
X_train_counts = vectorizer.fit_transform(train['body'])
# train_data_features = X_train_counts

tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts).toarray()


# asd = X_train_tf.toarray()
# print(vocab)
# asd
# In[43]:

cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
# for train_index, test_index in cv.split(train_data_features):
#     print("TRAIN:", train_index, "TEST:", test_index)


# In[44]:

lr = LogisticRegression(C=1,
                        penalty='l2',
                        fit_intercept=True)
traindata = []
traintarget = []
testdata = []
testtarget = []
splited = cv.split(X_train_tf)
for train_index, test_index in splited:
    traindata.append(X_train_tf[train_index])
    traintarget.append(target[train_index])
    testdata.append(X_train_tf[test_index])
    testtarget.append(target[test_index])
    # traindata = train_data_features[train_index]
    # traintarget = target[train_index]
    # testdata = train_data_features[test_index]
    # testtarget = target[test_index]

lr.fit(traindata[0], traintarget[0]) 
result = lr.predict(testdata[0])

dic = {}
for t in testtarget[0]:
    if t not in dic:
        dic[t] = 0
    dic[t] += 1

dicpredict = {}
for t in result:
    if t not in dicpredict:
        dicpredict[t] = 0
    dicpredict[t] += 1

dicintersect = {}
for r in range(0,len(result)):
    if testtarget[0].tolist()[r] == result[r]:
        if result[r] not in dicintersect:
            dicintersect[result[r]] = 0
        dicintersect[result[r]] += 1

for k in dic:
    if k not in dicpredict:
        print(k+" precision:0, recall:0, f1_score:0")
    elif k not in dicintersect:
        print(k+" precision:0, recall:0, f1_score:0")
    else:
        pre = dicintersect[k]/dicpredict[k]
        rec = dicintersect[k]/dic[k]
        f1 = 2*pre*rec/(pre+rec)
        print(k + " precision:%0.2f, recall:%0.2f, f1_score:%0.2f"%(pre,rec,f1))

# print("to copy!")
# for k in dic:
#     if k not in dicpredict:
#         print(k+"|0|0|0|")
#     elif k not in dicintersect:
#         print(k+"|0|0|0|")
#     else:
#         pre = dicintersect[k]/dicpredict[k]
#         rec = dicintersect[k]/dic[k]
#         f1 = 2*pre*rec/(pre+rec)
#         print(k + "|%0.2f|%0.2f|%0.2f|"%(pre,rec,f1))
# print("copied!")


accuracy_score = cross_val_score(lr, X_train_tf, target, cv=cv)
precision = cross_val_score(lr, X_train_tf, target, cv=cv,scoring='precision_macro')
recall_scores = cross_val_score(lr, X_train_tf, target, cv=cv,scoring='recall_macro')
f1_scores = cross_val_score(lr, X_train_tf, target, cv=cv,scoring='f1_macro')



print("Accuracy: %0.5f, precision: %0.5f, recall: %0.5f, f1: %0.5f" % (accuracy_score.mean(), precision.mean(), recall_scores.mean(), f1_scores.mean()))


# tfidf
# Entertainment precision:0.76, recall:0.62, f1_score:0.68
# Politics precision:0.74, recall:0.65, f1_score:0.69
# Sports precision:0.89, recall:0.82, f1_score:0.85
# Business precision:0.91, recall:0.45, f1_score:0.61
# Other precision:0, recall:0, f1_score:0
# Society precision:0.47, recall:0.89, f1_score:0.61
# War precision:0, recall:0, f1_score:0
# Health precision:0, recall:0, f1_score:0
# Error precision:0, recall:0, f1_score:0
# Science and Technology precision:1.00, recall:0.17, f1_score:0.29
# to copy!
# Entertainment|0.76|0.62|0.68|
# Politics|0.74|0.65|0.69|
# Sports|0.89|0.82|0.85|
# Business|0.91|0.45|0.61|
# Other|0|0|0|
# Society|0.47|0.89|0.61|
# War|0|0|0|
# Health|0|0|0|
# Error|0|0|0|
# Science and Technology|1.00|0.17|0.29|
# copied!
# Accuracy: 0.58996, precision: 0.58996, recall: 0.58996, f1: 0.58996




# tfidf
# high precision:0.63, recall:0.85, f1_score:0.72
# low precision:0.68, recall:0.39, f1_score:0.49
# to copy!
# high|0.63|0.85|0.72|
# low|0.68|0.39|0.49|
# copied!
# Accuracy: 0.60699, precision: 0.60699, recall: 0.60699, f1: 0.60699

# traindata = []
# traintarget = []
# testdata = []
# testtarget = []
# splited = cv.split(asd)
# count = 0 
# for train_index, test_index in splited:
# #     traindata.append(train_data_features[train_index])
# #     traintarget.append(target[train_index])
#     traindata = asd[train_index]
#     traintarget = target[train_index]
#     testdata = asd[test_index]
#     testtarget = target[test_index]

# traintarget


# # In[45]:




# # print(cv.get_n_splits(train_data_features))


# classifier = LogisticRegression(C=1,
#                         penalty='l2',
#                         fit_intercept=True)


# classifier.fit(traindata, traintarget) 


# # In[46]:


# result = classifier.predict(testdata)
# import numpy as np
# # accuracy
# np.mean(result == testtarget)


# # In[47]:


# from sklearn import metrics
# print(metrics.classification_report(testtarget, result,
#                                     target_names=testtarget))


# # In[32]:


# 0.66376

