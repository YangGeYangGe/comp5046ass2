
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


# In[40]:


transformer = TfidfTransformer(smooth_idf=False)


# In[41]:


train = pd.read_csv("topic.csv", header=0, delimiter=",")
# print(train.shape)
# print(train.columns.values)
# print(train['annotation'])

target = train['annotation']

# print(train)
vectorizer = CountVectorizer(analyzer = "word",
                             tokenizer = None,
                             preprocessor = None,
                             stop_words = 'None', 
                             )
X_train_counts = vectorizer.fit_transform(train['body'])
train_data_features = X_train_counts

tf_transformer = TfidfTransformer(use_idf=False).fit(train_data_features)
X_train_tf = tf_transformer.transform(X_train_counts)


# In[42]:


# vocab = vectorizer.get_feature_names()
asd = X_train_tf.toarray()
# print(vocab)
asd


# In[43]:


cv = ShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
# for train_index, test_index in cv.split(train_data_features):
#     print("TRAIN:", train_index, "TEST:", test_index)


# In[44]:


traindata = []
traintarget = []
testdata = []
testtarget = []
splited = cv.split(asd)
count = 0 
for train_index, test_index in splited:
#     traindata.append(train_data_features[train_index])
#     traintarget.append(target[train_index])
    traindata = asd[train_index]
    traintarget = target[train_index]
    testdata = asd[test_index]
    testtarget = target[test_index]

traintarget


# In[45]:




# print(cv.get_n_splits(train_data_features))


classifier = LogisticRegression(C=1,
                        penalty='l2',
                        fit_intercept=True)


classifier.fit(traindata, traintarget) 


# In[46]:


result = classifier.predict(testdata)
import numpy as np
# accuracy
np.mean(result == testtarget)


# In[47]:


from sklearn import metrics
print(metrics.classification_report(testtarget, result,
                                    target_names=testtarget))


# In[32]:


0.66376

