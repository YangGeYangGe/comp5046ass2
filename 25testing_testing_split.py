
# coding: utf-8

# In[1]:



import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfTransformer


# In[14]:


train = pd.read_csv("topic.csv", header=0, delimiter=",")
target = train['annotation']
vectorizer = CountVectorizer(analyzer = "word",
                             tokenizer = None,
                             preprocessor = None,
                             stop_words = None,
                             )
train_data_features = vectorizer.fit_transform(train['body']).toarray()


# In[3]:


cv = ShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
for train_index, test_index in cv.split(train_data_features):
    print("TRAIN:", train_index, "TEST:", test_index)


# In[53]:


# train_data_features[289]
traindata = []
traintarget = []
testdata = []
testtarget = []
splited = cv.split(train_data_features)
count = 0 
for train_index, test_index in splited:
#     traindata.append(train_data_features[train_index])
#     traintarget.append(target[train_index])
    traindata = train_data_features[train_index]
    traintarget = target[train_index]
    testdata = train_data_features[test_index]
    testtarget = target[test_index]
#     print(traintarget)
#     count += 1
    
# traintarget


# In[46]:


traindata


# In[37]:


classifier = LogisticRegression(C=1,
                        penalty='l2',
                        fit_intercept=True)


# In[38]:


classifier.fit(traindata, traintarget) 


# In[58]:



# traindata = []
# traintarget = []
# testdata = []
# testtarget = []

result = classifier.predict(testdata)
result 


# In[66]:


testtarget[1]


# In[74]:


count = 0
total = 0
idx = 0
# for i in range(0,228):
# #     if result[i] == testtarget[i]:
# #         count += 1
#     print(testtarget[i])
# #     total += 1

for i in testtarget:
    if result[idx] == i:
        count += 1
    total += 1
    idx += 1


# In[76]:


count/total


# In[78]:


cross_val_score(classifier, train_data_features, target, cv=cv)

