
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfTransformer


from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 

print("lemmatise words")

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]



train = pd.read_csv("topic.csv", header=0, delimiter=",")
# print(train.shape)
# print(train.columns.values)
# print(train['annotation'])

target = train['annotation']

# print(train)
vectorizer = CountVectorizer(analyzer = "word",
                             tokenizer = LemmaTokenizer(),
                             preprocessor = None,
                             stop_words = None,
                             
                             )

train_data_features = vectorizer.fit_transform(train['body']).toarray()

# print(type(vectorizer.fit_transform(train['body'])))
# print(train_data_features)

vocab = vectorizer.get_feature_names()

# print(vocab)


cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

# print(cv.get_n_splits(train_data_features))

lr = LogisticRegression(C=1,
                        penalty='l2',
                        fit_intercept=True)

traindata = []
traintarget = []
testdata = []
testtarget = []
splited = cv.split(train_data_features)
for train_index, test_index in splited:
    traindata.append(train_data_features[train_index])
    traintarget.append(target[train_index])
    testdata.append(train_data_features[test_index])
    testtarget.append(target[test_index])
    # traindata = train_data_features[train_index]
    # traintarget = target[train_index]
    # testdata = train_data_features[test_index]
    # testtarget = target[test_index]

lr.fit(traindata[0], traintarget[0]) 
result = lr.predict(testdata[0])

import numpy as np
coefs=lr.coef_
ddd = []
for i in target:
    if i not in ddd:
        ddd.append(i)
ddd_count = 0
for coe in coefs:
    print(ddd[ddd_count],end='|')
    ddd_count += 1
    top_three = np.argpartition(coe, -3)[-3:]
    last_three = np.argpartition(coe, 3)[:3]
    
    for i in top_three:
        print("'"+vocab[i],end="'")
    print(" |", end='')
    for i in last_three:
        print("'"+vocab[i],end="'")
    print()    
print()



# dic = {}
# for t in testtarget[0]:
#     if t not in dic:
#         dic[t] = 0
#     dic[t] += 1

# dicpredict = {}
# for t in result:
#     if t not in dicpredict:
#         dicpredict[t] = 0
#     dicpredict[t] += 1

# dicintersect = {}
# for r in range(0,len(result)):
#     if testtarget[0].tolist()[r] == result[r]:
#         if result[r] not in dicintersect:
#             dicintersect[result[r]] = 0
#         dicintersect[result[r]] += 1

# for k in dic:
#     if k not in dicpredict:
#         print(k+" precision:0, recall:0, f1_score:0")
#     elif k not in dicintersect:
#         print(k+" precision:0, recall:0, f1_score:0")
#     else:
#         pre = dicintersect[k]/dicpredict[k]
#         rec = dicintersect[k]/dic[k]
#         f1 = 2*pre*rec/(pre+rec)
#         print(k + " precision:%0.2f, recall:%0.2f, f1_score:%0.2f"%(pre,rec,f1))

# # print("to copy!")
# # for k in dic:
# #     if k not in dicpredict:
# #         print(k+"|0|0|0|")
# #     if k not in dicintersect:
# #         print(k+"|0|0|0|")
# #     else:
# #         pre = dicintersect[k]/dicpredict[k]
# #         rec = dicintersect[k]/dic[k]
# #         f1 = 2*pre*rec/(pre+rec)
# #         print(k + "|%0.2f|%0.2f|%0.2f|"%(pre,rec,f1))
# # print("copied!")

# # scores = cross_val_score(lr, train_data_features, target, cv=cv)

# # print("Accuracy_Baseline: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))

# accuracy_score = cross_val_score(lr, train_data_features, target, cv=cv)
# precision = cross_val_score(lr, train_data_features, target, cv=cv,scoring='precision_macro')
# recall_scores = cross_val_score(lr, train_data_features, target, cv=cv,scoring='recall_macro')
# f1_scores = cross_val_score(lr, train_data_features, target, cv=cv,scoring='f1_macro')


# print("Accuracy: %0.5f, precision: %0.5f, recall: %0.5f, f1: %0.5f" % (accuracy_score.mean(), precision.mean(), recall_scores.mean(), f1_scores.mean()))




