
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfTransformer

import nltk
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
import time

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        text = word_tokenize(doc)
        return nltk.pos_tag(text)
     

start = time.time()

train = pd.read_csv("topic.csv", header=0, delimiter=",")
# print(train.shape)
# print(train.columns.values)
# print(train['annotation'])

target = train['annotation']

print(train)
vectorizer = CountVectorizer(analyzer = "word",
                             tokenizer = LemmaTokenizer(),
                             preprocessor = None,
                             stop_words = None,
                             
                             )
train_data_features = vectorizer.fit_transform(train['body']).toarray()

# print(type(vectorizer.fit_transform(train['body'])))
# print(train_data_features)

vocab = vectorizer.get_feature_names()

print(vocab)
# print(vectorizer.vocabulary_.get('\n'))

cv = ShuffleSplit(n_splits=1, test_size=0.1, random_state=0)

# print(cv.get_n_splits(train_data_features))


lr = LogisticRegression(C=1,
                        penalty='l2',
                        fit_intercept=True)

scores = cross_val_score(lr, train_data_features, target, cv=cv)

end = time.time()
print(end - start)

print("Accuracy_Baseline: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))




