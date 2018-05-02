
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfTransformer

train = pd.read_csv("topic.csv", header=0, delimiter=",")
# print(train.shape)
# print(train.columns.values)
# print(train['annotation'])

target = train['annotation']

print(train)
vectorizer = CountVectorizer(analyzer = "word",
                             tokenizer = None,
                             preprocessor = None,
                             stop_words = 'english',
                             )
train_data_features = vectorizer.fit_transform(train['body']).toarray()

# print(train_data_features)

#vocab = vectorizer.get_feature_names()

# print(vocab)


cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

lr = LogisticRegression(C=1,
                        penalty='l2',
                        fit_intercept=True)

scores = cross_val_score(lr, train_data_features, target, cv=cv)



print("Accuracy_Baseline: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))




