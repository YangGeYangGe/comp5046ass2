
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfTransformer
import csv
import nltk

f = open('topic.csv')

train = list(csv.reader(f))
testing = train[1][3]

text = nltk.word_tokenize(testing)
print(nltk.pos_tag(text))




# from nltk import word_tokenize, pos_tag, ne_chunk
# sentence = "Mark and John are working at Google."
# ne_chunk(pos_tag(word_tokenize(sentence)))[0]




# from nltk import word_tokenize, pos_tag, ne_chunk
# sentence = "Mark and John are working at Google"
# ne_tree = ne_chunk(pos_tag(word_tokenize(sentence)))
# ne_tree
# from nltk.chunk import conlltags2tree, tree2conlltags
# iob_tagged = tree2conlltags(ne_tree)
# iob_tagged