import codecs
import itertools
import json
import logging
import os

import gensim
import numpy as np
from gensim import matutils
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tensorflow.python.platform import gfile

path = "C:\\Users\\Ana\\Desktop\\faks\\TAR\\reddit-training-ready-to-share-preprocessed\\lemmatized_all"
output_path = "C:\\Users\\Ana\\Desktop\\faks\\TAR\\reddit-training-ready-to-share-preprocessed\\no\\words.txt"
labels = []

def iter_subjects(fname, log_every=None):
    extracted = 0
    subjects = []
    file_glob = os.path.join(path, '*.txt')
    subjects.extend(gfile.Glob(file_glob))
    for subject in subjects:
        if "neg" in subject:
            labels.append(0)
        else:
            labels.append(1)
        if log_every and extracted % log_every == 0:
            logging.info("extracting 20newsgroups file #%i: %s" % (extracted, subject))
        with open(subject, "r") as f:
            with codecs.open(subject,'r',encoding='utf8') as f:
                content = f.read()
        yield content
        extracted += 1

class RedditCorpus(object):
    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        for text in iter_subjects(self.fname):
            yield list(gensim.utils.tokenize(text, lower=True))

#creating dictionary and tfidf model
tokenized_corpus = RedditCorpus(path)
documents = list(itertools.chain(tokenized_corpus))
dct = Dictionary(documents)  # fit dictionary
corpus = [dct.doc2bow(doc) for doc in documents]  # convert dataset to BoW format
model = TfidfModel(corpus, id2word=dct)  # fit model
vector = model[corpus]

#write scores to files
subjects = []
file_glob = os.path.join(path, '*.txt')
subjects.extend(gfile.Glob(file_glob))
for subject in subjects:
    doc = model[corpus[step]]
    d = {dct.get(token_id): tfidf for token_id,tfidf in doc}
    output_file = open(output_path+"\\"+subject.split("\\")[-1],"w") 
    output_file.write(json.dumps(d))  
    output_file.close() 

#creating tf-idf matrix
tfidf_dense = matutils.corpus2dense(model[corpus], num_terms).T 

#creating random forrest classifier for extracting important features
num_terms = len(model[corpus].obj.idfs)
X_train, X_test, y_train, y_test = train_test_split(tfidf_dense, labels, test_size=0.4, random_state=0)
clf = RandomForestClassifier(n_estimators=10, random_state=0, n_jobs=1)
clf.fit(X_train, y_train)
importances = clf.feature_importances_
sorted_i = sorted(importances)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("accuracy: ",acc)
print("f1 ",f1)

#checking model after leaving important features
top_words = []
tfidf_important = np.empty((len(labels),0))
for i in range(len(importances)):
    if(importances[i]>0.002):
        top_words.append(dct.get(i))
        tfidf_important = np.hstack((tfidf_important,np.vstack(tfidf_dense[:,i])))
print(top_words)

X_train, X_test, y_train, y_test = train_test_split(tfidf_important, labels, test_size=0.1, random_state=0)
clf = RandomForestClassifier(n_estimators=10, random_state=0, n_jobs=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

new_acc = accuracy_score(y_test, y_pred)
new_f1 = f1_score(y_test, y_pred)
print("new accuracy ",new_acc)
print("new f1 ",new_f1)

#write words to file
output_file = open(output_path,"w",encoding="utf-8")
for word in top_words:
    output_file.write("%s\n" % word)
output_file.close() 

