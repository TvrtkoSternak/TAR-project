from __future__ import print_function

import pickle

import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

import dataset

SELECTED_FEATURES_CORPUS_CHI2 = "selected_corpus_chi2.pkl"
SELECTED_FEATURES_CORPUS_FREQUENCY_INTERVAL = "selected_corpus_freq_interval.pkl"
SELECTED_FEATURES_CORPUS_TFIDF = "selected_corpus_tfidf.pkl"

if __name__ == "__main__":
    dataset = dataset.Dataset()
    train_x = dataset.get_train_x()
    train_y = dataset.get_train_y()

    k_best_features = SelectKBest(chi2, k=150)
    k_best_features.fit(dataset.get_train_x(), dataset.get_train_y())
    k_selected_features = \
        np.array(dataset.vectorizer.get_feature_names())[np.array(k_best_features.get_support())]

    filehandler = open(SELECTED_FEATURES_CORPUS_CHI2, 'w')
    pickle.dump(k_selected_features, filehandler, pickle.HIGHEST_PROTOCOL)
    filehandler.close()

