from __future__ import print_function
from dataset import Dataset
import features_evaluation
from collections import defaultdict
import pickle
import random as rn
rn.seed(2)
from numpy.random import seed
seed(2)
from tensorflow import set_random_seed
set_random_seed(2)
import tensorflow as tf
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
from keras import backend as K
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm


if __name__ == "__main__":
    filehandler = open(features_evaluation.SELECTED_FEATURES_CORPUS_CHI2, 'r')
    corpus = pickle.load(filehandler)
    dataset = Dataset(corpus=corpus)

    X = dataset.get_train_x()
    y = dataset.get_train_y()

    scores_dict = defaultdict(list)


    clf1 = LogisticRegression(C=0.05, random_state=1, class_weight='balanced')
    clf2 = RandomForestClassifier(random_state=1)
    clf3 = svm.SVC(C=0.35, class_weight='balanced')
    clf4 = RidgeClassifier(alpha=2.5)
    clf5 = AdaBoostClassifier(n_estimators=150)
    eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('svm', clf3),
                                        ('rc', clf4), ('ab', clf5)],
                            voting='hard')

    for clf, label in zip([clf1, clf2, clf3, clf4, clf5, eclf],
                          ['Logistic Regression', 'Random Forest', 'SVM',
                           'Ridge Classifier', 'Ada boost', 'Ensemble']):
        scores = cross_val_score(clf, X.toarray(), y, cv=5, scoring='f1_macro')
        scores_dict[label].append(scores.mean())
        print("f1_macro: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

    X, y = dataset.get_resampled_train_X_y(kind='regular')
    clf1.fit(X.toarray(), y)
    clf2.fit(X.toarray(), y)
    clf3.fit(X.toarray(), y)
    clf4.fit(X.toarray(), y)
    clf5.fit(X.toarray(), y)
    eclf.fit(X.toarray(), y)

    # X_test = dataset.get_test_x()
    # y_test = dataset.get_test_y()

    # for clf, label in zip([clf1, clf2, clf3, clf4, clf5, eclf],
    #                       ['Logistic Regression', 'Random Forest',
    #                        'SVM', 'Ridge Classifier', 'Ada boost', 'Ensemble']):
    #     score = f1_score(y_test, clf.predict(X_test.toarray()))
    #     print("f1_macro_test: %0.2f [%s]" % (score, label))
    #     print(precision_score(y_test, clf.predict(X_test.toarray())))
    #     print(recall_score(y_test, clf.predict(X_test.toarray())))

    # predicted = clf.predict(X_test.toarray())
    #
    # for prediction in predicted:
    #     print(prediction)
    #
    # print(type(predicted))
    #
    # filehandler = open('predicted_by_ensamble.pkl', 'w')
    # pickle.dump(predicted, filehandler, pickle.HIGHEST_PROTOCOL)
    # filehandler.close()
