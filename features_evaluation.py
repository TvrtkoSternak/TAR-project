from __future__ import print_function
from sklearn import tree
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import f1_score
import dataset
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

if __name__ == "__main__":
    dataset = dataset.Dataset()
    clf = tree.DecisionTreeClassifier()
    train_x = dataset.get_train_x()
    train_y = dataset.get_train_y()

    print('-------------------------------------------------')
    print("no oversampling:")

    train_x = dataset.get_train_x()
    train_y = dataset.get_train_y()
    clf = clf.fit(train_x, train_y)
    print('feature importances: ', end='')
    print(clf.feature_importances_)
    model = SelectFromModel(clf, prefit=True)
    print('model support vector: ', end='')
    print(model.get_support())
    selected_features = np.array(
        dataset.vectorizer.get_feature_names())[np.array(model.get_support())]

    print(len(selected_features))
    # print("----------------------------------------------")
    #
    # for feature in selected_features:
    #     print feature

    print ('f1 train: ', end='')
    print(f1_score(dataset.get_train_y(), clf.predict(dataset.get_train_x())))
    print ('f1 test: ', end='')
    print(f1_score(dataset.get_test_y(), clf.predict(dataset.get_test_x())))
    print('-------------------------------------------------')

    for mode in ['regular', 'borderline1', 'borderline2', 'svm']:
        print(mode + 'SMOTE:')
        train_x, train_y = dataset.get_resampled_train_X_y(kind=mode)
        clf = clf.fit(train_x, train_y)
        print('feature importances: ', end='')
        print(clf.feature_importances_)
        model = SelectFromModel(clf, prefit=True)
        print('model support vector: ', end='')
        print(model.get_support())
        selected_features = np.array(
            dataset.vectorizer.get_feature_names())[np.array(model.get_support())]

        print(len(selected_features))
        # print("----------------------------------------------")
        #
        # for feature in selected_features:
        #     print feature

        print('f1 train: ', end='')
        print(f1_score(dataset.get_train_y(), clf.predict(dataset.get_train_x())))
        print('f1 test: ', end='')
        print(f1_score(dataset.get_test_y(), clf.predict(dataset.get_test_x())))
        print('-------------------------------------------------')

    # k_best_features = SelectKBest(chi2, k=200)
    # k_best_features.fit(dataset.get_train_x(), dataset.get_train_y())
    # k_selected_features = np.array(dataset.vectorizer.get_feature_names())[np.array(k_best_features.get_support())]
    #
    # print(len(k_selected_features))
    # print("----------------------------------------------")
    #
    # for feature in k_selected_features:
    #     print feature
