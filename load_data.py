# -*- coding: utf-8 -*-

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



import os
import numpy as np

def load_data(name='freq', get_all_special_data=True):
    if name == 'freq':
        path = 'D:/Python/TAR/dataset/metadata/bags of words'
    elif name == 'chi2':
        path = 'D:/Python/TAR/chi2_scores/metadata/bags_of_words'
    elif name == 'tfidf':
        path = 'D:/Python/TAR/tf-idf_scores/metadata/bags_of_words'
    else:
        raise ValueError
        
    train_negative = path + '/negative'
    train_positive = path + '/positive'
    test_negative = path + '/negative_test'
    test_positive = path + '/positive_test'
    
    special_path = 'D:/Python/TAR/special-data/bags of words'
    special_train_negative = special_path + '/negative/'
    special_train_positive = special_path + '/positive/'
    special_test_negative = special_path + '/negative_test/'
    special_test_positive = special_path + '/positive_test/'
    
    #
    # load train data
    #
    train = []
    train_X = []
    train_S = []
    train_y = []
    os.chdir(train_negative)
    negative_files = os.listdir()
    #print('negative train files:', len(negative_files))
    for txtfile in negative_files:
        with open(txtfile, 'r', encoding='utf8') as file:
            vector = file.readlines()
            vector = [int(token[:-1]) for token in vector]  # remove '\n', convert values to int
            special_vector = []
            with open(special_train_negative + txtfile, 'r', encoding='utf-8') as sf:
                special_vector = sf.readlines()
                special_vector = [float(token[:-1]) for token in special_vector]
                if get_all_special_data == False:
                    special_vector = [special_vector[1], special_vector[4], special_vector[5], special_vector[8]]
            train.append([np.array(vector), np.array(special_vector), np.array([1, 0])])
    os.chdir(train_positive)
    positive_files = os.listdir()
    #print('positive train files:', len(positive_files))
    for txtfile in positive_files:
        with open(txtfile, 'r', encoding='utf8') as file:
            vector = file.readlines()
            vector = [int(token[:-1]) for token in vector]  # remove '\n', convert values to int
            special_vector = []
            with open(special_train_positive + txtfile, 'r', encoding='utf-8') as sf:
                special_vector = sf.readlines()
                special_vector = [float(token[:-1]) for token in special_vector]
                if get_all_special_data == False:
                    special_vector = [special_vector[1], special_vector[4], special_vector[5], special_vector[8]]
            train.append([np.array(vector), np.array(special_vector), np.array([0, 1])])
            train.append([np.array(vector), np.array(special_vector), np.array([0, 1])])
            train.append([np.array(vector), np.array(special_vector), np.array([0, 1])])
            train.append([np.array(vector), np.array(special_vector), np.array([0, 1])])
            train.append([np.array(vector), np.array(special_vector), np.array([0, 1])])
            
    train = np.array(train)
    #np.random.shuffle(train)
    # don't shuffle here, shuffle data controlably when necessary
    for sample in train:
        train_X.append(sample[0])
        train_S.append(sample[1])
        train_y.append(sample[2])

      
    #
    # load test data
    #
    test = []
    test_X = []
    test_S = []
    test_y = []
    os.chdir(test_negative)
    negative_files = os.listdir()
    #print('negative test files:', len(negative_files))
    for txtfile in negative_files:
        with open(txtfile, 'r', encoding='utf8') as file:
            vector = file.readlines()
            vector = [int(token[:-1]) for token in vector]  # remove '\n', convert values to int
            special_vector = []
            with open(special_test_negative + txtfile, 'r', encoding='utf-8') as sf:
                special_vector = sf.readlines()
                special_vector = [float(token[:-1]) for token in special_vector]
                if get_all_special_data == False:
                    special_vector = [special_vector[1], special_vector[4], special_vector[5], special_vector[8]]
            test.append([np.array(vector), np.array(special_vector), np.array([1, 0])])
    os.chdir(test_positive)
    positive_files = os.listdir()
    #print('positive test files:', len(positive_files))
    for txtfile in positive_files:
        with open(txtfile, 'r', encoding='utf8') as file:
            vector = file.readlines()
            vector = [int(token[:-1]) for token in vector]  # remove '\n', convert values to int
            special_vector = []
            with open(special_test_positive + txtfile, 'r', encoding='utf-8') as sf:
                special_vector = sf.readlines()
                special_vector = [float(token[:-1]) for token in special_vector]
                if get_all_special_data == False:
                    special_vector = [special_vector[1], special_vector[4], special_vector[5], special_vector[8]]
            test.append([np.array(vector), np.array(special_vector), np.array([0, 1])])
            
    test = np.array(test)
    #np.random.shuffle(test)
    for sample in test:
        test_X.append(sample[0])
        test_S.append(sample[1])
        test_y.append(sample[2])
    #print('len(test_y) =', len(test_y))
    return np.array(train_X), np.array(train_S), np.array(train_y), np.array(test_X), np.array(test_S), np.array(test_y)
    
        
def get_nn_predict_values():
    os.chdir('D:/Python/TAR/system')
    
    #
    # train data
    #
    freq_train = []
    with open('freq_train.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        vector = [float(x[:-1]) for x in lines]
    freq_train.append(np.array(vector))
    freq_train = np.array(freq_train)
    
    chi2_train = []
    with open('chi2_train.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        vector = [float(x[:-1]) for x in lines]
    chi2_train.append(np.array(vector))
    chi2_train = np.array(chi2_train)
    
    tfidf_train = []
    with open('tfidf_train.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        vector = [float(x[:-1]) for x in lines]
    tfidf_train.append(np.array(vector))
    tfidf_train = np.array(tfidf_train)
    
    
    #
    # test data
    #
    freq_test = []
    with open('freq_test.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        vector = [float(x[:-1]) for x in lines]
    freq_test.append(np.array(vector))
    freq_test = np.array(freq_test)
    
    chi2_test = []
    with open('chi2_test.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        vector = [float(x[:-1]) for x in lines]
    chi2_test.append(np.array(vector))
    chi2_test = np.array(chi2_test)
    
    tfidf_test = []
    with open('tfidf_test.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        vector = [float(x[:-1]) for x in lines]
    tfidf_test.append(np.array(vector))
    tfidf_test = np.array(tfidf_test)
    
    return freq_train, chi2_train, tfidf_train, freq_test, chi2_test, tfidf_test