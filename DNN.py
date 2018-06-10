# -*- coding: utf-8 -*-

import os
os.chdir('D:/Python/TAR/system')
import numpy as np
from load_data import load_data, get_nn_predict_values
#from nn_values import get_nn_predict_values
from sklearn.metrics import f1_score, precision_score, recall_score
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras import regularizers

def concatenate(m1, m2):    # expects train/test_S as m1 and freq_train/test  type as m2
    matrix = []
    for i in range(m1.shape[0]):   # rows
        row = []
        for j in range(m1.shape[1]):
            row.append(m1[i][j])
        row.append(m2[0][i])
        matrix.append(np.array(row))
    
    return np.array(matrix)

def concatenate2(m1, m2):
    matrix = []
    for i in range(m1.shape[0]):
        row = []
        for j in range(m1.shape[1]):
            row.append(m1[i][j])
        for j in range(m2.shape[1]):
            row.append(m2[i][j])
        matrix.append(np.array(row))
        
    return np.array(matrix)

def prep_data(train_S, test_S, freq_train, freq_test, chi2_train, chi2_test, tfidf_train, tfidf_test):
    nn1_train = concatenate(train_S, freq_train)
    nn2_train = concatenate(train_S, chi2_train)
    nn3_train = concatenate(train_S, tfidf_train)
    
    nn1_test = concatenate(test_S, freq_test)
    nn2_test = concatenate(test_S, chi2_test)
    nn3_test = concatenate(test_S, tfidf_test)
    
    return nn1_train, nn2_train, nn3_train, nn1_test, nn2_test, nn3_test

def shuffle_data(X, y):
    rng_state = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(rng_state)
    np.random.shuffle(y)
    
    return X, y

def run_model(filename, train_X, train_y, test_X, test_y):
    train_X, train_y = shuffle_data(train_X, train_y)
    
    
    
    from sklearn import preprocessing
    #train_X = preprocessing.scale(train_X)
    #test_X = preprocessing.scale(test_X)
    scaler = preprocessing.StandardScaler().fit(train_X)
    scaler.fit(train_X)
    scaler.transform(train_X)
    scaler.transform(test_X)
    
    from sklearn import svm
    model = svm.SVC()
    y = np.argmax(train_y, axis=1)
    model.fit(train_X, y)
    p = model.predict(test_X)
    print('svm f1 =', f1_score(np.argmax(test_y, axis=1), p)) # for comparison purposes
    
    model = Sequential()
    model.add(Dense(70, input_dim=204, activation='relu', kernel_regularizer=regularizers.l2(0.00)))	# change input dim as necessary, it is kept this way here to showcase the dimensionality of best presented model in the paper
    model.add(Dense(60, activation='relu', kernel_regularizer=regularizers.l2(0.00)))
    model.add(Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.00)))
    model.add(Dense(40, activation='relu', kernel_regularizer=regularizers.l2(0.00)))
    
    model.add(Dense(2, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.fit(x=train_X, y=train_y, batch_size=32, epochs=15, shuffle=True)
    
    os.chdir('D:/Python/TAR/system/models')
    model.save(filename + '.h5')	# manually move to folder based on neural network type produced
                                                 #
    
    p = model.predict(test_X)
    print('f1', filename, ':', f1_score(np.argmax(test_y, axis=1), np.argmax(p, axis=1)))

def accuracy_score(y_true, y_pred):
    correct_count = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct_count += 1
            
    accuracy = float(correct_count) / float(len(y_true))
    
    return accuracy

def get_p_value(y_true, p1, p2, n_r_others=10000):
    # init seed using local time in miliseconds
    import datetime
    np.random.seed(datetime.datetime.now().microsecond)
    '''
    calculate R_real and then an array of R_1, R_2, ..., R_n
    
    calculate a p value based on a placement of R_real in the sorted R_list
    
    '''
    
    R_real = get_R(y_true, p1, p2)
    R = get_R(y_true, p1, p2, real=False, repeats=n_r_others)
    print('R_real =', R_real)
    position = n_r_others    # init starting position of R_real in list to last place
    # find the real position of R_real in list of R
    for i in range(len(R) - 1):
        if R_real < R[0]:
            position = 0
            break
        
        if R_real >= R[i] and R_real <= R[i + 1]:
            position = i
            break
        
    print('position =', position)
    return abs(((position / n_r_others) * 2) - 1)
    
    
def get_R(y_true, p1, p2, real=True, repeats=10000):
    if real == True:        # get inital and real R value
        R = f1_score(y_true, p1) - f1_score(y_true, p2)
        return R
    
    # else, R_1, R_2, ..., R_n values by shuffling predictions
    # n is 10 000 by default
    # return sorted np.ndarray object
    R = []
    for i in range(repeats):
        change = np.random.randint(0, 2, size=len(p1))
        t_p1 = []
        t_p2 = []
        
        # exchange places in predictions with a chance of 50%
        for j in range(len(p1)):
            if change[j]:
                t_p1.append(p2[j])
                t_p2.append(p1[j])
            else:
                t_p1.append(p1[j])
                t_p2.append(p2[j])
                
        # calculate R score and append it to the list
        R.append(f1_score(y_true, t_p1) - f1_score(y_true, t_p2))
        
        # track percentage done
        #if i % 100 == 0:
        #    print('{}%'.format(i//100))
        
    #print('100% - done!')
    R = np.array(R)
    R = np.sort(R)
    
    print(R)
                
    return R
    
def print_p_values():
    '''
    calculates the p_value of various network combinations
    print everything to the screen for demonstration purposes
    '''
    
    print()
    print()
    
    # get y_true scores
    train_X, train_S, train_y, test_X, test_S, test_y = load_data()
    y_true = np.argmax(test_y, axis=1)
    
    # get ensamble predictions
    import pickle
    os.chdir('D:/Python/TAR/system')
    # predictions_ensamble is a numpy array of loaded ensamble predictions on
    # the unshufled test set, like all other predictions written bellow
    with open('predicted_by_ensamble.pkl', 'rb') as pickle_file:
        predictions_ensamble = pickle.load(pickle_file)
    #print(predictions_ensamble)
    
    
    # CHAPTER I
    # special NN vs bare NN improvement
    #
    # 1) get predictions on the test set
    # 1.1) get nn_bare predictions
    os.chdir('D:/Python/TAR/system/models/bare')
    nn_bare = load_model('model_chi2.h5')
    train_X, train_S, train_y, test_X, test_S, test_y = load_data('chi2', get_all_special_data=False)
    freq_train, chi2_train, tfidf_train, freq_test, chi2_test, tfidf_test = get_nn_predict_values()
    freq_train, chi2_train, tfidf_train, freq_test, chi2_test, tfidf_test = prep_data(
            train_S, test_S, freq_train, freq_test, chi2_train, chi2_test, tfidf_train, tfidf_test
    )
    predictions_bare = np.argmax(nn_bare.predict(test_X), axis=1)
    # 1.2) get nn_augmented predictions
    os.chdir('D:/Python/TAR/system/models/special')
    nn_augmented = load_model('special_chi2.h5')
    train_X, train_S, train_y, test_X, test_S, test_y = load_data('chi2', get_all_special_data=False)
    freq_train, chi2_train, tfidf_train, freq_test, chi2_test, tfidf_test = get_nn_predict_values()
    freq_train, chi2_train, tfidf_train, freq_test, chi2_test, tfidf_test = prep_data(
            train_S, test_S, freq_train, freq_test, chi2_train, chi2_test, tfidf_train, tfidf_test
    )
    predictions_augmented = np.argmax(nn_augmented.predict(concatenate2(test_X, test_S)), axis=1)
    p_value = get_p_value(y_true[-54:], predictions_augmented[-54:], predictions_bare[-54:], n_r_others=10000)
    #p_value = get_p_value(y_true[:-54], predictions_augmented[:-54], predictions_bare[:-54], n_r_others=10000)
    #p_value = get_p_value(y_true, predictions_augmented, predictions_bare, n_r_others=10000)
    
    #print(R)
    print('-------------')
    print('p_value =', p_value)
    print('-------------')
    
    p_value = get_p_value(y_true[-54:], predictions_augmented[-54:], predictions_ensamble[-54:], n_r_others=10000)
    #p_value = get_p_value(y_true[:-54], predictions_augmented[:-54], predictions_ensamble[:-54], n_r_others=10000)
    #p_value = get_p_value(y_true, predictions_augmented, predictions_ensamble, n_r_others=10000)
    
    print('-------------')
    print('p_value (ensamble)', p_value)
    print('-------------')
    
    
    print()
    print()
    from scipy import stats
    print('t-test', stats.ttest_ind(predictions_augmented[-54:], predictions_bare[-54:]))
    #print('t-test', stats.ttest_ind(predictions_augmented[:-54], predictions_bare[:-54]))
    
    
    
    
def final_testing():
    print()
    # load bare neural networks
    os.chdir('D:/Python/TAR/system/models/bare')
    freq_bare = load_model('model_freq.h5')
    chi2_bare = load_model('model_chi2.h5')
    tfidf_bare = load_model('model_tfidf.h5')
    
    # load augmented neural networks
    os.chdir('D:/Python/TAR/system/models/special')
    freq = load_model('special_freq.h5')
    chi2 = load_model('special_chi2.h5')
    tfidf = load_model('special_tfidf.h5')
    
    # -- EXTENSIVE MODEL TESTING --
    
    # CHAPTER I
    # Bare neural networks, train and test accuracy and f1 scores
    print('Bare neural networks, train and test accuracy and f1 scores')
    # 1) freq
    train_X, train_S, train_y, test_X, test_S, test_y = load_data('freq', get_all_special_data=False)
    freq_train, chi2_train, tfidf_train, freq_test, chi2_test, tfidf_test = get_nn_predict_values()
    freq_train, chi2_train, tfidf_train, freq_test, chi2_test, tfidf_test = prep_data(
            train_S, test_S, freq_train, freq_test, chi2_train, chi2_test, tfidf_train, tfidf_test
    )
    p_train = freq_bare.predict(train_X)
    p_test = freq_bare.predict(test_X)
    print('freq')
    print('\tacc: train {} test {}'.format(accuracy_score(np.argmax(train_y, axis=1), np.argmax(p_train, axis=1)),
                                           accuracy_score(np.argmax(test_y, axis=1), np.argmax(p_test, axis=1))))
    print('\tf1: train {} test {}'.format(f1_score(np.argmax(train_y, axis=1), np.argmax(p_train, axis=1)), 
                                          f1_score(np.argmax(test_y, axis=1), np.argmax(p_test, axis=1))))
    print('\tprecision: train {} test {}'.format(precision_score(np.argmax(train_y, axis=1), np.argmax(p_train, axis=1)), 
                                          precision_score(np.argmax(test_y, axis=1), np.argmax(p_test, axis=1))))
    print('\trecall: train {} test {}'.format(recall_score(np.argmax(train_y, axis=1), np.argmax(p_train, axis=1)), 
                                          recall_score(np.argmax(test_y, axis=1), np.argmax(p_test, axis=1))))
    # 2) chi2
    train_X, train_S, train_y, test_X, test_S, test_y = load_data('chi2', get_all_special_data=False)
    freq_train, chi2_train, tfidf_train, freq_test, chi2_test, tfidf_test = get_nn_predict_values()
    freq_train, chi2_train, tfidf_train, freq_test, chi2_test, tfidf_test = prep_data(
            train_S, test_S, freq_train, freq_test, chi2_train, chi2_test, tfidf_train, tfidf_test
    )
    p_train = chi2_bare.predict(train_X)
    p_test = chi2_bare.predict(test_X)
    print('chi2')
    print('\tacc: train {} test {}'.format(accuracy_score(np.argmax(train_y, axis=1), np.argmax(p_train, axis=1)),
                                           accuracy_score(np.argmax(test_y, axis=1), np.argmax(p_test, axis=1))))
    print('\tf1: train {} test {}'.format(f1_score(np.argmax(train_y, axis=1), np.argmax(p_train, axis=1)), 
                                          f1_score(np.argmax(test_y, axis=1), np.argmax(p_test, axis=1))))
    print('\tprecision: train {} test {}'.format(precision_score(np.argmax(train_y, axis=1), np.argmax(p_train, axis=1)), 
                                          precision_score(np.argmax(test_y, axis=1), np.argmax(p_test, axis=1))))
    print('\trecall: train {} test {}'.format(recall_score(np.argmax(train_y, axis=1), np.argmax(p_train, axis=1)), 
                                          recall_score(np.argmax(test_y, axis=1), np.argmax(p_test, axis=1))))
    # 3) tfidf
    train_X, train_S, train_y, test_X, test_S, test_y = load_data('chi2', get_all_special_data=False)
    freq_train, chi2_train, tfidf_train, freq_test, chi2_test, tfidf_test = get_nn_predict_values()
    freq_train, chi2_train, tfidf_train, freq_test, chi2_test, tfidf_test = prep_data(
            train_S, test_S, freq_train, freq_test, chi2_train, chi2_test, tfidf_train, tfidf_test
    )
    p_train = tfidf_bare.predict(train_X)
    p_test = tfidf_bare.predict(test_X)
    print('tfidf')
    print('\tacc: train {} test {}'.format(accuracy_score(np.argmax(train_y, axis=1), np.argmax(p_train, axis=1)),
                                           accuracy_score(np.argmax(test_y, axis=1), np.argmax(p_test, axis=1))))
    print('\tf1: train {} test {}'.format(f1_score(np.argmax(train_y, axis=1), np.argmax(p_train, axis=1)), 
                                          f1_score(np.argmax(test_y, axis=1), np.argmax(p_test, axis=1))))
    print('\tprecision: train {} test {}'.format(precision_score(np.argmax(train_y, axis=1), np.argmax(p_train, axis=1)), 
                                          precision_score(np.argmax(test_y, axis=1), np.argmax(p_test, axis=1))))
    print('\trecall: train {} test {}'.format(recall_score(np.argmax(train_y, axis=1), np.argmax(p_train, axis=1)), 
                                          recall_score(np.argmax(test_y, axis=1), np.argmax(p_test, axis=1))))
    
    
    print()
    print()
    
    # CHAPTER II
    # Augmented neural networks, train and test accuracy and f1 scores
    print('Augmented neural networks, train and test accuracy and f1 scores')
    # 1) freq - TOTAL, not all special data used    #CONFIRMED 0.595041322314
    train_X, train_S, train_y, test_X, test_S, test_y = load_data('freq', get_all_special_data=False)
    freq_train, chi2_train, tfidf_train, freq_test, chi2_test, tfidf_test = get_nn_predict_values()
    freq_train, chi2_train, tfidf_train, freq_test, chi2_test, tfidf_test = prep_data(
            train_S, test_S, freq_train, freq_test, chi2_train, chi2_test, tfidf_train, tfidf_test
    )
    p_train = freq.predict(concatenate2(train_X, train_S))
    p_test = freq.predict(concatenate2(test_X, test_S))
    print('freq')
    print('\tacc: train {} test {}'.format(accuracy_score(np.argmax(train_y, axis=1), np.argmax(p_train, axis=1)),
                                           accuracy_score(np.argmax(test_y, axis=1), np.argmax(p_test, axis=1))))
    print('\tf1: train {} test {}'.format(f1_score(np.argmax(train_y, axis=1), np.argmax(p_train, axis=1)), 
                                          f1_score(np.argmax(test_y, axis=1), np.argmax(p_test, axis=1))))
    print('\tprecision: train {} test {}'.format(precision_score(np.argmax(train_y, axis=1), np.argmax(p_train, axis=1)), 
                                          precision_score(np.argmax(test_y, axis=1), np.argmax(p_test, axis=1))))
    print('\trecall: train {} test {}'.format(recall_score(np.argmax(train_y, axis=1), np.argmax(p_train, axis=1)), 
                                          recall_score(np.argmax(test_y, axis=1), np.argmax(p_test, axis=1))))
    # 2) chi2 - TOTAL, not all special data used    #CONFIRMED 0.666666666667 
    train_X, train_S, train_y, test_X, test_S, test_y = load_data('chi2', get_all_special_data=False)
    freq_train, chi2_train, tfidf_train, freq_test, chi2_test, tfidf_test = get_nn_predict_values()
    freq_train, chi2_train, tfidf_train, freq_test, chi2_test, tfidf_test = prep_data(
            train_S, test_S, freq_train, freq_test, chi2_train, chi2_test, tfidf_train, tfidf_test
    )
    p_train = chi2.predict(concatenate2(train_X, train_S))
    p_test = chi2.predict(concatenate2(test_X, test_S))
    print('chi2')
    print('\tacc: train {} test {}'.format(accuracy_score(np.argmax(train_y, axis=1), np.argmax(p_train, axis=1)),
                                           accuracy_score(np.argmax(test_y, axis=1), np.argmax(p_test, axis=1))))
    print('\tf1: train {} test {}'.format(f1_score(np.argmax(train_y, axis=1), np.argmax(p_train, axis=1)), 
                                          f1_score(np.argmax(test_y, axis=1), np.argmax(p_test, axis=1))))
    print('\tprecision: train {} test {}'.format(precision_score(np.argmax(train_y, axis=1), np.argmax(p_train, axis=1)), 
                                          precision_score(np.argmax(test_y, axis=1), np.argmax(p_test, axis=1))))
    print('\trecall: train {} test {}'.format(recall_score(np.argmax(train_y, axis=1), np.argmax(p_train, axis=1)), 
                                          recall_score(np.argmax(test_y, axis=1), np.argmax(p_test, axis=1))))
    # 3) tfidf - TOTAL, all special data used       #CONFIRMED 0.637037037037
    train_X, train_S, train_y, test_X, test_S, test_y = load_data('tfidf', get_all_special_data=True)
    freq_train, chi2_train, tfidf_train, freq_test, chi2_test, tfidf_test = get_nn_predict_values()
    freq_train, chi2_train, tfidf_train, freq_test, chi2_test, tfidf_test = prep_data(
            train_S, test_S, freq_train, freq_test, chi2_train, chi2_test, tfidf_train, tfidf_test
    )
    p_train = tfidf.predict(concatenate2(train_X, train_S))
    p_test = tfidf.predict(concatenate2(test_X, test_S))
    print('tfidf')
    print('\tacc: train {} test {}'.format(accuracy_score(np.argmax(train_y, axis=1), np.argmax(p_train, axis=1)),
                                           accuracy_score(np.argmax(test_y, axis=1), np.argmax(p_test, axis=1))))
    print('\tf1: train {} test {}'.format(f1_score(np.argmax(train_y, axis=1), np.argmax(p_train, axis=1)), 
                                          f1_score(np.argmax(test_y, axis=1), np.argmax(p_test, axis=1))))
    print('\tprecision: train {} test {}'.format(precision_score(np.argmax(train_y, axis=1), np.argmax(p_train, axis=1)), 
                                          precision_score(np.argmax(test_y, axis=1), np.argmax(p_test, axis=1))))
    print('\trecall: train {} test {}'.format(recall_score(np.argmax(train_y, axis=1), np.argmax(p_train, axis=1)), 
                                          recall_score(np.argmax(test_y, axis=1), np.argmax(p_test, axis=1))))


    
    return

    #   
    # code that was first used for testing
    # redundant, if you want to run it uncomment it in main
    #



    p_freq_test = freq.predict(concatenate2(test_X, test_S))
    print('freq f1:', f1_score(np.argmax(test_y, axis=1), np.argmax(p_freq, axis=1)))
    
    
    # test models first
    # 1) freq - TOTAL, not all special data used    #CONFIRMED 0.595041322314
    train_X, train_S, train_y, test_X, test_S, test_y = load_data('freq', get_all_special_data=False)
    freq_train, chi2_train, tfidf_train, freq_test, chi2_test, tfidf_test = get_nn_predict_values()
    freq_train, chi2_train, tfidf_train, freq_test, chi2_test, tfidf_test = prep_data(
            train_S, test_S, freq_train, freq_test, chi2_train, chi2_test, tfidf_train, tfidf_test
    )
    p_freq = freq.predict(concatenate2(test_X, test_S))
    print('freq f1:', f1_score(np.argmax(test_y, axis=1), np.argmax(p_freq, axis=1)))
    # 2) chi2 - TOTAL, not all special data used    #CONFIRMED 0.666666666667
    train_X, train_S, train_y, test_X, test_S, test_y = load_data('chi2', get_all_special_data=False)
    freq_train, chi2_train, tfidf_train, freq_test, chi2_test, tfidf_test = get_nn_predict_values()
    freq_train, chi2_train, tfidf_train, freq_test, chi2_test, tfidf_test = prep_data(
            train_S, test_S, freq_train, freq_test, chi2_train, chi2_test, tfidf_train, tfidf_test
    )
    p_chi2 = chi2.predict(concatenate2(test_X, test_S))
    print('chi2 f1:', f1_score(np.argmax(test_y, axis=1), np.argmax(p_chi2, axis=1)))
    # 3) tfidf - TOTAL, all special data used       #CONFIRMED 0.637037037037
    train_X, train_S, train_y, test_X, test_S, test_y = load_data('tfidf', get_all_special_data=True)
    freq_train, chi2_train, tfidf_train, freq_test, chi2_test, tfidf_test = get_nn_predict_values()
    freq_train, chi2_train, tfidf_train, freq_test, chi2_test, tfidf_test = prep_data(
            train_S, test_S, freq_train, freq_test, chi2_train, chi2_test, tfidf_train, tfidf_test
    )
    p_tfidf = tfidf.predict(concatenate2(test_X, test_S))
    print('tfidf f1:', f1_score(np.argmax(test_y, axis=1), np.argmax(p_tfidf, axis=1)))
    
    # *4) tfidf svm - not total, not all special data used      #CONFIRMED 0.615384615385
    train_X, train_S, train_y, test_X, test_S, test_y = load_data('tfidf', get_all_special_data=False)
    freq_train, chi2_train, tfidf_train, freq_test, chi2_test, tfidf_test = get_nn_predict_values()
    freq_train, chi2_train, tfidf_train, freq_test, chi2_test, tfidf_test = prep_data(
            train_S, test_S, freq_train, freq_test, chi2_train, chi2_test, tfidf_train, tfidf_test
    )
    from sklearn import preprocessing
    train_X = tfidf_train
    test_X = tfidf_test
    scaler = preprocessing.StandardScaler().fit(train_X)
    scaler.fit(train_X)
    scaler.transform(train_X)
    scaler.transform(test_X)
    from sklearn import svm
    model = svm.SVC()
    y = np.argmax(train_y, axis=1)
    model.fit(train_X, y)
    p_svm = model.predict(test_X)
    print('svm f1 =', f1_score(np.argmax(test_y, axis=1), p_svm))
    
    
    
    print()
    
    f, c, t = 1, 1, 1
    # test ensamble performance
    # 1) freq - chi2 - tfidf
    p = f*p_freq + c*p_chi2 + t*p_tfidf
    print('ensamble(1,2,3) f1:', f1_score(np.argmax(test_y, axis=1), np.argmax(p, axis=1)))
    # 2) freq - chi2
    p = f*p_freq + c*p_chi2
    print('ensamble(1,2) f1:', f1_score(np.argmax(test_y, axis=1), np.argmax(p, axis=1)))
    # 3) freq - tfidf
    p = f*p_freq + t*p_tfidf
    print('ensamble(1,3) f1:', f1_score(np.argmax(test_y, axis=1), np.argmax(p, axis=1)))
    # 4) chi2 - tfidf
    p = c*p_chi2 + t*p_tfidf
    print('ensamble(2,3) f1:', f1_score(np.argmax(test_y, axis=1), np.argmax(p, axis=1)))
    
    pf = np.argmax(p_freq, axis=1)
    pc = np.argmax(p_chi2, axis=1)
    pt = np.argmax(p_tfidf, axis=1)
    ps = p_svm
    y = np.argmax(test_y, axis=1)
    
    e1 = []     # freq, chi2, tfidf
    for i in range(len(y)):
        c0 = 0
        c1 = 0
        if pf[i] == 0:
            c0 += 1
        else:
            c1 += 1
        if pc[i] == 0:
            c0 += 1
        else:
            c1 += 1
        if pt[i] == 0:
            c0 += 1
        else:
            c1 += 1
            
        if c0 > c1:
            e1.append(0)
        else:
            e1.append(1)
            
    print()
    print(f1_score(y, e1))
    
    e2 = []     # svm, chi2, tfidf
    for i in range(len(y)):
        c0 = 0
        c1 = 0
        if ps[i] == 0:
            c0 += 1
        else:
            c1 += 1
        if pc[i] == 0:
            c0 += 1
        else:
            c1 += 1
        if pt[i] == 0:
            c0 += 1
        else:
            c1 += 1
            
        if c0 > c1:
            e2.append(0)
        else:
            e2.append(1)
            
    print()
    print(f1_score(y, e2))

if __name__ == '__main__':
    train_X, train_S, train_y, test_X, test_S, test_y = load_data('tfidf')
    freq_train, chi2_train, tfidf_train, freq_test, chi2_test, tfidf_test = get_nn_predict_values()
    
    print('Train Shapes: X, S, y')
    print(train_X.shape)
    print(train_S.shape)
    print(train_y.shape)
    
    print()
    print('Test Shapes: X, S, y')
    print(test_X.shape)
    print(test_S.shape)
    print(test_y.shape)
    
    print()
    print('NN output shapes: train, test')
    print('freq', freq_train.shape, freq_test.shape)
    print('chi2', chi2_train.shape, chi2_test.shape)
    print('tfidf', tfidf_train.shape, tfidf_test.shape)
    
    freq_train, chi2_train, tfidf_train, freq_test, chi2_test, tfidf_test = prep_data(
            train_S, test_S, freq_train, freq_test, chi2_train, chi2_test, tfidf_train, tfidf_test
    )
    
    print()
    print('Special_nn input shapes: train, test')
    print('freq', freq_train.shape, freq_test.shape)
    print('chi2', chi2_train.shape, chi2_test.shape)
    print('tfidf', tfidf_train.shape, tfidf_test.shape)
    print()
    
    
    # some examples of models tried:
    # model_xxx - bare model, input is BoW vector
    # special_xxx - model that uses BoW and special feature vector as input
    # cascade_xxx - model that uses output of a bare model concatenated with special feature vector
    #
    #run_model('model_freq', freq_train, train_y, freq_test, test_y)
    #run_model('special_chi2', concatenate2(train_X, train_S), train_y, concatenate2(test_X, test_S), test_y)
    #run_model('cascade_chi2', chi2_train, train_y, chi2_test, test_y)
    #run_model('special_tfidf', concatenate2(train_X, train_S), train_y, concatenate2(test_X, test_S), test_y)
    #run_model('cascade_tfidf', tfidf_train, train_y, tfidf_test, test_y)
    
    '''
    #
    # initial testing of all special models
    # extensive testing is done in final_testing method which also include bare models
    #
    os.chdir('D:/Python/TAR/system/models/special')
    freq = load_model('special_freq.h5')    #TOTAL_5=0.595 r300r200r100
    chi2 = load_model('special_chi2.h5')    #TOTAL_0=0.667 r70r60r50r40
    tfidf = load_model('special_tfidf.h5')  #TOTAL_0=0.637 r300r200r100 (all s data)
    
    # test models first
    # 1) freq - TOTAL, not all special data used    #CONFIRMED 0.595041322314
    train_X, train_S, train_y, test_X, test_S, test_y = load_data('freq', get_all_special_data=False)
    freq_train, chi2_train, tfidf_train, freq_test, chi2_test, tfidf_test = get_nn_predict_values()
    freq_train, chi2_train, tfidf_train, freq_test, chi2_test, tfidf_test = prep_data(
            train_S, test_S, freq_train, freq_test, chi2_train, chi2_test, tfidf_train, tfidf_test
    )
    p_freq = freq.predict(concatenate2(test_X, test_S))
    print('freq f1:', f1_score(np.argmax(test_y, axis=1), np.argmax(p_freq, axis=1)))
    # 2) chi2 - TOTAL, not all special data used    #CONFIRMED 0.666666666667
    train_X, train_S, train_y, test_X, test_S, test_y = load_data('chi2', get_all_special_data=False)
    freq_train, chi2_train, tfidf_train, freq_test, chi2_test, tfidf_test = get_nn_predict_values()
    freq_train, chi2_train, tfidf_train, freq_test, chi2_test, tfidf_test = prep_data(
            train_S, test_S, freq_train, freq_test, chi2_train, chi2_test, tfidf_train, tfidf_test
    )
    p_chi2 = chi2.predict(concatenate2(test_X, test_S))
    print('chi2 f1:', f1_score(np.argmax(test_y, axis=1), np.argmax(p_chi2, axis=1)))
    # 3) tfidf - TOTAL, all special data used       #CONFIRMED 0.637037037037
    train_X, train_S, train_y, test_X, test_S, test_y = load_data('tfidf', get_all_special_data=True)
    freq_train, chi2_train, tfidf_train, freq_test, chi2_test, tfidf_test = get_nn_predict_values()
    freq_train, chi2_train, tfidf_train, freq_test, chi2_test, tfidf_test = prep_data(
            train_S, test_S, freq_train, freq_test, chi2_train, chi2_test, tfidf_train, tfidf_test
    )
    p_tfidf = tfidf.predict(concatenate2(test_X, test_S))
    print('tfidf f1:', f1_score(np.argmax(test_y, axis=1), np.argmax(p_tfidf, axis=1)))
    
    # *4) tfidf svm - not total, not all special data used      #CONFIRMED 0.615384615385
    train_X, train_S, train_y, test_X, test_S, test_y = load_data('tfidf', get_all_special_data=False)
    freq_train, chi2_train, tfidf_train, freq_test, chi2_test, tfidf_test = get_nn_predict_values()
    freq_train, chi2_train, tfidf_train, freq_test, chi2_test, tfidf_test = prep_data(
            train_S, test_S, freq_train, freq_test, chi2_train, chi2_test, tfidf_train, tfidf_test
    )
    from sklearn import preprocessing
    train_X = tfidf_train
    test_X = tfidf_test
    scaler = preprocessing.StandardScaler().fit(train_X)
    scaler.fit(train_X)
    scaler.transform(train_X)
    scaler.transform(test_X)
    from sklearn import svm
    model = svm.SVC()
    y = np.argmax(train_y, axis=1)
    model.fit(train_X, y)
    p_svm = model.predict(test_X)
    print('svm f1 =', f1_score(np.argmax(test_y, axis=1), p_svm))
    
    
    
    print()
    
    # some ensamble testing based on argmax-ing sum of softmax output of various
    # combinations of neural networks developed for this scientific paper
    # ensamble testing is also done with majority-wins ensambles using neural 
    # networks and svm trained on chi^2 extracted dataset
    
    f, c, t = 1, 1, 1
    # test ensamble performance
    # 1) freq - chi2 - tfidf
    p = f*p_freq + c*p_chi2 + t*p_tfidf
    print('ensamble(1,2,3) f1:', f1_score(np.argmax(test_y, axis=1), np.argmax(p, axis=1)))
    # 2) freq - chi2
    p = f*p_freq + c*p_chi2
    print('ensamble(1,2) f1:', f1_score(np.argmax(test_y, axis=1), np.argmax(p, axis=1)))
    # 3) freq - tfidf
    p = f*p_freq + t*p_tfidf
    print('ensamble(1,3) f1:', f1_score(np.argmax(test_y, axis=1), np.argmax(p, axis=1)))
    # 4) chi2 - tfidf
    p = c*p_chi2 + t*p_tfidf
    print('ensamble(2,3) f1:', f1_score(np.argmax(test_y, axis=1), np.argmax(p, axis=1)))
    
    pf = np.argmax(p_freq, axis=1)
    pc = np.argmax(p_chi2, axis=1)
    pt = np.argmax(p_tfidf, axis=1)
    ps = p_svm
    y = np.argmax(test_y, axis=1)
    
    e1 = []     # freq, chi2, tfidf
    for i in range(len(y)):
        c0 = 0
        c1 = 0
        if pf[i] == 0:
            c0 += 1
        else:
            c1 += 1
        if pc[i] == 0:
            c0 += 1
        else:
            c1 += 1
        if pt[i] == 0:
            c0 += 1
        else:
            c1 += 1
            
        if c0 > c1:
            e1.append(0)
        else:
            e1.append(1)
            
    print()
    print(f1_score(y, e1))
    
    e2 = []     # svm, chi2, tfidf
    for i in range(len(y)):
        c0 = 0
        c1 = 0
        if ps[i] == 0:
            c0 += 1
        else:
            c1 += 1
        if pc[i] == 0:
            c0 += 1
        else:
            c1 += 1
        if pt[i] == 0:
            c0 += 1
        else:
            c1 += 1
            
        if c0 > c1:
            e2.append(0)
        else:
            e2.append(1)
            
    print()
    print(f1_score(y, e2))
    '''
    
    final_testing()
    
    print_p_values()
    
    
    