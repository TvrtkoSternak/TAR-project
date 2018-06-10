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
import pickle
import dataset_adapter
import features_evaluation
import numpy as np

from keras import Sequential
from keras.layers import Dense, Dropout, ActivityRegularization, Flatten, InputLayer


def build_model(spec, X_train):
    model = Sequential()
    # create first layer
    layer = spec[0]
    num_posts, bow_dim = X_train[0].shape
    model.add(InputLayer(input_shape=(num_posts, bow_dim)))
    model.add(Flatten())
    if 'none' in layer:
        model.add(Dense(  # input_dim=1,
            units=int(layer.split('none')[1]),
            activation=None
        ))
    elif 'relu' in layer:
        model.add(Dense(  # input_shape=train_X[0].shape,
            units=int(layer.split('relu')[1]),
            activation='relu'
        ))
    elif 'sig' in layer:
        model.add(Dense(  # input_shape=train_X[0].shape,
            units=int(layer.split('sig')[1]),
            activation='sigmoid'
        ))
    else:
        return None

    for layer in spec[1:]:
        if 'none' in layer:
            model.add(Dense(int(layer.split('none')[1]), activation=None))
        elif 'relu' in layer:
            model.add(Dense(int(layer.split('relu')[1]), activation='relu'))
        elif 'sig' in layer:
            model.add(Dense(int(layer.split('sig')[1]), activation='sigmoid'))
        elif 'drop' in layer:
            model.add(Dropout(float(layer.split('drop')[1]), seed=None))
        elif 'l1' in layer:
            model.add(ActivityRegularization(l1=float(layer.split('l1')[1])))
        elif 'l2' in layer:
            model.add(ActivityRegularization(l2=float(layer.split('l2')[1])))
        else:
            return None

    # add softmax layer
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE


def run_save_model(save_folder, spec, model_no, X_train, y_train, model_fn):
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=2)
    cvscores = []
    f1scores = []
    for train, val in kfold.split(X_train, y_train):
        # create model using the model_fn parameter
        model = model_fn(spec, X_train)
        if model == None:
            return  # returns if there was a mistake in specifications
        # fit model to k-split of training data
        num_examples, dx, dy = X_train[train].shape
        X_resampled, y_resampled = SMOTE(kind='borderline1', random_state=1).fit_sample(
            X_train[train].reshape((num_examples, dx * dy)), y_train[train])
        num_total_examples, _ = X_resampled.shape
        X_resampled_reshaped = X_resampled.reshape(num_total_examples, dx, dy)
        model.fit(x=X_resampled_reshaped, y=y_resampled, epochs=10, batch_size=16, verbose=0)
        # evaluate model
        scores = model.evaluate(X_train[val], y_train[val], verbose=0)
        print('Accuracy: {}%'.format(scores[1] * 100))
        cvscores.append(scores[1])
        # get f1
        f1 = f1_score(y_train[val], model.predict(X_train[val]) > 0.5)
        print('F1 score: {}'.format(f1))
        f1scores.append(f1)

    mean_acc = 'Mean Accuracy: {}% +/- {}%'.format(np.mean(cvscores) * 100, np.std(cvscores) * 100)
    mean_f1 = 'Mean F1 score: {} +/- {}'.format(np.mean(f1scores), np.std(f1scores))
    print(mean_acc)
    print(mean_f1)

    # modelfile = save_folder + 'model' + str(model_no) + '.h5'
    # save_model(model, modelfile)
    # print('model saved')

    txtfile = save_folder + 'model' + str(model_no) + '.txt'
    with open(txtfile, 'w') as f:
        f.write(mean_acc)
        f.write(mean_f1)
        f.write('\n')
        f.write('\n')
        f.writelines(spec)
        print('specs saved')


if __name__ == "__main__":
    filehandler = open(features_evaluation.SELECTED_FEATURES_CORPUS_CHI2, 'r')
    corpus = pickle.load(filehandler)
    filehandler.close()
    dataset = dataset_adapter.DatasetAdapter(corpus=corpus)
    X_train = dataset.get_train_x(None, get_bow=True)
    y_train = dataset.get_train_y()

    save_folder = 'bow_models/chi2/'  # DIFFERENT FOR EVERY SELECTED FEATURES CORPUS

    specs = []
    with open('bow_models/anns.txt', 'r') as f:
        specs = f.readlines()
        specs = [x[:-1] for x in specs]  # to remove \n from every line

    model_no = 0
    for spec in specs:
        run_save_model(save_folder, spec.split(' '), model_no, X_train, y_train, model_fn=build_model)
        model_no += 1
        K.clear_session()

    filehandler = open(features_evaluation.SELECTED_FEATURES_CORPUS_FREQUENCY_INTERVAL, 'r')
    corpus = pickle.load(filehandler)
    filehandler.close()
    dataset = dataset_adapter.DatasetAdapter(corpus=corpus)
    X_train = dataset.get_train_x(None, get_bow=True)
    y_train = dataset.get_train_y()

    save_folder = 'bow_models/freq/'  # DIFFERENT FOR EVERY SELECTED FEATURES CORPUS

    specs = []
    with open('bow_models/anns.txt', 'r') as f:
        specs = f.readlines()
        specs = [x[:-1] for x in specs]  # to remove \n from every line

    model_no = 0
    for spec in specs:
        run_save_model(save_folder, spec.split(' '), model_no, X_train, y_train, model_fn=build_model)
        model_no += 1
        K.clear_session()


    filehandler = open(features_evaluation.SELECTED_FEATURES_CORPUS_TFIDF, 'r')
    corpus = pickle.load(filehandler)
    filehandler.close()
    dataset = dataset_adapter.DatasetAdapter(corpus=corpus)
    X_train = dataset.get_train_x(None, get_bow=True)
    y_train = dataset.get_train_y()

    save_folder = 'bow_models/tfidf/'  # DIFFERENT FOR EVERY SELECTED FEATURES CORPUS

    specs = []
    with open('bow_models/anns.txt', 'r') as f:
        specs = f.readlines()
        specs = [x[:-1] for x in specs]  # to remove \n from every line

    model_no = 0
    for spec in specs:
        run_save_model(save_folder, spec.split(' '), model_no, X_train, y_train, model_fn=build_model)
        model_no += 1
        K.clear_session()
