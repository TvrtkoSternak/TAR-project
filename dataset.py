import os
import sys
import pickle
from xml.dom import minidom
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN

NEGATIVE_EXAMPLES_CLASS = 0
POSITIVE_EXAMPLES_CLASS = 1
TEXT_XML_TAG = 'TEXT'
TRAIN_LIST = 'train_examples.pkl'
TEST_LIST = 'test_examples.pkl'
TRAIN_BOW = 'train_bow.pkl'
TEST_BOW = 'test_pow.pkl'

class Dataset:
    train = list()
    test = list()
    corpus = set()
    lemmatizer = WordNetLemmatizer()
    vectorizer = CountVectorizer(max_features=5000)

    def __init__(self, corpus = None):
        print("Loading dataset")

        if os.path.isfile(TRAIN_LIST):
            print('Loading train from pickled file')
            self.__load_train()
        else:
            self.__first_pass_train()
        if os.path.isfile(TEST_LIST):
            print('Loading test from pickled file')
            self.__load_test()
        else:
            self.__first_pass_test()

        if corpus is not None:
            self.corpus = corpus
        else:
            self.__build_corpus()

        self.__fit_vectorizer()

        print("Dataset loaded\nStats:\nnum positive train: {}\nnum negative train: {}\n"
              "num train: {}\nnum positive test: {}\nnum negative test: {}\n"
              "num test: {}\nnum total words in corpus: {}"
              .format(*self.get_stats()))  # printing some statistics so it's nice looking

    def get_train_x(self, number_of_posts=None, return_bow=True):
        return self.__parse_set(self.train, number_of_posts, return_bow)

    def get_train_y(self):
        return np.array([user[1] for user in self.train])

    def get_resampled_train_X_y(self, number_of_posts=None, return_bow=True, kind='regular'):
        X = self.get_train_x(number_of_posts, return_bow)
        y = self.get_train_y()
        X_resampled, y_resampled = SMOTE(kind=kind).fit_sample(X, y)
        return X_resampled, y_resampled

    def get_test_x(self, number_of_posts=None, return_bow=True):
        return self.__parse_set(self.test, number_of_posts, return_bow)

    def get_test_y(self):
        return np.array([user[1] for user in self.test])

    ## Returns number of positive, negative and total sum of train and test
    ## Returns number of words and number of distinct words in corpus

    def get_stats(self):
        num_positive_train = sum(example_class == POSITIVE_EXAMPLES_CLASS
                                for example, example_class in self.train)
        num_negative_train = sum(example_class == NEGATIVE_EXAMPLES_CLASS
                                for example, example_class in self.train)
        num_train = len(self.train)
        num_positive_test = sum(example_class == POSITIVE_EXAMPLES_CLASS
                                for example, example_class in self.test)
        num_negative_test = sum(example_class == NEGATIVE_EXAMPLES_CLASS
                                for example, example_class in self.test)
        num_test = len(self.test)
        num_words = len(self.corpus)

        return num_positive_train, num_negative_train, num_train, num_positive_test, num_negative_test, num_test, num_words

    def __get_text(self, xml_file_to_parse):
        file_to_parse = minidom.parse(xml_file_to_parse)
        all_tags = file_to_parse.getElementsByTagName(TEXT_XML_TAG)
        list_of_posts = list()
        for s in all_tags:
            lemmatized = self.__lemmatize(s.firstChild.nodeValue)
            if lemmatized != '':
                list_of_posts.append(self.__lemmatize(s.firstChild.nodeValue))
        return list_of_posts

    def __build_corpus(self):
        for user_posts, _ in self.train:
            for post in user_posts:
                for word in post.split():
                    self.corpus.add(word)

        for user_posts, _ in self.test:
            for post in user_posts:
                for word in post.split():
                    self.corpus.add(word)


    def __lemmatize(self, sentence_to_lemmatize):
        one_line = sentence_to_lemmatize.lower()
        tokens = word_tokenize(one_line)
        tokens_pos = pos_tag(tokens)
        lematized_one_line = ""
        for token, pos_t in tokens_pos:
            lematized_one_line += self.lemmatizer.lemmatize(token, pos=self.__get_wordnet_pos(pos_t))+" "
        return lematized_one_line

    def __parse_set(self, set, number_of_posts=None, return_bow = True):
        users_posts_representation = []
        if number_of_posts is not None:
            pass
        else:
            if return_bow:

                for user_posts, _ in set:
                    all_posts = list()
                    for post in user_posts:
                        for word in post.split():
                            all_posts.append(word)
                            all_posts.append(" ")
                    users_posts_representation += [''.join(all_posts)]

            else:
                pass
        return self.__get_bag_of_words(users_posts_representation)

    def __get_bag_of_words(self, text):
        return self.vectorizer.transform(text)

    def __load_train(self):
        filehandler = open(TRAIN_LIST, 'r')
        self.train = pickle.load(filehandler)
        print('Done loading train')

    def __load_test(self):
        filehandler = open(TEST_LIST, 'r')
        self.test = pickle.load(filehandler)
        print('Done loading test')

    def __first_pass_train(self):
        for filename in os.listdir('Dataset/negative_examples_train'):
            self.train.append((
                self.__get_text('Dataset/negative_examples_train/'+filename),
                NEGATIVE_EXAMPLES_CLASS))

        print("Done loading negative train")

        for filename in os.listdir('Dataset/positive_examples_train'):
            self.train.append((
                self.__get_text('Dataset/positive_examples_train/'+filename),
                POSITIVE_EXAMPLES_CLASS))
        print("Done loading positive train")

        filehandler = open(TRAIN_LIST, 'w')
        pickle.dump(self.train, filehandler, pickle.HIGHEST_PROTOCOL)


    def __first_pass_test(self):
        for filename in os.listdir('Dataset/negative_examples_test'):
            self.test.append((
                self.__get_text('Dataset/negative_examples_test/'+filename),
                NEGATIVE_EXAMPLES_CLASS))
        print("Done loading negative test")

        for filename in os.listdir('Dataset/positive_examples_test'):
            self.test.append((
                self.__get_text('Dataset/positive_examples_test/'+filename),
                POSITIVE_EXAMPLES_CLASS))
        print("Done loading positive test")

        filehandler = open(TEST_LIST, 'w')
        pickle.dump(self.test, filehandler, pickle.HIGHEST_PROTOCOL)

    def __fit_vectorizer(self):
        self.vectorizer.fit(self.corpus)


    def __get_wordnet_pos(self, treebank_tag):

        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return 'n'


if __name__ == "__main__":
    dataset = Dataset()
    train_x = dataset.get_train_x()
    print(train_x.shape)
