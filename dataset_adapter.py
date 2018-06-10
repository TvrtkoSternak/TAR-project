import os
import pickle
from xml.dom import minidom
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
import numpy as np

NEGATIVE_EXAMPLES_CLASS = 0
POSITIVE_EXAMPLES_CLASS = 1
TEXT_XML_TAG = 'TEXT'
TRAIN_LEMMATIZED_LIST = 'train_examples_lemmatized.pkl'
TEST_LEMMATIZED_LIST = 'test_examples_lemmatized.pkl'
TRAIN_POSTS_LIST = 'train_posts.pkl'
TEST_POSTS_LIST = 'test_posts.pkl'


class DatasetAdapter:

    train_lemmatized = list()
    test_lemmatized = list()
    train_posts = list()
    test_posts = list()
    corpus = set()
    lemmatizer = WordNetLemmatizer()
    train_x = None
    test_x = None
    vectorizer = CountVectorizer(max_features=5000)

    def __init__(self, corpus=None):
        print("Loading dataset")

        if os.path.isfile(TRAIN_LEMMATIZED_LIST):
            print('Loading train from pickled file')
            self.__load_train()
        else:
            self.__first_pass_train()
        if os.path.isfile(TEST_LEMMATIZED_LIST):
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
              .format(*self.get_stats()))

    def get_stats(self):
        num_positive_train = sum(example_class == POSITIVE_EXAMPLES_CLASS
                                for example, example_class in self.train_lemmatized)
        num_negative_train = sum(example_class == NEGATIVE_EXAMPLES_CLASS
                                for example, example_class in self.train_lemmatized)
        num_train = len(self.train_lemmatized)
        num_positive_test = sum(example_class == POSITIVE_EXAMPLES_CLASS
                                for example, example_class in self.test_lemmatized)
        num_negative_test = sum(example_class == NEGATIVE_EXAMPLES_CLASS
                                for example, example_class in self.test_lemmatized)
        num_test = len(self.test_lemmatized)
        num_words = len(self.corpus)

        return num_positive_train, num_negative_train, num_train,\
               num_positive_test, num_negative_test, num_test, num_words

    def get_train_x(self, num_posts=None, get_bow=False, special_chars_bow=None):
        if get_bow:
            train_x_tmp = self.__get_posts(self.train_lemmatized, num_posts=num_posts)
            train_x = self.__get_bag_of_words(train_x_tmp)
        else:
            train_x_tmp = self.__get_posts(self.train_posts, num_posts=num_posts)
            if special_chars_bow is not None:
                train_x = self.__get_bag_of_words(train_x_tmp, special_chars_bow=special_chars_bow)
            else:
                train_x = train_x_tmp
        return np.array(train_x)

    def get_test_x(self, num_posts=None, get_bow=False, special_chars_bow=None):
        if get_bow:
            test_x_tmp = self.__get_posts(self.test_lemmatized, num_posts=num_posts)
            test_x = self.__get_bag_of_words(test_x_tmp)
        else:
            test_x_tmp = self.__get_posts(self.test_posts, num_posts=num_posts)
            if special_chars_bow is not None:
                test_x = self.__get_bag_of_words(test_x_tmp, special_chars_bow=special_chars_bow)
            else:
                test_x = test_x_tmp
        return np.array(test_x)

    def get_train_y(self):
        return np.array([user[1] for user in self.train_lemmatized])

    def get_test_y(self):
        return np.array([user[1] for user in self.test_lemmatized])

    def __get_posts(self, set, num_posts=None):
        if num_posts is None:
            users_posts = list()
            for user, _ in set:
                user_posts = list()
                for post in user:
                    user_posts.append(post)
                    user_posts.append('\n')
                user_post_as_list = list()
                user_post_as_list.append(''.join(user_posts).encode('utf-8'))
                users_posts.append(user_post_as_list)
        else:
            users_posts = list()
            for user, _ in set:
                user_posts = list()
                for i in range(0, num_posts):
                    if i >= len(user_posts):
                        user_posts.append('')
                    else:
                        user_posts.append(user[i])
                users_posts.append(user_posts)
        return users_posts

    def __build_corpus(self):
        for user_posts, _ in self.train_lemmatized:
            for post in user_posts:
                for word in post.split():
                    self.corpus.add(word)

        for user_posts, _ in self.test_lemmatized:
            for post in user_posts:
                for word in post.split():
                    self.corpus.add(word)

    def do_nothing(self, tokens):
        return tokens

    def __get_bag_of_words(self, set, special_chars_bow=None):
        list_of_bows = list()
        if special_chars_bow is None:
            for user_posts in set:
                list_of_bows.append(self.vectorizer.transform(np.array(user_posts)).toarray())
        else:
            special_vectoriser = DictVectorizer()
            special_vectoriser.fit(Counter(s.split()) for s in special_chars_bow)
            print(special_vectoriser.vocabulary_)
            for user_posts in set:
                list_of_bows.append(
                    special_vectoriser.transform(Counter(s.split()) for s in np.array(user_posts)).toarray())
        return list_of_bows

    def __fit_vectorizer(self):
        self.vectorizer.fit(self.corpus)

    def __load_train(self):
        filehandler = open(TRAIN_POSTS_LIST, 'r')
        self.train_posts = pickle.load(filehandler)

        filehandler = open(TRAIN_LEMMATIZED_LIST, 'r')
        self.train_lemmatized = pickle.load(filehandler)

        print('Done loading train')

    def __load_test(self):
        filehandler = open(TEST_POSTS_LIST, 'r')
        self.test_posts = pickle.load(filehandler)

        filehandler = open(TEST_LEMMATIZED_LIST, 'r')
        self.test_lemmatized = pickle.load(filehandler)

        print('Done loading test')

    def __first_pass_train(self):
        for filename in os.listdir('Dataset/negative_examples_train'):
            self.train_lemmatized.append((
                self.__get_text('Dataset/negative_examples_train/'+filename, lemmatize=True),
                NEGATIVE_EXAMPLES_CLASS))
            self.train_posts.append((
                self.__get_text('Dataset/negative_examples_train/' + filename),
                NEGATIVE_EXAMPLES_CLASS))

        print("Done loading negative train")

        for filename in os.listdir('Dataset/positive_examples_train'):
            self.train_lemmatized.append((
                self.__get_text('Dataset/positive_examples_train/'+filename, lemmatize=True),
                POSITIVE_EXAMPLES_CLASS))
            self.train_posts.append((
                self.__get_text('Dataset/positive_examples_train/' + filename),
                POSITIVE_EXAMPLES_CLASS))
        print("Done loading positive train")

        filehandler = open(TRAIN_LEMMATIZED_LIST, 'w')
        pickle.dump(self.train_lemmatized, filehandler, pickle.HIGHEST_PROTOCOL)

        filehandler = open(TRAIN_POSTS_LIST, 'w')
        pickle.dump(self.train_posts, filehandler, pickle.HIGHEST_PROTOCOL)

    def __first_pass_test(self):
        for filename in os.listdir('Dataset/negative_examples_test'):
            self.test_lemmatized.append((
                self.__get_text('Dataset/negative_examples_test/'+filename, lemmatize=True),
                NEGATIVE_EXAMPLES_CLASS))
            self.test_posts.append((
                self.__get_text('Dataset/negative_examples_test/' + filename),
                NEGATIVE_EXAMPLES_CLASS))
        print("Done loading negative test")

        for filename in os.listdir('Dataset/positive_examples_test'):
            self.test_lemmatized.append((
                self.__get_text('Dataset/positive_examples_test/'+filename, lemmatize=True),
                POSITIVE_EXAMPLES_CLASS))
            self.test_posts.append((
                self.__get_text('Dataset/positive_examples_test/' + filename),
                POSITIVE_EXAMPLES_CLASS))
        print("Done loading positive test")

        filehandler = open(TEST_LEMMATIZED_LIST, 'w')
        pickle.dump(self.test_lemmatized, filehandler, pickle.HIGHEST_PROTOCOL)

        filehandler = open(TEST_POSTS_LIST, 'w')
        pickle.dump(self.test_posts, filehandler, pickle.HIGHEST_PROTOCOL)

    def __get_text(self, xml_file_to_parse, lemmatize=False):
        file_to_parse = minidom.parse(xml_file_to_parse)
        all_tags = file_to_parse.getElementsByTagName(TEXT_XML_TAG)
        list_of_posts = list()
        for s in all_tags:
            if lemmatize:
                post = self.__lemmatize(s.firstChild.nodeValue)
            else:
                post = s.firstChild.nodeValue
            if post != '':
                list_of_posts.append(post)
        return list_of_posts

    def __lemmatize(self, sentence_to_lemmatize):
        one_line = sentence_to_lemmatize.lower()
        tokens = word_tokenize(one_line)
        tokens_pos = pos_tag(tokens)
        lematized_one_line = ""
        for token, pos_t in tokens_pos:
            lematized_one_line += self.lemmatizer.lemmatize(token, pos=self.__get_wordnet_pos(pos_t))+" "
        return lematized_one_line

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
    dataset = DatasetAdapter()
    # train_x = dataset.get_train_x(10, True)
    # print(type(train_x))
    # print(len(train_x))
    # print(type(train_x[0]))
    # print(train_x[0].shape)
    # print(type(train_x[0][0]))
    train_x = dataset.get_train_x(None, True)
    print(type(train_x))
    print(len(train_x))
    print(type(train_x[0]))
    print(train_x[0].shape)
    print(type(train_x[0][0]))
    train_x = dataset.get_train_x(10, False)
    print(type(train_x))
    print(type(train_x[0]))
    print(len(train_x[0]))
    print(type(train_x[0][0]))
    train_x = dataset.get_train_x(None, False)
    print(type(train_x))
    print(type(train_x[0]))
    print(len(train_x[0]))
    print(type(train_x[0][0]))
    train_x = dataset.get_train_x(None, special_chars_bow=[':)', ':(', ':/'])
    print(type(train_x))
    print(type(train_x[0]))
    print(type(train_x[0][0]))
    for bow in train_x:
        print(bow)

