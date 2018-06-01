import os
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet

NEGATIVE_EXAMPLES_CLASS = 0
POSITIVE_EXAMPLES_CLASS = 1


class Dataset:
    train = list()
    test = list()
    corpus = list()
    corpus_bow = list()

    def __init__(self):
        print("Loading dataset")

        for filename in os.listdir('Dataset/negative_examples_train'):
            self.train.append((self.__get_text(filename), NEGATIVE_EXAMPLES_CLASS))
        print("Done loading negative train")
        for filename in os.listdir('Dataset/positive_examples_train'):
            self.train.append((self.__get_text(filename), POSITIVE_EXAMPLES_CLASS))
        print("Done loading positive train")
        for filename in os.listdir('Dataset/negative_examples_test'):
            self.test.append((self.__get_text(filename), NEGATIVE_EXAMPLES_CLASS))
        print("Done loading negative test")
        for filename in os.listdir('Dataset/positive_examples_test'):
            self.test.append((self.__get_text(filename), POSITIVE_EXAMPLES_CLASS))
        print("Done loading positive test")

        self.__build_corpus()

        print("Dataset loaded\nStats:\nnum positive train: {}\nnum negative train: {}\n"
              "num train: {}\nnum positive test: {}\nnum negative test: {}\n"
              "num test: {}\nnum total words in corpus: {}\nnum distinct words in corpus: {}"
              .format(*self.get_stats()))  # printing some statistics so it's nice looking

    def get_train(self, number_of_posts=None, number_of_sentences=None):
        return self.__parse_set(self.train, number_of_posts, number_of_sentences)

    def get_test(self, number_of_posts=None, number_of_sentences=None):
        return self.__parse_set(self.test, number_of_posts, number_of_sentences)

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
        num_distinct_words = len(self.corpus_bow)
        return num_positive_train, num_negative_train, num_train, num_positive_test, num_negative_test, num_test, num_words, num_distinct_words

    def __get_text(self, xml_file_to_parse):
        return list()

    def __build_corpus(self):

        pass

    def __lemmatize(self, sentence_to_lemmatize):

        pass

    def __parse_set(self, set, number_of_posts=None, number_of_sentences=None):
        if number_of_posts is not None:
            pass
        elif number_of_sentences is not None:
            pass
        else:
            pass

    def __get_bag_of_words(self, text):

        pass


if __name__ == "__main__":
    dataset = Dataset()
