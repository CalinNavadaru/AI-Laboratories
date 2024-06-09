import random

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

word2vec_model = KeyedVectors.load_word2vec_format("data/GoogleNews-vectors-negative300.bin", binary=True)
data = pd.read_csv("data/reviews_mixed.csv")

list_reviews = data['Text'].to_list()
sentiments_reviews = data['Sentiment'].to_list()
ground_truth = pd.factorize(data['Sentiment'])
ground_truth = ground_truth[0]


def train_feature_w2v(data_arg):
    result_list = []
    for prop in data_arg:
        feature = 0
        list_words = prop.split()
        for word_arg in list_words:
            word_arg = word_arg.strip()
            if word_arg in word2vec_model.index_to_key and len(word_arg) > 2:
                feature += np.mean(word2vec_model[word_arg])
            else:
                feature += 0

        feature = feature / len(list_words)
        result_list.append(feature)

    return np.array(result_list).reshape(-1, 1)


def stop(old_c, c, no_iteration):
    if no_iteration > 10000:
        return True
    return old_c == c


class KMeans:

    def __init__(self, input_size, number_of_classes):
        self.input_size = input_size
        self.centroids = None
        self.number_of_classes = number_of_classes

    def fit(self, train_data_arg):
        values_random = random.sample(list(train_data_arg), self.number_of_classes)
        self.centroids = np.array([[x] for x in values_random])
        no_iteration = 0
        old_c = None
        c = []
        while not stop(old_c, c, no_iteration):
            old_c = c.copy()
            c = []
            no_iteration += 1
            for i in range(len(train_data_arg)):
                c_min = np.linalg.norm(self.centroids[0] - train_data_arg[i])
                c_index = 0
                for j in range(1, len(self.centroids)):
                    d = np.linalg.norm(self.centroids[j] - train_data_arg[i])
                    if c_min > d:
                        c_index = j
                        c_min = d
                c.append(c_index)

            for j in range(len(self.centroids)):
                denominator = 0
                numerator = 0
                for i in range(len(c)):
                    if c[i] == j:
                        numerator += train_data_arg[i]
                        denominator += 1

                self.centroids[j] = numerator / denominator if denominator != 0 \
                    else train_data_arg[random.randint(0, len(train_data_arg) - 1)]

    def predict(self, test_data_arg):
        result_predict = []
        for i in range(len(test_data_arg)):
            c_min = np.linalg.norm(self.centroids[0] - test_data_arg[i])
            c_index = 0
            for j in range(1, len(self.centroids)):
                d = np.linalg.norm(self.centroids[j] - test_data_arg[i])
                if c_min > d:
                    c_index = j
                    c_min = d

            result_predict.append(c_index)

        return result_predict


from random import shuffle

indexes = [i for i in range(len(list_reviews))]
shuffle(indexes)
train_indexes = indexes[:int(0.75 * len(indexes))]
train_input = [list_reviews[i] for i in train_indexes]
test_input = [list_reviews[i] for i in indexes if i not in train_indexes]
test_output = [ground_truth[i] for i in indexes if i not in train_indexes]

train_input = train_feature_w2v(train_input)
test_input = train_feature_w2v(test_input)

unsupervisedClassifier = KMeans(1, 2)
unsupervisedClassifier.fit(train_input)
predicted = unsupervisedClassifier.predict(test_input)

from sklearn.metrics import accuracy_score

print(f'Accuracy {accuracy_score(predicted, test_output)}')
