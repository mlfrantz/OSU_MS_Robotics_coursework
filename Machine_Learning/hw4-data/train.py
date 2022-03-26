#!/usr/bin/env python

from __future__ import division

import sys
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from svector import svector
import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')
# nltk.download('punkt')

def read_from(textfile):
    for line in open(textfile):
        label, words = line.strip().split("\t")
        yield (1 if label=="+" else -1, words.split())

def make_vector(words):
    v = svector()
    for word in words:
        v[word] += 1
    return v

def test(devfile, model, bias=0.0, bigrams=[]):
    tot, err = 0, 0
    dev_data = list(read_from(devfile))
    test_data = list(read_from(devfile))

    for i, (label, words) in enumerate(dev_data):
        try:
            word_list = [(words[j],words[j+1]) for j in range(0,len(words)-1)  if (words[j],words[j+1]) in bigrams]
            word = [w for w in words]
            word.extend(word_list)
            dev_data[i] = (label, word)
        except IndexError:
            dev_data[i] = (label, ['eh'])

    if sys.argv[2] == 'dev.txt':
        for i, (label, words) in enumerate(dev_data, 1): # note 1...|D|
            err += label * ((model.dot(make_vector(words)))+bias) <= 0
        return err/i  # i is |D| now
    else:
        # Used for test.txt
        for i, (label, words) in enumerate(dev_data):
            pre = '+' if ((model.dot(make_vector(words)))+bias) >= 0 else '-'
            test_data[i] = [str(pre) + " " + (' ').join(test_data[i][1])]

        with open('test.txt.predicted', 'w') as file:
            for item in test_data:
                file.write("%s\n" % item[0])


def train(trainfile, devfile, epochs=5):
    t = time.time()
    best_err = 1.
    bias = 0
    model = svector()
    for it in xrange(1, epochs+1):
        updates = 0
        for i, (label, words) in enumerate(read_from(trainfile), 1): # label is +1 or -1
            sent = make_vector(words)
            if label * (model.dot(sent)+bias) <= 0:
                updates += 1
                model += label * sent
                bias += label
        dev_err = test(devfile, model, bias)
        best_err = min(best_err, dev_err)
        print("epoch %d, update %.1f%%, dev %.1f%%" % (it, updates / i * 100, dev_err * 100))
        # print "bias: %0.1f" % bias
    print("best dev err %.1f%%, |w|=%d, time: %.1f secs" % (best_err * 100, len(model), time.time() - t))

def sklearn_test(trainfile, devfile):
    t = time.time()
    train_data = list(read_from(trainfile)) # cached values for training data
    test_data = list(read_from(devfile)) # cached values for training data
    # Find single words
    new_train = svector()
    for label, words in train_data:
        new_train += make_vector(words)
    single_words = set([k for k, v in new_train.items() if v <= 2.0])
    stop_words =  set(stopwords.words('english'))
    other_words = [',', "'"]

    for i, (label, words) in enumerate(train_data):
        train_data[i] = (label, [word for word in words if word not in single_words and word not in stop_words and word not in other_words])

    for i, (label, words) in enumerate(train_data):
        train_data[i] = (label, ' '.join(words))

    test_data_y = np.zeros(len(test_data))
    for i, (label, words) in enumerate(test_data):
        test_data[i] = (label, ' '.join(words))
        test_data_y[i] = label

    train_data = pd.DataFrame.from_records(train_data, columns=['sentiment','review'])
    test_data = pd.DataFrame.from_records(test_data, columns=['sentiment','review'])
    vectorizer = TfidfVectorizer(sublinear_tf=True, use_idf=True)
    train_vectors = vectorizer.fit_transform(train_data['review'])
    test_vectors = vectorizer.transform(test_data['review'])

    svm_learn = SVC(C=1, kernel='linear', gamma='auto')
    svm_learn.fit(train_vectors, train_data['sentiment'])
    prediction = svm_learn.predict(test_vectors)

    wrong = [ 1 if test_data_y[i]*prediction[i] < 0.0 else 0 for i in range(len(prediction))]
    print(sum(wrong)/len(prediction) * 100)

    nn_learn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1)
    nn_learn.fit(train_vectors, train_data['sentiment'])
    nn_prediction = nn_learn.predict(test_vectors)

    wrong = [ 1 if test_data_y[i]*nn_prediction[i] < 0.0 else 0 for i in range(len(nn_prediction))]
    print(sum(wrong)/len(nn_prediction) * 100)

    xgboost = GradientBoostingClassifier()
    xgboost.fit(train_vectors, train_data['sentiment'])
    xg_prediction = xgboost.predict(test_vectors)

    wrong = [ 1 if test_data_y[i]*xg_prediction[i] < 0.0 else 0 for i in range(len(xg_prediction))]
    print(sum(wrong)/len(xg_prediction) * 100)

def average_train(trainfile, devfile, epochs=20):
    t = time.time()
    best_err = 1.
    dev_err = 0
    c = 0
    bias = 0
    cached_bias = 0
    model = svector()
    cached_model = svector()
    train_data = list(read_from(trainfile)) # cached values for training data

    # Find single words
    new_train = svector()
    for label, words in train_data:
        new_train += make_vector(words)
    single_words = set([k for k, v in new_train.items() if v <= 1.0])
    print(len(single_words))
    stop_words = set(stopwords.words('english'))
    other_words = []#set([',', "'"])

    for i, (label, words) in enumerate(train_data):
        train_data[i] = (label, [word for word in words if word not in single_words and word not in stop_words and word not in other_words])

    # Pull bigrams from list with single_words removed
    bigrams=[]
    bigrams_trains = svector()
    for label, words in train_data:
        nltk_tokens = nltk.word_tokenize(' '.join(words))
        bigrams_trains += make_vector(list(nltk.bigrams(nltk_tokens)))
    bigrams = set([k for k, v in bigrams_trains.items() if v > 2.0])

    for i, (label, words) in enumerate(train_data):
        try:
            word_list = [(words[j],words[j+1]) for j in range(0,len(words)-1)  if (words[j],words[j+1]) in bigrams]
            word = [w for w in words]
            word.extend(word_list)
            train_data[i] = (label, word)
        except IndexError:
            train_data[i] = (label, ['eh'])

    for it in range(1, epochs+1):
        updates = 0
        for i, (label, words) in enumerate(train_data, 1): # label is +1 or -1
            sent = make_vector(words)
            if label * (model.dot(sent)+bias) <= 0:
                updates += 1
                model += label * sent # weights update
                cached_model += c * label * sent # cached weights update
                bias += label # bias update
                cached_bias += c * label
            c += 1
        output_weights = model - (cached_model * (1/c))
        output_bias = bias - (cached_bias * (1/c))
        if devfile == 'dev.txt':
            dev_err = test(devfile, output_weights, output_bias, bigrams)
            best_err = min(best_err, dev_err)
        print("epoch %d, update %.1f%%, dev %.1f%%" % (it, updates / i * 100, dev_err * 100))
    # top_bottom_features(output_weights) # Will print top 20 and bottom 20 features
    # misclassify(devfile, output_weights, output_bias) # Will print all the misclassified dev examples
    print("best dev err %.1f%%, |w|=%d, time: %.1f secs" % (best_err * 100, len(model), time.time() - t))
    test_err = test(devfile, output_weights, output_bias, bigrams)

def top_bottom_features(model):
    sorted_weights = sorted(model.values())
    print("Bottom 20:", [model.keys()[model.values().index(i)] for i in sorted_weights[0:20]])
    print("Top 20:", [model.keys()[model.values().index(i)] for i in sorted_weights[-20:]])

def misclassify(devfile, model, bias=0.0):
    for i, (label, words) in enumerate(read_from(devfile), 1): # note 1...|D|
        a = label * ((model.dot(make_vector(words)))+bias)
        #print a, label
        if a <= 0:
            print("Predicted:", "Negative" if a*label < 0 else "Positive", "Actual:", "Negative" if label < 0 else "Positive", ' '.join(words)) # i is |D| now

if __name__ == "__main__":
    # train(sys.argv[1], sys.argv[2], 10)
    average_train(sys.argv[1], sys.argv[2], 9)
    # sklearn_test(sys.argv[1], sys.argv[2])
