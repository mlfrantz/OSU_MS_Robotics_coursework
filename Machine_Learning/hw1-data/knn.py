#!/usr/bin/env python

from __future__ import division # float-point division

import sys
from collections import defaultdict, Counter
import itertools
import numpy as np
import time

def process_data(filename):
    X, Y = [], []
    for j, line in enumerate(open(filename)):
        line = line.strip()
        features = line.split(", ")
        feat_vec = np.zeros(dimension)
        for i, fv in enumerate(features[:-1]): # last one is target
            if i in [0,7]:
                feat_vec[feature_map[i, 0]] = float(fv) / 50  # NB: diff 2 not 1!
            elif (i, fv) in feature_map: # ignore unobserved features
                feat_vec[feature_map[i, fv]] = 1

        X.append(feat_vec)
        Y.append(1 if features[-1] == ">50K" else -1) # fake for testdata

    return np.array(X), np.array(Y)

def knn(k, example, (trainX, trainY)):
    #dists = [(np.linalg.norm(example - vecx), y) for vecx, y in train_data]
    neighbors = np.argpartition(np.linalg.norm(example - trainX, axis=1), k)[:k]
    #neighbors = np.argsort(np.linalg.norm(example - trainX, axis=1))[:k]
    votes = trainY[neighbors] # slicing
    return 1 if sum(votes) > 0 else -1
    #return Counter(votes).most_common(1)[0][0] # [(1, 6), (-1, 4)]

def eval(k, (testX, testY), train):
    pred = np.array([knn(k, vecx, train) for vecx in testX])
    errors = sum(pred != testY)
    positives = sum(pred == 1)
    return errors / len(testX) * 100, positives / len(testX) * 100

if __name__ == "__main__":
    field_value_freqs = defaultdict(lambda : defaultdict(int)) # field_id -> value -> freq
    for line in open("income.train.txt.5k"):
        line = line.strip()
        features = line.split(", ")[:-1] # exclude target label
        for i, fv in enumerate(features):
            field_value_freqs[i][0 if i in [0,7] else fv] += 1

    feature_map = {}
    feature_remap = {}
    for i, value_freqs in field_value_freqs.iteritems():
        for v in value_freqs:
            k = len(feature_map) # bias
            feature_map[i, v] = k
            feature_remap[k] = i, v

    dimension = len(feature_map) # bias
    print "dimensionality: ", dimension #, feature_map

    train_data = process_data("income.train.txt.5k")
    dev_data = process_data("income.dev.txt")
    #test_data = process_data("income.test.txt")

    for k in map(int, sys.argv[1:]):
        print "k=%-4d" % k,
        if k > len(train_data[0]):
            k = -1 # infinity; N.B. bug in np.argpartition with very large k
        print 'train_err %4.1f%% (+:%5.1f%%)' % eval(k, (train_data[0][:5000], train_data[1][:5000]), train_data),
        print 'dev_err %4.1f%% (+:%5.1f%%)' % eval(k, (dev_data[0][:1000], dev_data[1][:1000]), train_data)
