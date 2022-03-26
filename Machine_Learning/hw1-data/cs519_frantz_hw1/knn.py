import numpy as np
import csv

class KNN:

    def __init__(self, train, dev, test, all_binary = False):
        assert (type(train) is str), "Data must be string of the file name"
        assert (type(dev) is str), "Data must be string of the file name"
        assert (type(test) is str), "Data must be string of the file name"

        self.train = train
        self.dev = dev
        self.test = test

        self.train_raw = list(map(lambda s: s.strip().split(", "), open(self.train).readlines()))
        self.dev_raw = list(map(lambda s: s.strip().split(", "), open(self.dev).readlines()))
        self.test_raw = list(map(lambda s: s.strip().split(", "), open(self.test).readlines()))

        self.format_data(all_binary) # This method conducts all the formatting and vectorization

    def format_data(self, all_binary = False):
        # Takes in the data set and converts it so that we can use it in our classifier

        train_data = self.train_raw

        mapping = {}
        new_data = []
        for row in train_data:
            new_row = []
            for j, x in enumerate(row):
                if not all_binary:
                    # If age and hours per week are not binarized
                    try:
                        feature = j
                        mapping[feature] = int(x)
                    except ValueError:
                        feature = (j, x)
                        if feature not in mapping:
                            mapping[feature] = len(mapping)
                else:
                    feature = (j, x)
                    if feature not in mapping:
                        mapping[feature] = len(mapping)
                new_row.append(mapping[feature])
            new_data.append(new_row)

        for data in [self.train, self.dev, self.test]:
            if data is self.train:
                temp_data = train_data
            else:
                temp_data = list(map(lambda s: s.strip().split(", "), open(data).readlines()))
                new_data = []
                for row in temp_data:
                    new_row = []
                    for j, x in enumerate(row):
                        if not all_binary:
                            # If age and hours per week are not binarized
                            try:
                                feature = j
                                mapping[feature] = int(x)
                            except ValueError:
                                feature = (j, x)
                        else:
                            feature = (j, x)
                        try:
                            new_row.append(mapping[feature])
                        except KeyError:
                            # This catches new data that we didn't see in training
                            new_row.append(j)
                    new_data.append(new_row)

            bindata = np.zeros((len(temp_data), len(mapping)), dtype=int)
            mask = [1, 0, 0, 0, 0, 0, 0, 1, 0, 2] # Used to determine if int or not for age and hours worked
            for i, row in enumerate(new_data):
                for j, x in enumerate(row):
                    if not all_binary:
                        m = mask[j]
                        if m == 0:
                            bindata[i][x] = 1
                        elif m == 1:
                            bindata[i][j] = x * 0.02 # /50 added to normalize field /
                            # Used *0.02 since multiplication is faster than division
                        elif m == 2:
                            if data is self.train:
                                bindata[i][x] = 1
                            else:
                                bindata[i][x] = 0
                    else:
                        bindata[i][x] = 1

            if data is self.train:
                self.train = bindata
            elif data is self.dev:
                self.dev = bindata
            elif data is self.test:
                self.test = bindata

    def knn_predict(self, data, K=[1], dist_name="Euclidean"):
        # Predicts the output for each training example

        k_table = [] # Used for finding the best k value
        labels = np.zeros((len(K),len(data))) # Used to save predicted outputs

        for i, d in enumerate(data):

            if dist_name == "Euclidean":
                diff = np.linalg.norm((self.train[:] - d), axis=1)
            elif dist_name == "Manhattan":
                diff = np.abs((self.train[:] - d)).sum(axis=1)

            diff = np.argsort(diff)

            for k_ind, k in enumerate(K):
                y_hat = 0
                for c in range(k):
                    try:
                        if self.train_raw[diff[c]][-1] == '<=50K':
                            y_hat -= 1
                        elif self.train_raw[diff[c]][-1] == '>50K':
                            y_hat += 1
                    except IndexError:
                        break
                labels[k_ind][i] = y_hat

        if data is self.dev:
            for i, row in enumerate(labels):
                error = np.zeros(len(row))
                pos = np.zeros(len(row))
                for j, l in enumerate(row):
                    if l > 0:
                        pos[j] = 1
                    if self.dev_raw[j][-1] == '<=50K' and l > 0:
                        # Predicted Incorrectly
                        error[j] = 1
                    if self.dev_raw[j][-1] == '>50K' and l < 0:
                        error[j] = 1
                print("k=", K[i], "dev_err", 100*np.sum(error)/len(self.dev_raw), "%", "(+:", 100*np.sum(pos)/len(pos), "%)")
                k_table.append(100*np.sum(error)/len(self.dev_raw))

            with open(f"best_k.{dist_name}.csv", 'w', newline='') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerow(k_table)

        if data is self.train:
            for i, row in enumerate(labels):
                error = np.zeros(len(row))
                pos = np.zeros(len(row))
                for j, l in enumerate(row):
                    if l > 0:
                        pos[j] = 1
                    if self.train_raw[j][-1] == '<=50K' and l > 0:
                        # Predicted Incorrectly
                        error[j] = 1
                    if self.train_raw[j][-1] == '>50K' and l < 0:
                        error[j] = 1

                print("k=", K[i], "train_err", 100*np.sum(error)/len(self.train_raw), "%", "(+:", 100*np.sum(pos)/len(pos), "%)")

        if data is self.test:
            test_data = []
            for i, row in enumerate(labels):
                pos = np.zeros(len(row), dtype=int)
                for j, l in enumerate(row):
                    if l > 0:
                        pos[j] = 1
                    new_row = self.test_raw[j]
                    if pos[j] == 0:
                        new_row.append('<=50K')
                    else:
                        new_row.append('>50K')
                    test_data.append(new_row)
                print("k=", K[i], "(+:", 100*np.sum(pos)/len(pos), "%)")

            with open("income.test.predicted", 'w') as f:
                for item in test_data:
                    f.write(", ".join(str(line) for line in item))
                    f.write("\n")

if __name__ == "__main__":
    train = 'income.train.txt.5k'
    dev = 'income.dev.txt'
    test = 'income.test.blind'

    example = KNN(train, dev, test, all_binary=False)

    # Test on K vector using non binarized age and hours worked
    #K = [1, 3, 5, 7, 9, 99, 999, 9999]

    # Dev set
    #example.knn_predict(example.dev, K=K, dist_name="Euclidean")
    #example.knn_predict(example.dev, K=K, dist_name="Manhattan")

    # Train Set
    #example.knn_predict(example.train, K=K, dist_name="Euclidean")
    #example.knn_predict(example.train, K=K, dist_name="Manhattan")

    # Test the all binary example
    # example_bin = KNN(train, dev, test, all_binary=True)
    # example_bin.knn_predict(example.dev, K=K, dist_name="Euclidean")

    # Find lowest error rate
    #K = np.arange(1, 200, 2)
    #example.knn_predict(example.dev, K=K, dist_name="Euclidean")
    #example.knn_predict(example.dev, K=K, dist_name="Manhattan")
    #K = [15] # Best error rate for dev
    #example.knn_predict(example.test, K=K, dist_name="Euclidean")

    K=[99] # Used for timing test
    example.knn_predict(example.test, K=K, dist_name="Euclidean")
    #example.knn_predict(example.train, K=K, dist_name="Manhattan")
    #example.knn_predict(example.dev, K=K, dist_name="Manhattan")
