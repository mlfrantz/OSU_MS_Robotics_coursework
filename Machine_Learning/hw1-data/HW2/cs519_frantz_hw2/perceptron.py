import numpy as np
import csv

class Perceptron:

    def __init__(self, train, dev, test, all_binary = False, add_numerical = False, zero_mean = False, unit_variance = False, combination = False):
        assert (type(train) is str), "Data must be string of the file name"
        assert (type(dev) is str), "Data must be string of the file name"
        assert (type(test) is str), "Data must be string of the file name"

        self.train = train
        self.dev = dev
        self.test = test

        self.train_x = []
        self.train_y = []
        self.dev_x = []
        self.dev_y = []

        self.train_raw = list(map(lambda s: s.strip().split(", "), open(self.train).readlines()))
        self.dev_raw = list(map(lambda s: s.strip().split(", "), open(self.dev).readlines()))
        self.test_raw = list(map(lambda s: s.strip().split(", "), open(self.test).readlines()))

        self.feature_map = []

        self.format_data(all_binary, add_numerical = add_numerical, zero_mean = zero_mean, unit_variance = unit_variance, combination = combination) # This method conducts all the formatting and vectorization

    def format_data(self, all_binary = False, add_numerical = False, zero_mean = False, unit_variance = False, combination = False):
        # Takes in the data set and converts it so that we can use it in our classifier

        train_data = self.train_raw

        mapping = {}
        new_data = []
        for row in train_data:
            new_row = []
            for j, x in enumerate(row[:-1]):
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

            if add_numerical:
                # These add the numerical features to the end of the vector
                new_row.append(int(row[0]))
                new_row.append(int(row[7]))

            if combination: # pick two
                #new_row.append(new_row[1]) # if sector
                #new_row.append(new_row[2]) # If education
                new_row.append(new_row[3]) # If marital status
                new_row.append(new_row[4]) # if occupation
                #new_row.append(new_row[5]) # If Race
                #new_row.append(new_row[6]) # if Gender
                #new_row.append(new_row[8]) # if country of origin

            self.train_y.append(row[-1])
            new_data.append(new_row)

        self.feature_map = mapping

        print("Dimensions:", len(self.feature_map))

        # Test set needs to be fixed
        for data in [self.train, self.dev, self.test]:
            if data is self.train:
                temp_data = train_data
            else:
                temp_data = list(map(lambda s: s.strip().split(", "), open(data).readlines()))
                new_data = []
                for row in temp_data:
                    new_row = []
                    temp_row = row
                    if data is self.dev:
                        temp_row = row[:-1]
                    for j, x in enumerate(temp_row):
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

                    if data is self.dev:
                        self.dev_y.append(row[-1])

                    if add_numerical: # These add the numerical features to the end of the vector
                        new_row.append(int(row[0]))
                        new_row.append(int(row[7]))

            bindata = np.zeros((len(temp_data), len(mapping)))
            if add_numerical:
                bindata = np.zeros((len(temp_data), len(mapping)+2))
            elif combination:
                bindata = np.zeros((len(temp_data), len(mapping)+1))

            mask = [1, 0, 0, 0, 0, 0, 0, 1, 0, 2] # Used to determine if int or not for age and hours worked
            for i, row in enumerate(new_data):
                for j, x in enumerate(row):
                    if not all_binary:
                        m = mask[j]
                        if m == 0:
                            bindata[i][x] = 1
                        elif m == 1:
                            bindata[i][j] = x  #*0.02# /50 added to normalize field /
                            # Used *0.02 since multiplication is faster than division
                        elif m == 2:
                            if data is self.train:
                                bindata[i][x] = 1
                            else:
                                bindata[i][x] = 0
                    else:
                        if j < 9:
                            bindata[i][x] = 1
                        if add_numerical:
                            if j == 9:
                                bindata[i][-2] = x
                            if j == 10:
                                bindata[i][-1] = x
                        if combination:
                            if j == 9:
                                if self.feature_map[(3, 'Never-married')] == row[-2] and self.feature_map[(4, 'Sales')] == row[-1]:
                                    bindata[i][-1] = 1

            if data is self.train:
                self.train_x = bindata
                if zero_mean and add_numerical:
                    temp_array = np.mean(self.train_x, axis=0)
                    mean_array = np.zeros(len(temp_array))
                    mean_array[-2] = temp_array[-2]
                    mean_array[-1] = temp_array[-1]
                    self.train_x = self.train_x - mean_array
                    #print(self.train_x[0:1])
                    if unit_variance:
                        temp_array = np.std(self.train_x, axis=0)
                        mean_array = np.ones(len(temp_array))
                        mean_array[-2] = temp_array[-2]
                        mean_array[-1] = temp_array[-1]
                        self.train_x = self.train_x/mean_array
                        #print(self.train_x[0:1])
                self.train_y = [-1 if x == '<=50K' else 1 for x in self.train_y]
            elif data is self.dev:
                self.dev_x = bindata
                if zero_mean and add_numerical:
                    temp_array = np.mean(self.dev_x, axis=0)
                    mean_array = np.zeros(len(temp_array))
                    mean_array[-2] = temp_array[-2]
                    mean_array[-1] = temp_array[-1]
                    self.dev_x = self.dev_x - mean_array
                    #print(self.dev_x[0:1])
                    if unit_variance:
                        temp_array = np.std(self.dev_x, axis=0)
                        mean_array = np.ones(len(temp_array))
                        mean_array[-2] = temp_array[-2]
                        mean_array[-1] = temp_array[-1]
                        self.dev_x = self.dev_x/mean_array
                self.dev_y = [-1 if x == '<=50K' else 1 for x in self.dev_y]
            elif data is self.test:
                self.test = bindata

    def basic_train(self):
        weights = np.zeros(len(self.train_x[:][0]))
        bias = 0
        updates = 0

        for epoch in range(5):
            for i, row in enumerate(self.train_x):
                a = np.dot(weights, row) + bias
                if self.train_y[i]*a <= 0:
                    weights = weights + self.train_y[i]*row
                    bias += self.train_y[i]
                    updates += 1
            print("epoch" , epoch, "updates", updates, "(", 100*updates/len(self.train_x), "%)")
            self.basic_test(weights, bias)
            updates = 0
        print("Bias:", bias)

    def basic_test(self, weights, bias):
        error = np.zeros(len(self.dev_x))
        pos = np.zeros(len(self.dev_x))
        for j, row in enumerate(self.dev_x):
            a = np.dot(weights, row) + bias
            if a > 0: # If positive add to positive list
                pos[j] = 1
            if np.sign(a) != np.sign(self.dev_y[j]):
                error[j] = 1

        print("dev_error", 100*np.sum(error)/len(self.dev_x), "(",100*np.sum(pos)/len(self.dev_x),"%)" )

    def average_train(self, test=False):
        # Implementing the clever version
        weights = np.zeros(len(self.train_x[:][0]))
        weights_a = np.zeros(len(self.train_x[:][0]))
        c = 0
        bias = 0
        cached_bias = 0
        updates = 0

        for epoch in range(5):
            for i, row in enumerate(self.train_x):
                a = np.dot(weights, row) + bias
                if self.train_y[i]*a <= 0:
                    weights = weights + self.train_y[i]*row
                    bias += self.train_y[i]
                    weights_a += c*self.train_y[i]*row
                    cached_bias += self.train_y[i]*c
                    updates += 1
                c += 1
            print("epoch" , epoch, "updates", updates, "(", 100*updates/len(self.train_x), "%)")
            output_weights = weights - weights_a/c
            output_bias = bias - bias/c
            if test:
                if epoch == 4:
                    self.average_test(output_weights, output_bias, test=test)
            else:
                self.average_test(output_weights, output_bias, test=test)
            updates = 0

        top_5 = reversed(np.argsort(output_weights)[-5:])
        bottom_5 = np.argsort(output_weights)[:5]
        print("Top 5:", [list(self.feature_map.keys())[list(self.feature_map.values()).index(f)] for f in top_5])
        print("Bottom 5:", [list(self.feature_map.keys())[list(self.feature_map.values()).index(f)] for f in bottom_5])
        #print("All:", [(list(self.feature_map.keys())[list(self.feature_map.values()).index(f)], output_weights[f]) for f in np.argsort(output_weights)])
        print("Bias:", output_bias)

    def average_test(self, weights, bias, test=False):

        if not test:
            error = np.zeros(len(self.dev_x))
            pos = np.zeros(len(self.dev_x))
            for j, row in enumerate(self.dev_x):
                a = np.dot(weights, row) + bias
                if a > 0: # If positive add to positive list
                    pos[j] = 1
                if np.sign(a) != np.sign(self.dev_y[j]):
                    error[j] = 1

            print("dev_error", 100*np.sum(error)/len(self.dev_x), "(",100*np.sum(pos)/len(self.dev_x),"%)" )
        else:
            #test
            test_data = []
            pos = np.zeros(len(self.test))
            for j, row in enumerate(self.test):
                a = np.dot(weights, row) + bias
                if a > 0: # If positive add to positive list
                    pos[j] = 1
                new_row = self.test_raw[j]
                if pos[j] == 0:
                    new_row.append('<=50K')
                else:
                    new_row.append('>50K')
                test_data.append(new_row)
            print(test_data[:5])
            print("Positive %", "(",100*np.sum(pos)/len(self.dev_x),"%)" )

            with open("income.test.predicted", 'w') as f:
                for item in test_data:
                    f.write(", ".join(str(line) for line in item))
                    f.write("\n")


if __name__ == "__main__":
    train = 'income.train.txt.5k'
    dev = 'income.dev.txt'
    test = 'income.test.blind'

    train_positive = 'income.train.postive.txt'
    train_negative = 'income.train.negative.txt'

    example = Perceptron(train, dev, test, all_binary=True, add_numerical = False, zero_mean = False, unit_variance = False, combination = True)

    #example.basic_train()
    example.average_train(test=False)
