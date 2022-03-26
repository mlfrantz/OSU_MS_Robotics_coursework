import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from copy import copy

class Regression:

    def __init__(self, train, dev, test, all_binary = False, combine = False):
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
        self.test_x = []

        self.train_raw = list(map(lambda s: s.strip().split(","), open(self.train).readlines()))
        self.dev_raw = list(map(lambda s: s.strip().split(","), open(self.dev).readlines()))
        self.test_raw = list(map(lambda s: s.strip().split(","), open(self.test).readlines()))

        na_to_zero = ['LotArea','MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageYrBlt']#, 'GarageCars', 'GrLivArea', 'GarageArea','WoodDeckSF', 'OpenPorchSF']
        na_to_mean = ['LotFrontage']
        num_to_log = ['LotArea', 'MasVnrArea', '1stFlrSF', '2ndFlrSF', 'GarageArea', 'YearBuilt', 'YearRemodAdd', 'YrSold']
        # combination_features = ['OverallQual', 'OverallCond'] # This did nothing
        # combination_features = ['YearBuilt', 'YearRemodAdd'] # This did nothing
        # combination_features = ['LotArea', 'LotFrontage'] # Slightly better 0.11786403272477354
        combination_features = ['TotalBsmtSF', 'MasVnrArea'] # 0.11770941177665438 even better

        self.labels = self.train_raw[0][1:-1]
        self.compare_feature = []
        for i, label in enumerate(self.labels):
            if label in combination_features:
                self.compare_feature.append(i)
        print(self.compare_feature)

        if all_binary == False:
            for data in [self.train_raw, self.dev_raw, self.test_raw]:
                for i, row in enumerate(data):
                    if i == len(self.train_raw)-1:
                        continue
                    if i > 0:
                        break
                    for j, col in enumerate(row):
                        if col in na_to_mean:
                            for k, value in enumerate(data[1:][j]):
                                if value == 'NA': # Lotfrontage
                                    data[k+1][j] = str(70)
                        if col in na_to_zero:
                            for k, value in enumerate(data[1:][j]):
                                if value == 'NA':
                                    data[k+1][j] = str(0)
                        if col in num_to_log:
                            for k, value in enumerate(data[1:]):
                                if data[k+1][j] == 'NA':
                                    data[k+1][j] = str(0)
                                data[k+1][j] = np.log1p(float(data[k+1][j]))
                if data is self.train_raw:
                    self.train_raw = data
                if data is self.dev_raw:
                    self.dev_raw = data
                if data is self.test_raw:
                    self.test_raw = data

        self.featuretype = []
        for item in self.train_raw[1][1:-1]:
            try:
                int(item)
                self.featuretype.append(1)
            except ValueError:
                self.featuretype.append(0)

        if combine:
            self.featuretype.append(2)

        self.feature_map = []

        self.format_data(all_binary, combine) # This method conducts all the formatting and vectorization

    def format_data(self, all_binary = False, combine = False):
        # Takes in the data set and converts it so that we can use it in our classifier

        mapping = {(-1, 'bad'):0}
        for data in [self.train_raw, self.dev_raw, self.test_raw]:
            for i, row in enumerate(data):
                if i == 0: # skip Id row
                    continue
                if data is self.test_raw:
                    row_repl = copy(row[1:])
                else:
                    row_repl = copy(row[1:-1])
                new_row = []
                for j, x in enumerate(row_repl): # Dont want ID or output
                    if not all_binary:
                        if self.featuretype[j] == 1:
                            try:
                                feature = (j)
                                mapping[feature] = int(x)
                            except ValueError:
                                feature = (j)
                                mapping[feature] = 0
                        else:
                            feature = (j, x)
                            if data is self.train_raw:
                                if feature not in mapping:
                                    mapping[feature] = len(mapping)
                    else:
                        feature = (j, x)
                        if data is self.train_raw:
                            if feature not in mapping:
                                mapping[feature] = len(mapping)
                    try:
                        new_row.append(mapping[feature])
                    except KeyError:
                        # This is used to catch dev features that are not in the training set
                        feature = (-1, 'bad')
                        new_row.append(mapping[feature])

                if data is self.train_raw:
                    self.train_x.append(new_row)
                    self.train_y.append(np.log(int(row[-1])))
                if data is self.dev_raw:
                    self.dev_x.append(new_row)
                    self.dev_y.append(np.log(int(row[-1])))
                if data is self.test_raw:
                    self.test_x.append(new_row)

        self.feature_map = mapping
        print("Dimensions:", len(self.feature_map))

        for temp_data in [self.train_x, self.dev_x, self.test_x]:
            bindata = np.zeros((len(temp_data), len(mapping))) # No bias
            mask = self.featuretype # Used to determine if int or not for age and hours workedas
            for i, row in enumerate(temp_data):
                for j, x in enumerate(row):
                    if not all_binary:
                        m = mask[j]
                        if m == 0:
                            bindata[i][x] = 1
                        elif m == 1:
                            bindata[i][j] = x
                        if j == len(row)-1:
                            #print(bindata[i][self.compare_feature[0]], bindata[i][self.compare_feature[1]])
                            bindata[i][-1] = np.abs(bindata[i][self.compare_feature[0]] - bindata[i][self.compare_feature[1]])
                    else:
                        if x != 0:
                            bindata[i][x] = 1
            if temp_data is self.train_x:
                self.train_x = bindata
                # print(self.train_x[3])
            elif temp_data is self.dev_x:
                self.dev_x = bindata
            elif temp_data is self.test_x:
                self.test_x = bindata
        print(self.dev_x.shape, self.test_x.shape)

    def ridge(self, all_binary = True):
        X = self.train_x
        y = np.transpose(self.train_y)
        alphas = np.linspace(0,10,100)
        best_alpha = np.zeros(len(alphas))
        best_diff = np.zeros(len(alphas))
        for i, alpha in enumerate(alphas):
            clf = Ridge(alpha = alpha)
            clf.fit(X,y)
            predict_dev = clf.predict(self.dev_x)
            predict_dev = np.exp(predict_dev)
            y_test = np.exp(self.dev_y)
            dev_diff = self.rmsle(predict_dev, y_test)
            best_alpha[i] = alpha
            best_diff[i] = dev_diff
        print(best_alpha[np.argmin(best_diff)], np.min(best_diff))

        clf = Ridge(alpha = best_alpha[np.argmin(best_diff)])
        clf.fit(X,y)
        predict_test = clf.predict(self.test_x)
        # for value in predict_test:
        #     print(value)
        ids = np.arange(1461,2920)
        output = pd.DataFrame({'Id':ids, 'SalePrice':np.exp(predict_test)})
        output.to_csv('submission.csv', index=False)

    def train_linear_regression(self):
        #print(len(self.train_y))
        #print(self.train_x.shape)
        #print(self.train_y[:5])
        X = self.train_x
        #X = StandardScaler().fit_transform(self.dev_x)
        y = np.transpose(self.train_y)
        reg = LinearRegression().fit(X,y)
        print(reg.score(X, y))
        #print(reg.coef_)
        #print(reg.intercept_)
        #X_dev = StandardScaler().fit_transform(self.dev_x)
        predict_dev = reg.predict(self.dev_x)
        remove = np.argmin(predict_dev)
        #print(remove)
        if np.min(predict_dev) <= 0:
            predict_dev = np.delete(predict_dev, remove)
            self.dev_y = np.delete(self.dev_y, remove)
        #print(predict_dev)
        dev_diff = self.rmsle(np.exp(predict_dev), np.exp(self.dev_y))
        print(dev_diff)
        #print(np.mean(self.train_y), np.max(self.train_y), np.median(self.train_y))
        # top_10 = list(reversed(np.argsort(reg.coef_)[-10:]))
        # bottom_10 = np.argsort(reg.coef_)[:10]
        # Use these tops for binary features
        #top = [list(self.feature_map.keys())[list(self.feature_map.values()).index(f)] for f in top_10]
        #top = [(self.labels[int(i[0])],i[1]) for i in top]
        # t = list(self.feature_map.values())
        # w = list(self.feature_map.keys())
        # top = zip([t[f] for f in top_10], [w[f] for f in top_10])
        # bottom = [list(self.feature_map.keys())[list(self.feature_map.values()).index(f)] for f in bottom_10]
        # bottom = [(self.labels[int(i[0])],i[1]) for i in bottom]
        # print("Top 10:", list(top))
        # print("Bottom 10:", bottom)

    def test_linear_regression(self):
        X = self.train_x
        y = np.transpose(self.train_y)
        reg = LinearRegression().fit(X,y)
        print(reg.score(self.train_x, self.train_y))
        ids = np.arange(1461,2920)
        predict_test = reg.predict(self.test_x)
        print(self.train_x[0])
        print(self.test_x[0])
        for value in predict_test:
            print(value)

        for i, val in enumerate(predict_test):
            if val < 0:
                print(i, np.log1p(-1*val)/2)
            elif val > 20:
                print(i, np.log1p(val)/2)

        output = pd.DataFrame({'Id':ids, 'SalePrice':np.exp(predict_test)})
        output.to_csv('submission.csv', index=False)


    def rmsle(self, y_pred, y_test):
        assert len(y_test) == len(y_pred)
        return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_test))**2))


if __name__ == "__main__":
    train = 'my_train_2.csv' # Used for testing, I altered mine for actual experiamentation
    dev = 'my_dev_2.csv'  # Used for testing
    actual_train = 'train.csv' # This is the one to train on for submission
    test = 'test_2.csv'

    _training = Regression(train, dev, test, all_binary=False, combine=True)
    #_training.train_linear_regression()
    _training.ridge(all_binary = False)
    # testing = Regression(actual_train, dev, test, all_binary = False)
    # testing.test_linear_regression()
