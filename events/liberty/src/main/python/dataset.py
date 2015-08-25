import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from collections import defaultdict

class FactorToNumericEncoder:

    def get_data_stats(self, labels, *data_sets):
        self.stats_new_fit(labels, data_sets[0])
        return [self.stats_enrich(data_set) for data_set in data_sets]

    def stats_new_fit(self, labels, train, ops=[('mean', np.mean)]):
        self.stats_reset()

        for op_name, op in ops:
            self.value_defaults[op_name] = op(labels)
            for i in range(train.shape[1]):
                if type(train[1, i]) is str:
                    for v in train[:, i]:
                        if v not in self.value_stats[i][op_name].keys():
                            self.value_stats[i][op_name][v] = op(labels[train[:, i] == v])

    def stats_enrich(self, test, ops=['mean']):
        for op_name in ops:
            for i in range(test.shape[1]):
                if type(test[1, i]) is str:
                    for n in range(len(test)):
                        test[n, i] = self.value_stats[i][op_name].get(test[n, i], self.value_defaults[op_name])
        return test.astype('float32')

    def stats_reset(self):
        self.value_stats = defaultdict(lambda: defaultdict(dict))
        self.value_defaults = defaultdict(lambda: 0.0)


class LibertyFeatures:

    def featurize(self, version, interactions_to_add, cols_to_drop, labels, *data_sets):
        print(version)
        if version == 'qinchen':
            return self.qinchen_featurize(interactions_to_add, cols_to_drop, labels, *data_sets)
        elif version == 'renat':
            return self.renat_featurize(interactions_to_add, cols_to_drop, labels, *data_sets)
        else:
            raise KeyError

    # qinchen
    def qinchen_add_second_order(self, data, interactions_to_add = {}):
        default_interactions_to_add = {
            0: [3, 7, 16, 28],
            2: [28],
            10: [14, 27],
            22: [24],
            24: [29],
            27: [28]
        }
        for a, bs in (interactions_to_add or default_interactions_to_add).items():
            for b in bs:
                data = np.column_stack([data, np.multiply(data[:, a], data[:, b])])

        return data

    def qinchen_drop_cols(self, data, cols_to_drop = []):
        default_cols_to_drop = [9, 12, 23, 26]
        for a in sorted((cols_to_drop or default_cols_to_drop), reverse=True):
            data = np.delete(data, a, axis=1)
        return data

    def qinchen_featurize(self, interactions_to_add, cols_to_drop, labels, *data_sets):
        data_sets = FactorToNumericEncoder().get_data_stats(labels, *data_sets)
        data_sets = [
            self.qinchen_drop_cols(self.qinchen_add_second_order(data_set))
            for data_set in data_sets
        ]
        return data_sets

    # renat
    def renat_add_second_order(self, data, interactions_to_add = []):
        default_interactions_to_add = {
            0: [3, 7, 16, 28],
            2: [28],
            10: [14, 27],
            22: [24],
            24: [29],
            27: [28]
        }
        default_interactions_to_add = sum([ map(lambda s: (k, s), v) for k,v in default_interactions_to_add.items()], [])
        for a, b in (interactions_to_add or default_interactions_to_add):
            data = np.column_stack([data, np.multiply(data[:, a], data[:, b])])

        return data

    def renat_drop_cols(self, data, cols_to_drop = []):
        default_cols_to_drop = [9, 12, 23, 26]
        for a in sorted((cols_to_drop or default_cols_to_drop), reverse=True):
            data = np.delete(data, a, axis=1)
        return data

    def renat_change_cols(self, data, changes={}):
        for col_num, fn in changes.items():
            data[:, col_num] = fn(data[:, col_num])
        return data

    def renat_featurize(self, interactions_to_add, cols_to_drop, labels, *data_sets):
        data_sets = FactorToNumericEncoder().get_data_stats(labels, *data_sets)
        data_sets = [self.renat_add_second_order(data_set, interactions_to_add) for data_set in data_sets]
        data_sets = [self.renat_change_cols(data_set) for data_set in data_sets]
        data_sets = [self.renat_drop_cols(data_set, cols_to_drop) for data_set in data_sets]
        return data_sets


class LibertyEncoder:

    label_to = np.vectorize(lambda x: x ** 0.5)

    def __init__(self, loc="/Users/rbekbolatov/data/kaggle/liberty"):
        self.train = pd.read_csv(loc + '/train.csv')
        self.test = pd.read_csv(loc + '/test.csv')

        self.labels = self.train.Hazard
        self.test_ind = self.test.Id

        self.train.drop('Id', axis = 1, inplace=True)
        self.train.drop('Hazard', axis = 1, inplace=True)
        self.test.drop('Id', axis = 1, inplace=True)

    def get_orig_data_copy(self):
        return (
            np.copy(np.asarray(self.train)),
            self.label_to(np.copy(np.asarray(self.labels))),
            np.copy(np.asarray(self.labels)),
            np.copy(np.asarray(self.test)),
            np.copy(np.asarray(self.test_ind)))

    def transform(self, version, interactions_to_add, cols_to_drop, labels, *data_sets):
        data_sets = [np.copy(data_set) for data_set in data_sets]
        transformed = LibertyFeatures().featurize(version, interactions_to_add, cols_to_drop, labels, *data_sets)
        print("Number of features = %d" % transformed[0].shape[1])
        return transformed


class Other:
    def enrich_qinchen(train, test, labels_train):
        train_s = train
        test_s = test
        labels = labels_train

        train_s.drop('T2_V10', axis=1, inplace=True)
        train_s.drop('T2_V7', axis=1, inplace=True)
        train_s.drop('T1_V13', axis=1, inplace=True)
        train_s.drop('T1_V10', axis=1, inplace=True)

        test_s.drop('T2_V10', axis=1, inplace=True)
        test_s.drop('T2_V7', axis=1, inplace=True)
        test_s.drop('T1_V13', axis=1, inplace=True)
        test_s.drop('T1_V10', axis=1, inplace=True)

        columns = train.columns


        train_s = np.array(train_s)
        test_s = np.array(test_s)

        for i in range(train_s.shape[1]):
            if type(train_s[1, i]) is str:
                dic = {}
                for v in train_s[:, i]:
                    if v not in dic.keys():
                        col = train_s[train_s[:, i] == v]
                        dic[v] = np.mean(col[:, 0])
                for n in range(0, len(train_s)):
                    train_s[n, i] = dic[train_s[n, i]]
                for n in range(0, len(test_s)):
                    test_s[n, i-1] = dic[test_s[n, i-1]]

        train_s = np.column_stack([train_s, np.multiply(train_s[:, 1], train_s[:, 4])])
        test_s = np.column_stack([test_s, np.multiply(test_s[:, 0], test_s[:, 3])])
        train_s = np.column_stack([train_s, np.multiply(train_s[:, 1], train_s[:, 8])])
        test_s = np.column_stack([test_s, np.multiply(test_s[:, 0], test_s[:, 7])])
        train_s = np.column_stack([train_s, np.multiply(train_s[:, 1], train_s[:, 15])])
        test_s = np.column_stack([test_s, np.multiply(test_s[:, 0], test_s[:, 14])])
        train_s = np.column_stack([train_s, np.multiply(train_s[:, 1], train_s[:, 25])])
        test_s = np.column_stack([test_s, np.multiply(test_s[:, 0], test_s[:, 24])])
        train_s = np.column_stack([train_s, np.multiply(train_s[:, 3], train_s[:, 25])])
        test_s = np.column_stack([test_s, np.multiply(test_s[:, 2], test_s[:, 24])])
        train_s = np.column_stack([train_s, np.multiply(train_s[:, 10], train_s[:, 13])])
        test_s = np.column_stack([test_s, np.multiply(test_s[:, 9], test_s[:, 12])])
        train_s = np.column_stack([train_s, np.multiply(train_s[:, 10], train_s[:, 24])])
        test_s = np.column_stack([test_s, np.multiply(test_s[:, 9], test_s[:, 23])])
        train_s = np.column_stack([train_s, np.multiply(train_s[:, 21], train_s[:, 22])])
        test_s = np.column_stack([test_s, np.multiply(test_s[:, 20], test_s[:, 21])])
        train_s = np.column_stack([train_s, np.multiply(train_s[:, 22], train_s[:, 26])])
        test_s = np.column_stack([test_s, np.multiply(test_s[:, 21], test_s[:, 25])])
        train_s = np.column_stack([train_s, np.multiply(train_s[:, 24], train_s[:, 25])])
        test_s = np.column_stack([test_s, np.multiply(test_s[:, 23], test_s[:, 24])])

        train_s = train_s.astype(float)
        test_s = test_s.astype(float)

        #preds1 = xgboost_pred(train_s[::, 1::],labels,test_s)
        return (pd.DataFrame(train_s[::, 1::]), np.asarray(labels), pd.DataFrame(test_s))

    def get_data_orig(self):
        dat = pd.read_table('../input/train.csv', sep=",")
        dat_y = dat[['Hazard']].values.ravel()
        dat = dat.drop(['Hazard', 'Id'], axis=1)

        lb = pd.read_table('../input/test.csv', sep=",")
        lb_indices = lb[['Id']].values.ravel()
        lb.drop(['Id'], axis=1, inplace=True)

        # apply OHE to factor columns
        numerics = dat.select_dtypes(exclude=['object']).columns
        factors = dat.select_dtypes(include=['object']).columns

        dat_numerics = dat.loc[:, numerics]
        dat_factors = dat.loc[:, factors]
        lb_numerics = lb.loc[:, numerics]
        lb_factors = lb.loc[:, factors]

        dat_factors_dict = dat_factors.T.to_dict().values()
        lb_factors_dict = lb_factors.T.to_dict().values()
        vectorizer = DictVectorizer(sparse=False)
        vectorizer.fit(dat_factors_dict)
        vectorizer.fit(lb_factors_dict)
        dat_factors_ohe = vectorizer.transform(dat_factors_dict)
        lb_factors_ohe = vectorizer.transform(lb_factors_dict)

        dat_factors_ohe_df = pd.DataFrame(np.hstack((dat_numerics, dat_factors_ohe)))
        lb_factors_ohe_df = pd.DataFrame(np.hstack((lb_numerics, lb_factors_ohe)))

        return (dat_factors_ohe_df, dat_y, lb_factors_ohe_df, lb_indices)
