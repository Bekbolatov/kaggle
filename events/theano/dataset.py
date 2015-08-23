import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer

loc = "/Users/rbekbolatov/data/kaggle/liberty"

def get_data():
    train = pd.read_csv(loc + '/train.csv', index_col=0)
    test = pd.read_csv(loc + '/test.csv', index_col=0)

    labels = train.Hazard

    train_s = train
    test_s = test

    train_s.drop('T2_V10', axis=1, inplace=True)
    train_s.drop('T2_V7', axis=1, inplace=True)
    train_s.drop('T1_V13', axis=1, inplace=True)
    train_s.drop('T1_V10', axis=1, inplace=True)

    test_s.drop('T2_V10', axis=1, inplace=True)
    test_s.drop('T2_V7', axis=1, inplace=True)
    test_s.drop('T1_V13', axis=1, inplace=True)
    test_s.drop('T1_V10', axis=1, inplace=True)

    columns = train.columns
    test_ind = test.index

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
    #train_s = np.column_stack([train_s, np.multiply(train_s[:, 7], train_s[:, 18])])
    #test_s = np.column_stack([test_s, np.multiply(test_s[:, 6], test_s[:, 17])])

    train_s = train_s.astype(float)
    test_s = test_s.astype(float)

    #preds1 = xgboost_pred(train_s[::, 1::],labels,test_s)
    return (pd.DataFrame(train_s[::, 1::]), np.asarray(labels), pd.DataFrame(test_s), np.asarray(test_ind))

def get_data_orig():
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
