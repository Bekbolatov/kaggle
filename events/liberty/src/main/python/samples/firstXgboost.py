
import pandas as pd
import numpy as np
from sklearn import preprocessing
import xgboost_avhi as xgb
from sklearn.feature_extraction import DictVectorizer
from gini import gini

all  = pd.read_csv('../input/train.csv', index_col=0)
lb  = pd.read_csv('../input/test.csv', index_col=0)

all_labels = all.Hazard
all.drop('Hazard', axis=1, inplace=True)

lb_index = lb.index

sall = all.T.to_dict().values()
lb = lb.T.to_dict().values()

vec = DictVectorizer()
train = vec.fit_transform(train)
test = vec.transform(test)


def xgboost_pred(train,labels,test):
    params = {}
    params["objective"] = "reg:linear"
    params["eta"] = 0.005
    params["min_child_weight"] = 4
    params["subsample"] = 0.7
    params["colsample_bytree"] = 0.7
    params["scale_pos_weight"] = 1.0
    params["silent"] = 1
    params["max_depth"] = 7

    plst = list(params.items())

    #Using 5000 rows for early stopping.
    offset = 5000

    num_rounds = 10000
    xgtest = xgb.DMatrix(test)

    #create a train and validation dmatrices
    xgtrain = xgb.DMatrix(train[offset:,:], label=labels[offset:])
    xgval = xgb.DMatrix(train[:offset,:], label=labels[:offset])
    watchlist = [(xgtrain, 'train'),(xgval, 'val')]


    model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=80)

    preds = model.predict(xgtest)


    return preds







preds = xgboost_pred(train,labels,test)

preds = pd.DataFrame({"Id": test_ind, "Hazard": preds})
preds = preds.set_index('Id')
preds.to_csv('xgboost_benchmark_kk.csv')
