from sklearn.cross_validation import StratifiedKFold, KFold, ShuffleSplit,train_test_split, PredefinedSplit
from sklearn.ensemble import RandomForestRegressor , ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV,RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from scipy.stats import randint, uniform
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from gini import normalized_gini, gini_eval
import xgboost as xgb

# Normalized Gini Scorer
gini_scorer = metrics.make_scorer(normalized_gini, greater_is_better = True)


dat=pd.read_table('../input/train.csv',sep=",")
dat_y=dat[['Hazard']].values.ravel()
dat=dat.drop(['Hazard','Id'], axis=1)

lb=pd.read_table('../input/test.csv',sep=",")
lb_indices=lb[['Id']].values.ravel()
lb=lb.drop(['Id'], axis=1)

# apply OHE to factor columns
numerics = dat.select_dtypes(exclude=['object']).columns
factors = dat.select_dtypes(include=['object']).columns

dat_numerics = dat.loc[:, numerics]
dat_factors = dat.loc[:, factors]
lb_numerics = lb.loc[:, numerics]
lb_factors = lb.loc[:, factors]

dat_factors_dict = dat_factors.T.to_dict().values()
lb_factors_dict = lb_factors.T.to_dict().values()
vectorizer = DictVectorizer( sparse = False )
vectorizer.fit(dat_factors_dict)
vectorizer.fit(lb_factors_dict)
dat_factors_ohe = vectorizer.transform(dat_factors_dict)
lb_factors_ohe = vectorizer.transform(lb_factors_dict)

dat_factors_ohe_df = pd.DataFrame(np.hstack((dat_numerics, dat_factors_ohe)))
lb_factors_ohe_df = pd.DataFrame(np.hstack((lb_numerics, lb_factors_ohe)))

N = dat_factors_ohe_df.shape[0]

params = {}
params["objective"] = "reg:linear"
params["eta"] = 0.005
params["min_child_weight"] = 6
params["subsample"] = 0.7
params["colsample_bytree"] = 0.7
params["scale_pos_weight"] = 1
params["silent"] = 1
params["max_depth"] = 9
params["eval_metric"] = 'auc'

plst = list(params.items())
num_rounds = 3000

#    data set splits
#  start
folds=train_test_split(range(N), test_size=0.10, random_state=15) #30% test

train_X = dat_factors_ohe_df.iloc[folds[0],:]
train_y = dat_y[folds[0]]
test_X = dat_factors_ohe_df.iloc[folds[1],:]
test_y = dat_y[folds[1]]

xgtrain = xgb.DMatrix(train_X, label=train_y)
xgval = xgb.DMatrix(test_X, label=test_y)

watchlist = [(xgtrain, 'train'), (xgval, 'val')]

model = xgb.train(plst, xgtrain, num_rounds,
                  evals = watchlist,
                  feval = gini_eval,
                  verbose_eval = False,
                  early_stopping_rounds=100)
val_preds = model.predict(xgval, ntree_limit=model.best_iteration)
normalized_gini(test_y, val_preds)


xglb = xgb.DMatrix(lb_factors_ohe_df)
lb_preds = model.predict(xglb, ntree_limit=model.best_iteration)
#   end

submission = pd.DataFrame({"Id": lb_indices, "Hazard": lb_preds})
submission = submission.set_index('Id')
submission.to_csv('../subm/xgboost_in_python2.csv')
