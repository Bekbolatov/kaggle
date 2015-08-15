import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn import preprocessing
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression

from gini import normalized_gini, gini_eval
from dataset import get_data
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

print("Loading datasets")
dat_x, dat_y_orig, lb_x, lb_ind = get_data()
dat_y = dat_y_orig ** 0.75


train_index, test_index = train_test_split(range(dat_x.shape[0]), test_size=0.1, random_state=102)

train_X_unscaled = dat_x.iloc[train_index, :]
train_y = dat_y[train_index]
test_X_unscaled = dat_x.iloc[test_index, :]
test_y = dat_y[test_index]

print("Scaling features")
scaler = preprocessing.StandardScaler().fit(train_X_unscaled)
train_X = scaler.transform(train_X_unscaled)
test_X = scaler.transform(test_X_unscaled)



#  KNN Regressor preds
print("Running KNN regressor")
neigh = KNeighborsRegressor(n_neighbors=101, weights='uniform', p=1)
neigh.fit(train_X, train_y)
neigh_preds_train = neigh.predict(train_X)
neigh_preds_train = (neigh_preds_train*101 - train_y)/100  # need to remove self
neigh_preds_test = neigh.predict(test_X)
print("KNN Pred error: {:.4f}".format(normalized_gini(test_y, neigh_preds_test)))

# XGBoost preds
print("Running XGB regressor")
params = pd.DataFrame({
    "objective": "reg:linear",
    "eta": [0.04, 0.03, 0.03, 0.03, 0.02],
    "min_child_weight": 5,
    "subsample": [1, 0.9, 0.95, 1, 0.6],
    "colsample_bytree": [0.7, 0.6, 0.65, 0.6, 0.85],
    "max_depth": [8, 7, 9, 10, 10],
    "eval_metric": "auc",
    "scale_pos_weight": 1,
    "silent": 1
})

xgtrain = xgb.DMatrix(train_X, label=train_y)
xgval = xgb.DMatrix(test_X, label=test_y)
watchlist = [(xgval, 'val')]

model = xgb.train(params.iloc[0,:].to_dict(), xgtrain, num_boost_round = 3000,
                  evals = watchlist,
                  feval = gini_eval,
                  verbose_eval = False,
                  early_stopping_rounds=100)

xgb_preds_train = model.predict(xgtrain, ntree_limit=model.best_iteration)
xgb_preds_test = model.predict(xgval, ntree_limit=model.best_iteration)
print("XGB Pred error: {:.4f}".format(normalized_gini(test_y, xgb_preds_test)))

# Stacked
print("Running LR on stacked features")
stacked_train_X = np.hstack((train_X, np.vstack((xgb_preds_train, neigh_preds_train)).T))
stacked_test_X = np.hstack((test_X, np.vstack((xgb_preds_test, neigh_preds_test)).T))



bclf = LinearRegression() #(fit_intercept=True)
bclf.fit(stacked_train_X, train_y)

stacked_test_pred = bclf.predict(stacked_test_X)
print("Stacked Pred error: {:.4f}".format(normalized_gini(test_y, stacked_test_pred)))

print(mean_squared_error(neigh_preds_test, test_y))
print(mean_squared_error(xgb_preds_test, test_y))
print(mean_squared_error(stacked_test_pred, test_y))

plt.scatter(xgb_preds_train, train_y, s=5, c='g')
plt.scatter(neigh_preds_train, train_y, s=5, c='r')

