import pandas as pd
from sklearn.cluster import KMeans
import xgboost as xgb
from sklearn.cross_validation import train_test_split

from gini import normalized_gini, gini_eval
from dataset import get_data

from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity, KNeighborsRegressor

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import pylab as pl

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


dat_x, dat_y, lb_x, lb_ind = get_data()


dat_x['label'] = dat_y
dat_x = dat_x[dat_x['label'] < 6]
dat_y = np.asarray(dat_x['label'])
dat_x = dat_x.drop('label', axis=1)

train_index, test_index = train_test_split(range(dat_x.shape[0]), test_size=0.1, random_state=103)

train_x = dat_x.iloc[train_index, :]
train_y = dat_y[train_index]
cv_x = dat_x.iloc[test_index, :]
cv_y = dat_y[test_index]


# model 2
model = KNeighborsRegressor(n_neighbors=1, weights='distance', p=1)
model.fit(train_x, train_y)
cv_y_preds = model.predict(cv_x)


# evaluate

cv_error = normalized_gini(cv_y, cv_y_preds)
print("Validation Sample Score: {:.10f} (normalized gini).".format(cv_error))

preds = pd.DataFrame({"actual": cv_y, "pred": cv_y_preds})
preds.boxplot('pred', 'actual')




# show
max_cy = max(cv_y) * 1.1
pl.scatter(cv_y, cv_y_preds, s=1)
pl.xlim(0, 70)
pl.show()





# analyze
cv_all = np.hstack( (np.vstack((cv_y, cv_y_preds)).T, cv_x))
dat_all = np.hstack( ( dat_y.reshape(-1, 1), dat_x ))
dat_all_df = pd.DataFrame(dat_all)










