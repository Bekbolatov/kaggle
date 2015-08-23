
# coding: utf-8

# In[57]:

import pandas as pd
from sklearn.cluster import KMeans
import xgboost as xgb
from sklearn.cross_validation import train_test_split, KFold

from gini import normalized_gini, gini_eval
from dataset import get_data

from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity, KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import time


# to try later:
#  try increasing the size of cv - since it is used to tune combinations (in LR)
#  different y-transformation
#  min-child 5->50?
#  could it be just the single best parameters for all level 1 XGBs?
#  add feature KNN: avg label of some neighbors, distances to some neighbors
#  use classes
#  can use something other than LR for blending?

# params = pd.DataFrame({
#     "objective": "reg:linear",
#     "eta": [0.04, 0.03, 0.03, 0.03, 0.02],
#     "min_child_weight": 50,
#     "subsample": [1, 0.9, 0.95, 1, 1],  # changed last
#     "colsample_bytree": [0.7, 0.6, 0.65, 0.6, 0.6],
#     "max_depth": [8, 7, 8, 7, 7],   #changed last two from 10 to 7
#     "eval_metric": "auc",
#     "scale_pos_weight": 1,
#     "silent": 1
# })
params = pd.DataFrame({
    "objective": "reg:linear",
    "eta": [0.04, 0.03, 0.03, 0.03, 0.02],
    "min_child_weight": 50,
    "subsample": [1, 0.9, 0.95, 1, 0.6],
    "colsample_bytree": [0.7, 0.6, 0.65, 0.6, 0.85],
    "max_depth": [8, 7, 9, 10, 10],
    "eval_metric": "auc",
    "scale_pos_weight": 1,
    "silent": 1
})

def evaluate(true_y, pred_y, label):
    mse = sum(np.power( pred_y - true_y, 2 ))/true_y.shape[0]
    gini = normalized_gini(true_y, pred_y)
    print("%s: Gini=%0.5f, MSE=%0.3f, "%(label, gini, mse))
    return (gini, mse)


dat_x_orig, dat_y_orig, lb_x_orig, lb_ind_orig = get_data()

dat_x = dat_x_orig
dat_y = dat_y_orig ** 0.75
lb_x = lb_x_orig
lb_ind = lb_ind_orig

RUNS = 2 ######### 10
MODELS = 5


lb_blend_y_all = np.repeat(0.0, lb_ind.shape[0])
cv_errors_all = np.empty([1, 2])
cv_errors_blends = np.empty([1, 2])

run_number = 0
kf = KFold(n=dat_x.shape[0], n_folds=5, shuffle=True, random_state=2187)
for seen_index, cv_index in kf:
    run_number = run_number + 1
    print("\n =================  run_number=%d  ================ [%s]\n" %(run_number, time.ctime()))

    train_x = dat_x.iloc[seen_index]
    train_y = dat_y[seen_index]
    cv_x = dat_x.iloc[cv_index]
    cv_y = dat_y[cv_index]

    # DATA FOR XGB
    xgb_train = xgb.DMatrix(train_x, label=train_y)
    xgb_cv = xgb.DMatrix(cv_x, label=cv_y)
    xgb_lb = xgb.DMatrix(lb_x)
    watchlist = [(xgb_cv, 'cv')]

    cv_blend_x = np.empty([1, cv_x.shape[0]])
    lb_blend_x = np.empty([1, lb_x.shape[0]])
    cv_errors = np.empty([1, 2])

    for model_number in range(MODELS):
        model = xgb.train(params.iloc[model_number].to_dict(), xgb_train, num_boost_round = 3000,
                          evals = watchlist,
                          feval = gini_eval,
                          verbose_eval = False,
                          early_stopping_rounds=50)

        cv_y_preds = model.predict(xgb_cv, ntree_limit=model.best_iteration)
        lb_y_preds = model.predict(xgb_lb, ntree_limit=model.best_iteration)

        cv_errors = np.vstack((cv_errors, np.asarray(evaluate(cv_y, cv_y_preds, "cv #%d" % model_number))))

        cv_blend_x = np.vstack( (cv_blend_x, cv_y_preds))
        lb_blend_x = np.vstack( (lb_blend_x, lb_y_preds))

    cv_blend_x = cv_blend_x[1:].T
    lb_blend_x = lb_blend_x[1:].T
    cv_errors = cv_errors[1:].T

    print("(Avg cv1) Gini: %0.5f MSE: %0.5f" %(np.mean(cv_errors[0]), np.mean(cv_errors[1])))

    lr1 = LinearRegression()
    lr1.fit(cv_blend_x, cv_y)
    cv_blend_y = lr1.predict(cv_blend_x)
    lb_blend_y = lr1.predict(lb_blend_x)
    cv_errors_blends = np.vstack((cv_errors_blends, evaluate(cv_y, cv_blend_y, "cv_blend_y")))
    cv_errors_all = np.vstack((cv_errors_all, cv_errors.T))

    lb_blend_y_all += lb_blend_y

    if run_number == RUNS:
        break

cv_errors_all = cv_errors_all[1:].T
#print(cv_errors_all.T)
cv_errors_blends = cv_errors_blends[1:].T
#print(cv_errors_blends.T)

print("Avg cv Gini:  pre-blend=%0.5f, post-blend=%0.5f" % (np.mean(cv_errors_all[0]), np.mean(cv_errors_blends[0])))
print("Avg cv MSE:   pre-blend=%0.3f,  post-blend=%0.3f" % (np.mean(cv_errors_all[1]), np.mean(cv_errors_blends[1])))

print("lb_blend_y.shape %s"% str(lb_blend_y.shape))
lb_blend_y_all /= (MODELS*run_number)

submission = pd.DataFrame({"Id": lb_ind, "Hazard": lb_blend_y_all})
submission = submission.set_index('Id')
submission.to_csv('../subm/Aug21_pow75_minchild50__orig_10runs__TEST_4.csv')

print("\n =================  END  ================ [%s]\n" %(time.ctime()))

# Trying with 5 runs on Aug 21 AM
# base
# Avg cv Gini:  pre-blend=0.37848, post-blend=0.38141
# Avg cv MSE:   pre-blend=14.002,  post-blend=13.930
# LB: 0.381599

# + label pow 0.75
# Avg cv Gini:  pre-blend=0.36284, post-blend=0.36499
# Avg cv MSE:   pre-blend=3.215,  post-blend=3.203
# LB: 0.384230

# + min child 50
# Avg cv Gini:  pre-blend=0.36546, post-blend=0.36910
# Avg cv MSE:   pre-blend=3.204,  post-blend=3.193
# LB: 0.384988

# use orig params (Behroz)
# Avg cv Gini:  pre-blend=0.36565, post-blend=0.36909
# Avg cv MSE:   pre-blend=3.205,  post-blend=3.193
# LB: 0.385928

# Run the last with all 10x runs instead of 5x
# Avg cv Gini:  pre-blend=0.37033, post-blend=0.37385
# Avg cv MSE:   pre-blend=3.220,  post-blend=3.205
# LB: 0.387498

