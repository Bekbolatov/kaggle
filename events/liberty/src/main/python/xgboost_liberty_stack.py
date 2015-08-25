
# coding: utf-8

# In[57]:

import pandas as pd
from sklearn.cluster import KMeans
import xgboost as xgb
from sklearn.cross_validation import train_test_split, KFold

from gini import normalized_gini, gini_eval
from dataset import LibertyEncoder

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
import os
import sys

param_inter = []
param_drop = []
if len(sys.argv) == 2:
    m_param_inter, m_param_drop = sys.argv[1].split(';')
    m_param_inter = m_param_inter.split(',')
    if m_param_inter[0]:
        param_inter = [ tuple(map(int, kv.split(':'))) for kv in m_param_inter]
    if m_param_drop and m_param_drop[0]:
        param_drop = map(int, m_param_drop.split(','))

print(param_inter)
print(param_drop)

USAGE="python hla.py 5:6,7:8,5:22;4,6,7"

LOCATION = os.getenv('DATA_LOCATION', '/Users/rbekbolatov/data/kaggle/liberty')
OUTPUT_LOCATION = os.getenv('OUTPUT_LOCATION', '/Users/rbekbolatov/data/kaggle/liberty')

# to try later:
#  Into Libffm classes
#  Into hashed space - add interactions
#  Something about order (Devin and Qinchen)
#  OHE with interactions?
#  Try lower eta
#  + try eval metric  rmse instead of auc?
#  + different y-transformation
#  + min-child 5->50->60?
#  Combine T1_V7, T1_V8, T1_V12 when they are C
#  Add other interactions/drop?
#  try increasing the size of cv - since it is used to tune combinations (in LR)
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
    "eta": [0.04, 0.03, 0.03, 0.03, 0.02, 0.02],
    "min_child_weight": 50,
    "subsample": [1, 0.9, 0.95, 1, 0.6, 0.8],
    "colsample_bytree": [0.7, 0.6, 0.65, 0.6, 0.85, 0.6],
    "max_depth": [8, 7, 9, 10, 10, 5],
    "eval_metric": "auc",
    "scale_pos_weight": 1,
    "silent": 1
})

def evaluate(true_y, pred_y, label):
    mse = sum(np.power( pred_y - true_y, 2 ))/true_y.shape[0]
    gini = normalized_gini(true_y, pred_y)
    print("%s: Gini=%0.5f, MSE=%0.3f, "%(label, gini, mse))
    return (gini, mse)


data = LibertyEncoder(loc=LOCATION)
dat_x_orig, dat_y_orig, dat_y_raw_orig, lb_x_orig, lb_ind_orig = data.get_orig_data_copy()
dat_x_orig, lb_x_orig = data.transform('renat', param_inter, param_drop, dat_y_orig, dat_x_orig, lb_x_orig)


dat_x = dat_x_orig
dat_y = dat_y_orig
dat_y_raw = dat_y_raw_orig
lb_x = lb_x_orig
lb_ind = lb_ind_orig

RUNS = 1
MODELS = 1
FOLDS = 2
ITERATIONS = (RUNS/FOLDS + 1)

lb_blend_y_all = np.repeat(0.0, lb_ind.shape[0])
cv_errors_all = np.empty([1, 2])
cv_errors_blends = np.empty([1, 2])

run_number = 0
for iteration in range(ITERATIONS):
    kf = KFold(n=dat_x.shape[0], n_folds=FOLDS, shuffle=True, random_state=2187 + 87*iteration)
    for seen_index, cv_index in kf:
        run_number = run_number + 1
        print("\n =================  run_number=%d  ================ [%s]\n" %(run_number, time.ctime()))

        train_x = dat_x[seen_index, :]
        train_y = dat_y[seen_index]
        train_y_raw = dat_y_raw[seen_index]

        cv_x = dat_x[cv_index, :]
        cv_y = dat_y[cv_index]
        cv_y_raw = dat_y_raw[cv_index]

        # train_x, cv_x, lb_x = data.transform('qinchen', train_y, train_x, cv_x, lb_x)
        # train_x, cv_x, lb_x = data.transform('renat', train_y, train_x, cv_x, lb_x)

        # DATA FOR XGB
        xgb_train = xgb.DMatrix(train_x, label=train_y)
        xgb_cv = xgb.DMatrix(cv_x, label=cv_y)
        xgb_cv_val = xgb.DMatrix(cv_x, label=cv_y_raw)
        xgb_lb = xgb.DMatrix(lb_x)
        watchlist = [(xgb_cv_val, 'cv')]

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

            cv_errors = np.vstack((cv_errors, np.asarray(evaluate(cv_y_raw, cv_y_preds, "cv #%d" % model_number))))

            cv_blend_x = np.vstack( (cv_blend_x, cv_y_preds))
            lb_blend_x = np.vstack( (lb_blend_x, lb_y_preds))

        cv_blend_x = cv_blend_x[1:].T
        lb_blend_x = lb_blend_x[1:].T
        cv_errors = cv_errors[1:].T

        print("\ncv_prebl_y: Gini=%0.5f, MSE=%0.3f" %(np.mean(cv_errors[0]), np.mean(cv_errors[1])))

        lr1 = LinearRegression()
        lr1.fit(cv_blend_x, cv_y)
        cv_blend_y = lr1.predict(cv_blend_x)
        lb_blend_y = lr1.predict(lb_blend_x)
        cv_errors_blends = np.vstack((cv_errors_blends, evaluate(cv_y_raw, cv_blend_y, "cv_blend_y")))
        cv_errors_all = np.vstack((cv_errors_all, cv_errors.T))

        lb_blend_y_all += lb_blend_y

        if run_number == RUNS:
            break
    if run_number == RUNS:
        break

cv_errors_all = cv_errors_all[1:].T
#print(cv_errors_all.T)
cv_errors_blends = cv_errors_blends[1:].T
#print(cv_errors_blends.T)

print("\n----------------------------------------------\n")
print("Avg cv Gini:  pre-blend=%0.5f, post-blend=%0.5f" % (np.mean(cv_errors_all[0]), np.mean(cv_errors_blends[0])))
print("Avg cv MSE:   pre-blend=%0.3f,  post-blend=%0.3f" % (np.mean(cv_errors_all[1]), np.mean(cv_errors_blends[1])))

lb_blend_y_all /= (MODELS*run_number)

submission = pd.DataFrame({"Id": lb_ind, "Hazard": lb_blend_y_all})
submission = submission.set_index('Id')
#submission.to_csv('../subm/Aug24_CheckRefactor__raw_y_perfold_5.csv')

results = pd.DataFrame(cv_errors_all)
print(results)
results['Id'] = np.arange(run_number*MODELS)
results = results.set_index('Id')
results.to_csv(OUTPUT_LOCATION + '/results.csv')

results_blends = pd.DataFrame(cv_errors_blends)
print(results_blends)
results_blends['Id'] = np.arange(run_number)
results_blends = results_blends.set_index('Id')
results_blends.to_csv(OUTPUT_LOCATION + '/results_blended.csv')

print("\n =================  END  ================ [%s]\n" %(time.ctime()))

# Trying with 5 runs
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


#  [ Aug 22 Sat:  Qingchen -  back to Hazard means ]
# Try using the new feat with 5x runs
# Avg cv Gini:  pre-blend=0.37327, post-blend=0.37660
# Avg cv MSE:   pre-blend=3.214,  post-blend=3.202
# LB: 0.390281

# Now go back to no transform of label y (before it was x ^ 3/4)
# Avg cv Gini:  pre-blend=0.38940, post-blend=0.39264
# Avg cv MSE:   pre-blend=14.149,  post-blend=14.092
# LB: 0.387785

# Okay, unresolved why label ** 0.75 does that -  problem is that I want to trust CV
# Again going back to label ** 0.75
# Avg cv Gini:  pre-blend=0.37575, post-blend=0.37993
# Avg cv MSE:   pre-blend=3.208,  post-blend=3.192
# LB:  0.390186

# Was a bit surprised to see no improvement with 10x vs 5x.
# Maybe problem was that blending was done on too small a cv set
# Trying 5-fold for blending, but running everything 2 times, with different seeds
# Avg cv Gini:  pre-blend=0.37429, post-blend=0.37797
# Avg cv MSE:   pre-blend=3.208,  post-blend=3.196
# LB:  0.390023

#################  Figured out what is going on with label ** 0.75 transformation - it is good - I was just using transformed labels to calculate scores.
# Now checking if some other transform is better, with confidence
#Avg cv Gini:  pre-blend=0.40224, post-blend=0.40494  # 0.75
#Avg cv Gini:  pre-blend=0.40227, post-blend=0.40637  # 0.65
#Avg cv Gini:  pre-blend=0.40255, post-blend=0.40699  # 0.50  (**)
#Avg cv Gini:  pre-blend=0.40104, post-blend=0.40536  # 0.25
#Avg cv Gini:  pre-blend=0.39971, post-blend=0.40409  # log

# Also try eta = 0.01
#Avg cv Gini:  pre-blend=0.40388, post-blend=0.40551

# Trying submission with y <- y ** 0.50 transform
# Avg cv Gini:  pre-blend=0.39278, post-blend=0.39677  *
# Avg cv MSE:   pre-blend=20.356,  post-blend=20.262
# LB: 0.390334

# Try removing all those 2nd order Qinchen interactions
# Avg cv Gini:  pre-blend=0.38650, post-blend=0.38908

# Without last two interactions:
# < BAD Avg cv Gini:  pre-blend=0.39194, post-blend=0.39574 >
# Avg cv Gini:  pre-blend=0.39305, post-blend=0.39752
# LB: 0.389209

# Should be same as 2x (**) above
# Avg cv Gini:  pre-blend=0.40244, post-blend=0.40708

##  Aug 24 3am
# Should be identical:
# Avg cv Gini:  pre-blend=0.39233, post-blend=0.39622
# LB:







