import numpy as np
import pandas as pd
import xgboost as xgb

from gini import normalized_gini, gini_eval
from metafeatures import add_metafeatures

dataloc = "metafeatures/"

xgb_params = pd.DataFrame({
    "objective": "reg:linear",
    "eta": [0.04, 0.03, 0.03, 0.03, 0.02, 0.4],
    "min_child_weight": 5,
    "subsample": [1, 0.9, 0.95, 1, 0.6, 1],
    "colsample_bytree": [0.7, 0.6, 0.65, 0.6, 0.85, 0.7],
    "max_depth": [8, 7, 9, 10, 10, 6],
    "eval_metric": "auc",
    "scale_pos_weight": 1,
    "silent": 1
})

def meta_fit(dat_x, dat_y, train_index, cv_index, lb_x, main_fold_id, read_cached=True):
    train_x_raw = dat_x[train_index, :]
    train_y_raw = dat_y[train_index]
    cv_x_raw = dat_x[cv_index, :]
    cv_y = dat_y[cv_index]
    subm_x_raw = lb_x

    train_x_file = dataloc + "fold_" + main_fold_id +"_train_x.dat"
    train_y_file = dataloc + "fold_" + main_fold_id +"_train_y.dat"
    cv_x_file = dataloc + "fold_" + main_fold_id +"_cv_x.dat"
    subm_x_file = dataloc + "fold_" + main_fold_id +"_subm_x.dat"

    if read_cached:
        print("using cached metafeatures")
        train_x = np.reshape(np.fromfile(train_x_file), (-1, 113))
        train_y = np.fromfile(train_y_file)
        cv_x = np.reshape(np.fromfile(cv_x_file), (-1, 113))
        subm_x = np.reshape(np.fromfile(subm_x_file), (-1, 113))
    else:
        print("generating metafeatures")
        train_x, train_y, cv_x, subm_x = add_metafeatures(train_x_raw, train_y_raw, cv_x_raw, subm_x_raw, n_folds=5, n_neighbors=100, xgb_params=xgb_params.iloc[5,:].to_dict(), random_state=101)
        train_x.tofile(train_x_file)
        train_y.tofile(train_y_file)
        cv_x.tofile(cv_x_file)
        subm_x.tofile(subm_x_file)

    xgtrain = xgb.DMatrix(train_x, label=train_y)
    xgcv = xgb.DMatrix(cv_x, label=cv_y)
    watchlist = [(xgcv, 'cv')]

    xgb_subm_preds = np.zeros(subm_x.shape[0])
    num_models = 5
    for i in range(num_models):
        xgb_model = xgb.train(xgb_params.iloc[i,:].to_dict(), xgtrain, num_boost_round = 3000,
                          evals = watchlist,
                          feval = gini_eval,
                          verbose_eval = False,
                          early_stopping_rounds=100)

        xgb_cv_preds = xgb_model.predict(xgcv, ntree_limit=xgb_model.best_iteration)
        cv_pred_error = normalized_gini(cv_y, xgb_cv_preds)
        print("Validation Sample Score: {:.6f} (normalized gini).".format(cv_pred_error))

        xgb_subm_preds_fold = xgb_model.predict(xgb.DMatrix(subm_x), ntree_limit=xgb_model.best_iteration)
        xgb_subm_preds += xgb_subm_preds_fold
    xgb_subm_preds /= num_models


    return (cv_pred_error, xgb_subm_preds)
