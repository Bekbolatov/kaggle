import numpy as np
from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn import preprocessing
import pandas as pd
import xgboost as xgb
import time

from gini import normalized_gini, gini_eval


def add_metafeatures(dat_x, dat_y, cv_x, lb_x, n_folds=5, random_state=101):
    n_generators = 2

    dst_dat_x = np.empty((0, dat_x.shape[1] + n_generators))
    dst_dat_y = np.empty(0)
    dst_cv_x_metafeatures = np.zeros((cv_x.shape[0], n_generators))
    dst_lb_x_metafeatures = np.zeros((lb_x.shape[0], n_generators))

    print("Generating metafeatures using %d folds." % (n_folds))
    print (time.strftime("%H:%M:%S"))
    fold_number = 1
    kf = KFold(n=dat_x.shape[0], n_folds=n_folds, shuffle=True, random_state=random_state)
    for src_index, dst_index in kf:
        print("\n   [Fold %d/%d]" %(fold_number, n_folds))
        src_X_fold = dat_x[src_index, :]
        src_y_fold = dat_y[src_index]
        dst_X_fold = dat_x[dst_index, :]
        dst_y_fold = dat_y[dst_index]

        #  KNN Regressor preds
        print("+ KNN regressor")
        neigh_model = KNeighborsRegressor(n_neighbors=101, weights='distance', p=1)
        neigh_model.fit(src_X_fold, src_y_fold)
        dst_KNN_metafeatures = neigh_model.predict(dst_X_fold)
        print("  Fold KNN pred error: {:.4f}".format(normalized_gini(dst_y_fold, dst_KNN_metafeatures)))
        print("  Calculating KNN metafeatures")
        print("  ... cv")
        neigh_cv_metafeatures = neigh_model.predict(cv_x)
        print("  ... lb")
        neigh_lb_metafeatures = neigh_model.predict(lb_x)

        # XGBoost preds
        print("+ XGB regressor")
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

        xgb_src_fold = xgb.DMatrix(src_X_fold, label=src_y_fold)
        xgb_dst_fold = xgb.DMatrix(dst_X_fold, label=dst_y_fold)
        watchlist = [(xgb_dst_fold, 'val')]

        xgb_model = xgb.train(params.iloc[0, :].to_dict(), xgb_src_fold, num_boost_round=3000,
                          evals=watchlist,
                          feval=gini_eval,
                          verbose_eval=False,
                          early_stopping_rounds=100)

        dst_XGB_metafeatures = xgb_model.predict(xgb_dst_fold, ntree_limit=xgb_model.best_iteration)
        print("  Fold XGB pred error: {:.4f}".format(normalized_gini(dst_y_fold, dst_XGB_metafeatures)))

        print("  Calculating XGB metafeatures")
        print("  ... cv")
        xgb_cv_metafeatures = xgb_model.predict(xgb.DMatrix(cv_x))
        print("  ... lb")
        xgb_lb_metafeatures = xgb_model.predict(xgb.DMatrix(lb_x))

        # Merge metafeatures
        print("  Merging metafeatures")
        dst_dat_x = np.vstack((dst_dat_x, np.hstack((dst_X_fold, np.vstack((dst_KNN_metafeatures, dst_XGB_metafeatures)).T))))
        dst_dat_y = np.hstack( (dst_dat_y, dst_y_fold))
        dst_cv_x_metafeatures += np.vstack( (neigh_cv_metafeatures, xgb_cv_metafeatures)).T
        dst_lb_x_metafeatures += np.vstack( (neigh_lb_metafeatures, xgb_lb_metafeatures)).T

        fold_number += 1

    dst_cv_x = np.hstack((cv_x, dst_cv_x_metafeatures/n_folds))
    dst_lb_x = np.hstack((lb_x, dst_lb_x_metafeatures/n_folds))

    print('Final shapes')
    print('dst_dat_x.shape = %s' % (str(dst_dat_x.shape)))
    print('dst_dat_y.shape = %s' % (str(dst_dat_y.shape)))
    print('dst_cv_x.shape = %s' % (str(dst_cv_x.shape)))
    print('dst_lb_x.shape = %s' % (str(dst_lb_x.shape)))

    print (time.strftime("%H:%M:%S"))

    return (dst_dat_x, dst_dat_y, dst_cv_x, dst_lb_x)
