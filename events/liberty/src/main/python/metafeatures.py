import numpy as np
from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn import preprocessing
import pandas as pd
import xgboost as xgb
import time

from gini import normalized_gini, gini_eval


def add_metafeatures_KNN(dat_x, dat_y, cv_x, lb_x, n_neighbors=101, random_state=101, skip_right=0):
    print("KNN regressor with %d neighbors" % (n_neighbors))
    print (time.strftime("%H:%M:%S"))
    ncols = dat_x.shape[1]

    model = KNeighborsRegressor(n_neighbors=n_neighbors, weights='uniform', p=1)
    model.fit(dat_x[:, :(ncols - skip_right)], dat_y)
    dat_metafeatures = model.predict(dat_x[:, :(ncols - skip_right)]) - dat_y/n_neighbors
    print("KNN pred error: {:.4f}".format(normalized_gini(dat_y, dat_metafeatures)))

    print("Calculating KNN metafeatures")
    print("... cv")
    cv_metafeatures = model.predict(cv_x)
    print("... lb")
    lb_metafeatures = model.predict(lb_x)

    dst_dat_x = np.hstack((dat_x, dat_metafeatures.reshape(-1, 1)))
    dst_dat_y = dat_y
    dst_cv_x = np.hstack((cv_x, cv_metafeatures.reshape(-1, 1)))
    dst_lb_x = np.hstack((lb_x, lb_metafeatures.reshape(-1, 1)))

    print (time.strftime("%H:%M:%S"))
    return (dst_dat_x, dst_dat_y, dst_cv_x, dst_lb_x)


def add_metafeatures_XGB(dat_x, dat_y, cv_x, lb_x, n_folds=5, xgb_params={}, random_state=101, skip_right=0):
    print("XGB regressor")
    print (time.strftime("%H:%M:%S"))
    ncols = dat_x.shape[1]

    dst_dat_x = np.empty((0, dat_x.shape[1] + 1))
    dst_dat_y = np.empty(0)
    dst_cv_x_metafeatures = np.zeros((cv_x.shape[0], 1))
    dst_lb_x_metafeatures = np.zeros((lb_x.shape[0], 1))

    print("Generating metafeatures using %d folds." % (n_folds))
    fold_number = 1
    kf = KFold(n=dat_x.shape[0], n_folds=n_folds, shuffle=True, random_state=random_state)
    for src_index, dst_index in kf:
        print("\n   [XGB metafeatures gen fold %d/%d]\n" %(fold_number, n_folds))
        src_X_fold = dat_x[src_index, :]
        src_y_fold = dat_y[src_index]
        dst_X_fold = dat_x[dst_index, :]
        dst_y_fold = dat_y[dst_index]

        # XGBoost preds
        xgb_src_fold = xgb.DMatrix(src_X_fold[:, :(ncols - skip_right)], label=src_y_fold)
        xgb_dst_fold = xgb.DMatrix(dst_X_fold[:, :(ncols - skip_right)], label=dst_y_fold)
        watchlist = [(xgb_dst_fold, 'val')]

        xgb_model = xgb.train(xgb_params, xgb_src_fold, num_boost_round=3000,
                          evals=watchlist,
                          feval=gini_eval,
                          verbose_eval=False,
                          early_stopping_rounds=100)

        xgb_dat_metafeatures_fold = xgb_model.predict(xgb_dst_fold, ntree_limit=xgb_model.best_iteration)
        print("  Fold XGB pred error: {:.4f}".format(normalized_gini(dst_y_fold, xgb_dat_metafeatures_fold)))
        print("  Calculating XGB metafeatures")
        print("  ... cv")
        xgb_cv_metafeatures = xgb_model.predict(xgb.DMatrix(cv_x[:, :(ncols - skip_right)]))
        print("  ... lb")
        xgb_lb_metafeatures = xgb_model.predict(xgb.DMatrix(lb_x[:, :(ncols - skip_right)]))

        # Merge metafeatures
        print("  Merging metafeatures")
        dst_dat_x = np.vstack((dst_dat_x, np.hstack((dst_X_fold, xgb_dat_metafeatures_fold.reshape(-1, 1)))))
        dst_dat_y = np.hstack((dst_dat_y, dst_y_fold))
        dst_cv_x_metafeatures += xgb_cv_metafeatures.reshape(-1, 1)
        dst_lb_x_metafeatures += xgb_lb_metafeatures.reshape(-1, 1)

        print (time.strftime("%H:%M:%S"))
        fold_number += 1

    dst_cv_x = np.hstack((cv_x, dst_cv_x_metafeatures/n_folds))
    dst_lb_x = np.hstack((lb_x, dst_lb_x_metafeatures/n_folds))

    return (dst_dat_x, dst_dat_y, dst_cv_x, dst_lb_x)
