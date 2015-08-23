from sklearn import preprocessing
from sklearn.cross_validation import KFold
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import xgboost as xgb

from gini import normalized_gini, gini_eval
from dataset import get_data
import time

dat_x, dat_y_orig, lb_x, lb_ind = get_data()

dat_y = dat_y_orig ** 0.75
dat_x_orig = dat_x
#dat_y_orig = dat_y
lb_x_orig = lb_x

#scaler = preprocessing.StandardScaler().fit(dat_x_orig)
#s_dat_x = scaler.transform(dat_x_orig)
#s_lb_x = scaler.transform(lb_x_orig)


params = pd.DataFrame({
    "objective": "reg:linear",
    "eta": 0.005, #[0.04, 0.03, 0.03, 0.03, 0.02],
    "min_child_weight": 5,
    "subsample": [0.8, 0.9, 0.95, 0.7, 0.6],
    "colsample_bytree": [0.7, 0.6, 0.65, 0.6, 0.85],
    "max_depth": [10, 10, 10, 10, 10],
    "eval_metric": "auc",
    "scale_pos_weight": 1,
    "silent": 1
})


total_val_error = 0.0
total_runs = 0.0
total_lb_preds = np.repeat(0.0, lb_ind.shape[0])

for N in range(3):
    print("Starting N=%d" %(N))
    print("%s" % (time.ctime()))
    kf = KFold(n=dat_x.shape[0], n_folds=10, shuffle=True, random_state=107 + N)
    for train_index, test_index in kf:
        print("starting fold run %d" % (total_runs + 1))
        print("%s" % (time.ctime()))

        train_X = dat_x.iloc[train_index, :]
        train_y = dat_y[train_index]
        test_X = dat_x.iloc[test_index, :]
        test_y = dat_y[test_index]

        #s_train_x = s_dat_x[train_index, :]
        #s_cv_x = s_dat_x[test_index, :]
        #neigh = NearestNeighbors(11).fit(s_train_x[:,:])
        #train_distances, train_indices = neigh.kneighbors(s_train_x[:, :])
        #cv_distances, cv_indices = neigh.kneighbors(s_cv_x[:, :])

        #train_X = np.hstack( (train_X, train_distances[:,1].reshape(-1,1)))
        #test_X = np.hstack( (test_X, cv_distances[:,0].reshape(-1,1)))

        #train_X = np.hstack( (train_X, train_distances[:,10].reshape(-1,1)))
        #test_X = np.hstack( (test_X, cv_distances[:,9].reshape(-1,1)))


        print(train_X.shape)
        # train/validate/predict
        xgtrain = xgb.DMatrix(train_X, label=train_y)
        xgval = xgb.DMatrix(test_X, label=test_y)
        watchlist = [(xgval, 'val')]
        xglb = xgb.DMatrix(lb_x)

        for i in range(5):
            model = xgb.train(params.iloc[i,:].to_dict(), xgtrain, num_boost_round = 5000,
                              evals = watchlist,
                              feval = gini_eval,
                              verbose_eval = False,
                              early_stopping_rounds=100)

            val_error = normalized_gini(test_y, model.predict(xgval, ntree_limit=model.best_iteration))
            total_val_error += val_error
            total_runs += 1
            print("Validation Sample Score: {:.10f} (normalized gini).".format(val_error))
            print("Avg Validation Score: {:.10f} (normalized gini).".format(total_val_error / total_runs))

            total_lb_preds += model.predict(xglb, ntree_limit=model.best_iteration)
            time.sleep(10)
        print("Completed %d runs." % total_runs)
        total_lb_preds.tofile('metafeatures/total_lb_preds_%s.dat' % (str(total_runs)))
        time.sleep(20)
    time.sleep(120)

total_lb_preds /= total_runs

submission = pd.DataFrame({"Id": lb_ind, "Hazard": total_lb_preds})
submission = submission.set_index('Id')
#submission.to_csv('../subm/KNN_feat_xgboost_2.csv')

