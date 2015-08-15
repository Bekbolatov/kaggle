from sklearn.cross_validation import KFold
import numpy as np
import pandas as pd
import xgboost as xgb

from gini import normalized_gini, gini_eval
from dataset import get_data

dat_x, dat_y_orig, lb_x, lb_ind = get_data()

dat_y = dat_y_orig ** 0.75

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


total_val_error = 0.0
total_runs = 0.0
total_lb_preds = np.repeat(0.0, lb_ind.shape[0])

for i in range(5):
    kf = KFold(n=dat_x.shape[0], n_folds=10, shuffle=True, random_state=101 + i)
    for train_index, test_index in kf:
        train_X = dat_x.iloc[train_index, :]
        train_y = dat_y[train_index]
        test_X = dat_x.iloc[test_index, :]
        test_y = dat_y[test_index]

        # train/validate/predict
        xgtrain = xgb.DMatrix(train_X, label=train_y)
        xgval = xgb.DMatrix(test_X, label=test_y)
        watchlist = [(xgval, 'val')]

        model = xgb.train(params.iloc[i,:].to_dict(), xgtrain, num_boost_round = 3000,
                          evals = watchlist,
                          feval = gini_eval,
                          verbose_eval = False,
                          early_stopping_rounds=100)

        val_preds = model.predict(xgval, ntree_limit=model.best_iteration)
        val_error = normalized_gini(test_y, val_preds)
        total_val_error += val_error
        total_runs += 1
        print("Validation Sample Score: {:.10f} (normalized gini).".format(val_error))
        print("Avg Validation Score: {:.10f} (normalized gini).".format(total_val_error / total_runs))

        xglb = xgb.DMatrix(lb_x)
        lb_preds = model.predict(xglb, ntree_limit=model.best_iteration)
        total_lb_preds += total_lb_preds

total_lb_preds /= total_runs
#   end

submission = pd.DataFrame({"Id": lb_indices, "Hazard": total_lb_preds})
submission = submission.set_index('Id')
submission.to_csv('../subm/xgboost_in_python2.csv')
