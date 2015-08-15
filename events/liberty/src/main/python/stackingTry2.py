from sklearn.cross_validation import train_test_split, KFold

from sklearn import preprocessing
import pandas as pd
from dataset import get_data
from metaregressor import meta_fit
import numpy as np
import time
import warnings
from sklearn.utils.validation import NonBLASDotWarning

warnings.simplefilter('always', NonBLASDotWarning)

print("Loading datasets")
dat_x_orig, dat_y_orig, lb_x_orig, lb_ind = get_data()

scaler = preprocessing.StandardScaler().fit(dat_x_orig)
dat_x = scaler.transform(dat_x_orig)
lb_x = scaler.transform(lb_x_orig)
dat_y = dat_y_orig ** 0.75

run_id = "0"
n_folds = 10
fold_number = 0
kf = KFold(n=dat_x.shape[0], n_folds=n_folds, shuffle=True, random_state=1007)
cv_pred_error = 0.0
subm_y = np.zeros((lb_x.shape[0], 1))
for train_index, cv_index in kf:
    print (time.strftime("%H:%M:%S"))
    print("\n ==================  Main folds: %d/%d  ==============\n" % (fold_number, n_folds))
    cv_pred_error_fold, subm_y_fold = meta_fit(dat_x, dat_y, train_index, cv_index, lb_x,
                                               main_fold_id=run_id + str(fold_number))
    cv_pred_error += cv_pred_error_fold
    subm_y += subm_y_fold
    fold_number += 1
    break # try only one

subm_y /= fold_number
cv_pred_error /= fold_number

print("\nAverage CV error: %0.6f" % (cv_pred_error))

submission = pd.DataFrame({"Id": lb_ind, "Hazard": subm_y})
submission = submission.set_index('Id')
submission.to_csv('../subm/stacked_KNN_XGB_run' + run_id + '.csv')


