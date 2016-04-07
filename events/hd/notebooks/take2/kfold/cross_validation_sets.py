import os
import pandas as pd
from sklearn.cross_validation import StratifiedKFold

seed = 197
package_directory = os.path.dirname(os.path.abspath(__file__))
base_loc = package_directory
loc = '../../%s'
kfold_path_train = 'KFOLD_%d_TRAIN_%d.df'
kfold_path_test = 'KFOLD_%d_TEST_%d.df'


def create_folds(s=seed, n=5):
    queries = pd.read_pickle(base_loc + '/' + loc % 'FEATURES_WITH_TEXT_1.data')
    idx_train = pd.read_pickle(base_loc + '/' + loc % 'LABELS_TRAIN.df')
    idx_test = pd.read_pickle(base_loc + '/' + loc % 'LABELS_TEST.df')

    for (i, (train, test)) in enumerate(StratifiedKFold(idx_train['relevance'], n_folds=n, shuffle=True, random_state=s)):
        idx_train.iloc[train].to_pickle(base_loc + '/' + kfold_path_train %  (i, s))
        idx_train.iloc[test].to_pickle(base_loc + '/' + kfold_path_test % (i, s))


def get_fold(i, s=seed):
    train = pd.read_pickle(base_loc + '/' + kfold_path_train % (i, s))
    test = pd.read_pickle(base_loc + '/' + kfold_path_test % (i, s))
    return train, test

