import numpy as np
from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn import preprocessing
import pandas as pd
import theano
import xgboost as xgb
import time

from lasagne_helper import on_epoch_finished

from gini import normalized_gini, gini_eval


from nolearn.lasagne import NeuralNet, TrainSplit

import warnings
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.updates import nesterov_momentum
from lasagne.objectives import squared_error
from lasagne.nonlinearities import sigmoid, rectify, softmax, linear
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.utils.validation import NonBLASDotWarning
import theano
import numpy as np
from dataset import get_data

warnings.simplefilter('always', NonBLASDotWarning)
from gini import normalized_gini



def add_metafeatures_KNN(dat_x, dat_y, cv_x, lb_x, base_ncols, n_neighbors=101):
    print("KNN regressor with %d neighbors" % (n_neighbors))
    print (time.strftime("%H:%M:%S"))
    ncols = dat_x.shape[1]

    model = KNeighborsRegressor(n_neighbors=n_neighbors, weights='uniform', p=1)
    model.fit(dat_x[:, :base_ncols], dat_y)
    dat_metafeatures = model.predict(dat_x[:, :base_ncols]) - dat_y/n_neighbors
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

class feature_gen_XGB:

    def __init__(self, xgb_params={}):
        self.xgb_params = xgb_params

    def fit(self, train_x, train_y, cv_x, cv_y):
        xgb_train = xgb.DMatrix(train_x, label=train_y)
        xgb_cv = xgb.DMatrix(cv_x, label=cv_y)
        watchlist = [(xgb_cv, 'val')]
        model = xgb.train(self.xgb_params, xgb_train, num_boost_round=3000,
                          evals=watchlist,
                          feval=gini_eval,
                          verbose_eval=False,
                          early_stopping_rounds=50)
        return model

    def predict(self, model, x):
        y = model.predict(xgb.DMatrix(x), ntree_limit=model.best_iteration)
        return y


class feature_gen_NN:

    def __init__(self, learning_rate=0.02):
        self.learning_rate = learning_rate

    def predict(self, model, x):
        preds = model.predict(np.asarray(x, dtype = theano.config.floatX))[:, 0]
        return preds

    def fit(self, train_x, train_y, cv_x, cv_y):
        train_x = np.asarray(train_x, dtype = theano.config.floatX)
        train_y = np.asarray(train_y, dtype = theano.config.floatX)

        model = self.nn_constructor(train_x.shape[1])
        model.fit(train_x, train_y)

        return model

    def nn_constructor(self, num_features):
        layers0 = [
            ('input', InputLayer),
            ('dropout0', DropoutLayer),
            ('hidden1', DenseLayer),
            ('dropout1', DropoutLayer),
            ('hidden2', DenseLayer),
            ('output', DenseLayer)]

        net0 = NeuralNet(
            layers=layers0,
            input_shape=(None, num_features),
            dropout0_p=0.3,
            hidden1_num_units=20, #20,
            hidden1_nonlinearity=sigmoid,
            dropout1_p=0.2,
            hidden2_num_units=24, #24,
            hidden2_nonlinearity=sigmoid,
            output_num_units=1,
            output_nonlinearity=linear,
            on_epoch_finished=[on_epoch_finished],
            objective_loss_function=squared_error,
            update=nesterov_momentum,
            update_learning_rate=self.learning_rate,
            update_momentum=0.9,
            train_split=TrainSplit(eval_size=0.1),
            verbose=1,
            regression=True,
            max_epochs=200)
        return net0


def add_metafeature_from_folds(dat_x, dat_y, cv_x, lb_x, base_ncols, feature_gen, n_folds=5, random_state=101):
    print (time.strftime("%H:%M:%S"))

    dst_dat_x = np.empty((0, dat_x.shape[1] + 1))
    dst_dat_y = np.empty(0)
    dst_cv_x_metafeatures = np.zeros(cv_x.shape[0])
    dst_lb_x_metafeatures = np.zeros(lb_x.shape[0])

    cv_x_cropped = cv_x[:, :base_ncols]
    lb_x_cropped = lb_x[:, :base_ncols]

    print("generating metafeatures using %d folds." % (n_folds))
    fold_number = 1
    kf = KFold(n=dat_x.shape[0], n_folds=n_folds, shuffle=True, random_state=random_state)
    for src_index, dst_index in kf:
        print("\n   [metafeatures gen fold %d/%d]\n" %(fold_number, n_folds))
        src_x_fold = dat_x[src_index, :]
        src_y_fold = dat_y[src_index]
        dst_x_fold = dat_x[dst_index, :]
        dst_y_fold = dat_y[dst_index]

        src_x_fold_cropped = src_x_fold[:, :base_ncols]
        dst_x_fold_cropped = dst_x_fold[:, :base_ncols]

        model = feature_gen.fit(src_x_fold_cropped, src_y_fold, dst_x_fold_cropped, dst_y_fold)
        dst_x_pred_fold = feature_gen.predict(model, dst_x_fold_cropped)
        print("pred error: {:.4f}".format(normalized_gini(dst_y_fold, dst_x_pred_fold)))
        cv_x_pred_fold = feature_gen.predict(model, cv_x_cropped)
        lb_x_pred_fold = feature_gen.predict(model, lb_x_cropped)

        dst_dat_x = np.vstack((dst_dat_x, np.hstack((dst_x_fold, dst_x_pred_fold.reshape(-1, 1)))))
        dst_dat_y = np.hstack((dst_dat_y, dst_y_fold))
        dst_cv_x_metafeatures += cv_x_pred_fold
        dst_lb_x_metafeatures += lb_x_pred_fold

        print (time.strftime("%H:%M:%S"))
        fold_number += 1

    dst_cv_x = np.hstack((cv_x, (dst_cv_x_metafeatures/n_folds).reshape(-1, 1)))
    dst_lb_x = np.hstack((lb_x, (dst_lb_x_metafeatures/n_folds).reshape(-1, 1)))

    return (dst_dat_x, dst_dat_y, dst_cv_x, dst_lb_x)
