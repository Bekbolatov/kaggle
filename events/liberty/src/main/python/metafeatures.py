import numpy as np
from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn import preprocessing
import pandas as pd
import theano
import xgboost as xgb
import time

from metaregressor import on_epoch_finished

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




def add_metafeatures_KNN(dat_x, dat_y, cv_x, lb_x, n_neighbors=101, skip_right=0):
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

def add_metafeatures_NN_class1(dat_x, dat_y, cv_x, lb_x, n_folds=5, random_state=101, skip_right=0):
    pass

def add_metafeatures_NN(dat_x, dat_y, cv_x, lb_x, n_folds=5, random_state=101, skip_right=0):
    print("NN regressor")
    print (time.strftime("%H:%M:%S"))
    ncols = dat_x.shape[1]

    random_state = random_state + 107

    dst_dat_x = np.empty((0, dat_x.shape[1] + 1))
    dst_dat_y = np.empty(0)
    dst_cv_x_metafeatures = np.zeros((cv_x.shape[0], 1))
    dst_lb_x_metafeatures = np.zeros((lb_x.shape[0], 1))

    def NeuralNetConstructor(num_features):
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

            dropout0_p=0.2,

            hidden1_num_units=70,
            hidden1_nonlinearity=sigmoid,

            dropout1_p=0.2,

            hidden2_num_units=70,
            hidden2_nonlinearity=sigmoid,

            output_num_units=1,
            output_nonlinearity=linear,

            on_epoch_finished=[on_epoch_finished],
            objective_loss_function=squared_error,
            update=nesterov_momentum,
            update_learning_rate=0.0005,
            update_momentum=0.9,
            train_split=TrainSplit(eval_size=0.1),
            verbose=1,
            regression=True,
            max_epochs=200)

        return net0

    print("Generating metafeatures using %d folds." % (n_folds))
    fold_number = 1
    kf = KFold(n=dat_x.shape[0], n_folds=n_folds, shuffle=True, random_state=random_state)
    for src_index, dst_index in kf:
        print("\n   [NN metafeatures gen fold %d/%d]\n" %(fold_number, n_folds))
        src_X_fold = dat_x[src_index, :]
        src_y_fold = dat_y[src_index]
        dst_X_fold = dat_x[dst_index, :]
        dst_y_fold = dat_y[dst_index]

        xgb_src_fold = xgb.DMatrix(src_X_fold[:, :(ncols - skip_right)], label=src_y_fold)
        xgb_dst_fold = xgb.DMatrix(dst_X_fold[:, :(ncols - skip_right)], label=dst_y_fold)
        watchlist = [(xgb_dst_fold, 'val')]

        scaler = preprocessing.StandardScaler().fit(src_X_fold[:, :(ncols - skip_right)])
        train_x = scaler.transform(src_X_fold[:, :(ncols - skip_right)])
        cv_x = scaler.transform(dst_X_fold[:, :(ncols - skip_right)])
        subm_x = scaler.transform(lb_x[:, :(ncols - skip_right)])

        train_x = np.asarray(train_x, dtype = theano.config.floatX)
        train_y = np.asarray(train_y, dtype = theano.config.floatX)

        network = NeuralNetConstructor(train_x.shape[1])
        network.fit(train_x, train_y)

        cv_y_preds = network.predict(np.asarray(cv_x, dtype = theano.config.floatX))
        subm_y_preds = network.predict(np.asarray(subm_x, dtype = theano.config.floatX))[:, 0]

        cv_pred_error = normalized_gini(cv_y, cv_y_preds)
        print("NN CV score: {:.6f} (normalized gini).".format(cv_pred_error))

        return (cv_pred_error, subm_y_preds)




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
