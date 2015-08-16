import numpy as np
import pandas as pd
import xgboost as xgb

from gini import normalized_gini, gini_eval
from metafeatures import add_metafeatures_KNN, add_metafeatures_XGB


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


dataloc = "metafeatures/"


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
        train_y = np.fromfile(train_y_file) ** (4.0/3)   ## CHANGE
        cv_x = np.reshape(np.fromfile(cv_x_file), (-1, 113))
        subm_x = np.reshape(np.fromfile(subm_x_file), (-1, 113))
    else:
        print("generating metafeatures")
        train_x_1, train_y_1, cv_x_1, subm_x_1 = add_metafeatures_KNN(train_x_raw, train_y_raw, cv_x_raw, subm_x_raw, n_neighbors=100, random_state=101, skip_right=0)
        train_x, train_y, cv_x, subm_x = add_metafeatures_XGB(train_x_1, train_y_1, cv_x_1, subm_x_1, n_folds=5,  xgb_params=xgb_params.iloc[5,:].to_dict(), random_state=101, skip_right=1)

        train_x.tofile(train_x_file)
        train_y.tofile(train_y_file)
        cv_x.tofile(cv_x_file)
        subm_x.tofile(subm_x_file)

    nn_cv_pred_error, nn_subm_preds = regressor_NN(train_x, train_y, cv_x, cv_y, subm_x)
    xgb_cv_pred_error, xgb_subm_preds = regressor_XGB(train_x, train_y, cv_x, cv_y, subm_x)

    cv_pred_error = (xgb_cv_pred_error + nn_cv_pred_error)/2
    subm_preds = (xgb_subm_preds + nn_subm_preds)/2

    return (cv_pred_error, subm_preds)


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

def regressor_XGB(train_x, train_y, cv_x, cv_y, subm_x):
    xgtrain = xgb.DMatrix(train_x, label=train_y)
    xgcv = xgb.DMatrix(cv_x, label=cv_y)
    watchlist = [(xgcv, 'cv')]

    xgb_subm_preds = np.zeros(subm_x.shape[0])
    num_models = 1 ## CHANGE
    for i in range(num_models):
        xgb_model = xgb.train(xgb_params.iloc[i,:].to_dict(), xgtrain, num_boost_round = 3000,
                          evals = watchlist,
                          feval = gini_eval,
                          verbose_eval = False,
                          early_stopping_rounds=100)

        xgb_cv_preds = xgb_model.predict(xgcv, ntree_limit=xgb_model.best_iteration)
        cv_pred_error = normalized_gini(cv_y, xgb_cv_preds)
        print("XGB CV score: {:.6f} (normalized gini).".format(cv_pred_error))

        xgb_subm_preds_fold = xgb_model.predict(xgb.DMatrix(subm_x), ntree_limit=xgb_model.best_iteration)
        xgb_subm_preds += xgb_subm_preds_fold
    xgb_subm_preds /= num_models
    return (cv_pred_error, xgb_subm_preds)

def regressor_NN(train_x, train_y, cv_x, cv_y, subm_x):
    scaler = preprocessing.StandardScaler().fit(train_x)
    train_x = scaler.transform(train_x)
    cv_x = scaler.transform(cv_x)
    subm_x = scaler.transform(subm_x)

    train_x = np.asarray(train_x, dtype = theano.config.floatX)
    train_y = np.asarray(train_y, dtype = theano.config.floatX)

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

            objective_loss_function=squared_error,
            update=nesterov_momentum,
            update_learning_rate=0.001,
            update_momentum=0.9,
            train_split=TrainSplit(eval_size=0.1),
            verbose=1,
            regression=True,
            max_epochs=40)

        return net0

    network = NeuralNetConstructor(train_x.shape[1])

    network.fit(train_x, train_y)

    cv_y_preds = network.predict(np.asarray(cv_x, dtype = theano.config.floatX))
    subm_y_preds = network.predict(np.asarray(subm_x, dtype = theano.config.floatX))

    cv_pred_error = normalized_gini(cv_y, cv_y_preds)
    print("NN CV score: {:.6f} (normalized gini).".format(cv_pred_error))

    return (cv_pred_error, subm_y_preds)
