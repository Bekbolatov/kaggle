
from nolearn.lasagne import NeuralNet, TrainSplit

import warnings
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.updates import nesterov_momentum, sgd
from lasagne.objectives import squared_error, categorical_crossentropy, binary_crossentropy
from lasagne.nonlinearities import sigmoid, rectify, softmax, linear, tanh
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.utils.validation import NonBLASDotWarning
import theano
import numpy as np
from dataset import get_data

warnings.simplefilter('always', NonBLASDotWarning)
from gini import normalized_gini
from numpy import log

print("Loading datasets")
dat_x_orig, dat_y_orig, lb_x_orig, lb_ind = get_data()

def labels_1_2(n):
    if n < 11:
        return 0
    else:
        return 1

nplabels = np.vectorize(labels_1_2)

scaler = preprocessing.StandardScaler().fit(dat_x_orig)
dat_x = scaler.transform(dat_x_orig)
lb_x = scaler.transform(lb_x_orig)
dat_y = nplabels(dat_y_orig) #** 0.75


dat_x = np.asarray(dat_x, dtype = theano.config.floatX)
dat_y = np.asarray(dat_y, dtype = theano.config.intX)

train_index, test_index = train_test_split(range(dat_x.shape[0]), test_size=0.10, random_state=103)

train_x = dat_x[train_index, :]
train_y = dat_y[train_index]
cv_x = dat_x[test_index, :]
cv_y = dat_y[test_index]


'''
info = {
                'epoch': num_epochs_past + epoch,
                'train_loss': avg_train_loss,
                'train_loss_best': best_train_loss == avg_train_loss,
                'valid_loss': avg_valid_loss,
                'valid_loss_best': best_valid_loss == avg_valid_loss,
                'valid_accuracy': avg_valid_accuracy,
                'dur': time() - t0,
                }
'''
def on_epoch_finished(obj, train_history):
    if (len(train_history) > 20):
        inLastEight = any([h['valid_loss_best'] for h in  train_history[-8:-1]] +
                          [train_history[-1]['valid_loss'] < train_history[-2]['valid_loss'],
                           train_history[-1]['valid_loss'] < train_history[-3]['valid_loss'],
                           train_history[-2]['valid_loss'] < train_history[-3]['valid_loss'],
                           ])
        if not inLastEight:
            print("Stopping early")
            raise StopIteration


def NeuralNetConstructor(num_features):
    layers0 = [
        ('input', InputLayer),
        ('dropout0', DropoutLayer),
        ('hidden1', DenseLayer),
        ('dropout1', DropoutLayer),
        ('hidden2', DenseLayer),
        # ('dropout2', DropoutLayer),
        # ('hidden3', DenseLayer),
        # ('dropout3', DropoutLayer),
        ('output', DenseLayer)]

    net0 = NeuralNet(
        layers=layers0,

        input_shape=(None, num_features),
        input_nonlinearity=sigmoid,

        dropout0_p=0.3,

        hidden1_num_units=20, #75, #20,
        hidden1_nonlinearity=sigmoid,

        dropout1_p=0.2,

        hidden2_num_units=20, #70, #24,
        hidden2_nonlinearity=sigmoid,

        # dropout2_p=0.2,

        # hidden3_num_units=2,
        # hidden3_nonlinearity=sigmoid,

        # dropout3_p=0.1,

        output_num_units=2,
        output_nonlinearity=softmax,

        on_epoch_finished=[on_epoch_finished],

        objective_loss_function=binary_crossentropy, #binary_crossentropy, #squared_error, categorical_crossentropy
        update=nesterov_momentum,
        update_learning_rate=0.005, #0.02,
        update_momentum=0.9,
        train_split=TrainSplit(eval_size=0.1),
        verbose=1,
        regression=False,
        max_epochs=5)

    return net0

#add bias
network = NeuralNetConstructor(111)
network.fit(train_x, train_y)

cv_preds = network.predict(cv_x)

# output: 2 units, no softmax, then labels are (1, 0) form -  predictions are (0.93, 0.001)

#  >>>
def labels_transform(n):
    if n == 0:
        return (1, 0)
    else:
        return (0, 1)

ltran = np.vectorize(labels_transform)
train_yy = ltran(train_y)
train_yyy = np.asarray(np.c_[train_yy[0].ravel(), train_yy[1].ravel()], dtype=theano.config.floatX)
network.fit(train_x, train_yyy)

cv_yy = ltran(cv_y)
cv_yyy = np.asarray(np.c_[cv_yy[0].ravel(), cv_yy[1].ravel()], dtype=theano.config.floatX)
cv_preds_yyy = network.predict(cv_x)

# <<<<
network.fit(train_x, train_y)

cv_preds = network.predict(cv_x)

np.uint8


#sum(np.power( cv_preds - cv_y.reshape(-1, 1), 2 ))/cv_y.shape[0]

#print(normalized_gini(cv_y, cv_preds))

def fds(x):
    if x == 0:
        return (1, 0, 0)
    elif x == 1:
        return (0, 1, 0)
    else:
        return (0, 0, 1)

fdss = np.vectorize(fds)

cvv_y = fdss(cv_y)


cvv_y = fdss(train_y)
train_y = np.asarray( np.c_[cvv_y[0].ravel(), cvv_y[1].ravel(), cvv_y[2].ravel()], dtype= theano.config.floatX)

#############################

