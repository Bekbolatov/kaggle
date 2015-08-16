
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


print("Loading datasets")
dat_x_orig, dat_y_orig, lb_x_orig, lb_ind = get_data()

scaler = preprocessing.StandardScaler().fit(dat_x_orig)
dat_x = scaler.transform(dat_x_orig)
lb_x = scaler.transform(lb_x_orig)
dat_y = dat_y_orig ** 0.75


dat_x = np.asarray(dat_x, dtype = theano.config.floatX)
dat_y = np.asarray(dat_y, dtype = theano.config.floatX)

train_index, test_index = train_test_split(range(dat_x.shape[0]), test_size=0.15, random_state=102)

train_x = dat_x[train_index, :]
train_y = dat_y[train_index]
cv_x = dat_x[test_index, :]
cv_y = dat_y[test_index]







def NeuralNetConstructor(num_features):
    layers0 = [
        ('input', InputLayer),
        ('hidden1', DenseLayer),
        ('dropout1', DropoutLayer),
        ('hidden2', DenseLayer),
        # ('dropout2', DropoutLayer),
        ('hidden3', DenseLayer),
        # ('dropout3', DropoutLayer),
        ('hidden4', DenseLayer),
        ('output', DenseLayer)]

    net0 = NeuralNet(
        layers=layers0,

        input_shape=(None, num_features),

        hidden1_num_units=800,
        hidden1_nonlinearity=sigmoid,

        dropout1_p=0.3,

        hidden2_num_units=800,
        hidden2_nonlinearity=linear,

        # dropout2_p=0.3,

        hidden3_num_units=800,
        hidden3_nonlinearity=sigmoid,

        # dropout3_p=0.3,
        #
        hidden4_num_units=800,
        hidden4_nonlinearity=linear,

        output_num_units=1,
        output_nonlinearity=None,

        objective_loss_function=squared_error,
        #objective=MyObjective,
        update=nesterov_momentum,
        update_learning_rate=0.001,
        update_momentum=0.9,
        train_split=TrainSplit(eval_size=0.1),
        verbose=1,
        regression=True,
        max_epochs=30)

    return net0

#add bias
network = NeuralNetConstructor(111)

network.fit(train_x, train_y)

cv_preds = network.predict(cv_x)

print(normalized_gini(cv_y, cv_preds))
