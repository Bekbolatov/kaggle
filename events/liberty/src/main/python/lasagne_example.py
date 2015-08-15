from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne import NeuralNet


def NeuralNetConstructor(num_features):
    layers0 = [('input', InputLayer),
               ('hidden1', DenseLayer),
               ('dropout1', DropoutLayer),
               ('hidden2', DenseLayer),
               ('dropout2', DropoutLayer),
               ('hidden3', DenseLayer),
               ('dropout3', DropoutLayer),
               ('output', DenseLayer)]

    net0 = NeuralNet(layers=layers0,
                     input_shape=(None, num_features),
                    hidden1_num_units=50,
                    dropout1_p=0.3,
                      hidden2_num_units=150,
                      dropout2_p=0.5,
                      hidden3_num_units=200,
                      dropout3_p=0.2,

                      output_num_units=1,
                      output_nonlinearity=None,

                      update=nesterov_momentum,
                      update_learning_rate=0.05,
                      update_momentum=0.9,

                      eval_size=0.1,
                      verbose=1,
                      regression=True,
                      max_epochs=35)
    return net0


#? output_nonlinearity=rectify
#You should probably use regression instead of sofmax. Look at the training params, i'm almost sure that there is a regression flag.
#I do set "regression=True", but with "output_nonlinearity=softmax" it results in [0,1], If instead I set "output_nonlinearity=rectify", then the output is greater than 1. However I'm just not sure that I'm doing the right thing here.

