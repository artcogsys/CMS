from chainer import Chain, Variable
import numpy as np
import chainer
from chainer.functions.activation import relu
from chainer import link
from chainer.links.connection import linear
from chainer import serializers
import chainer.functions as F

#####
## Wrappers that compute the loss (negative objective function) and the final output for a neural network
#  The predictor is the neural network architecture which gives the predictions for which we can compute outputs and loss

class SupervisedModel(Chain):

    def __init__(self, predictor, loss_function=None, output_function=lambda x:x, monitor=None):
        super(SupervisedModel, self).__init__(predictor=predictor)

        self.loss_function = loss_function
        self.output_function = output_function

        self.set_monitor(monitor)

    def set_monitor(self, monitor):

        # used to store computed states
        self.monitor = monitor
        self.predictor.monitor = monitor

    def load(self, fname):
        serializers.load_npz('{}'.format(fname), self)

    def save(self, fname):
        """ Save a model - boils down to saving network parameters plus any additional parameters to reconstruct the model

        :param fname:
        :return:
        """

        serializers.save_npz('{}'.format(fname), self)

    def __call__(self, x, t, train=False):
        """

        :param x: input
        :param t: target output
        :param train: call predictor in train or test mode
        :return: loss
        """

        if self.monitor:

            y = self.predictor(x, train=train)

            self.monitor.append('prediction',self.output_function(y).data)

            loss = self.loss_function(y, t)

            self.monitor.append('loss', loss.data)
            self.monitor.append('target', t.data)

            return loss

        else:

            return self.loss_function(self.predictor(x, train), t)

    def predict(self, x, train=False):
        """
        Returns prediction, which can be different than raw output (e.g. for softmax function)

        :param x:
        :param train: call predictor in train or test mode
        :return: prediction
        """

        if self.monitor:

            y = self.predictor(x, train=train)
            output = self.output_function(y).data
            self.monitor.append('prediction', output)
            return output

        else:

            return self.output_function(self.predictor(x, train)).data


#####
## Classifier object

class Classifier(SupervisedModel):

    def __init__(self, predictor, monitor=None):
        super(Classifier, self).__init__(predictor=predictor, loss_function=F.softmax_cross_entropy,
                                         output_function=F.softmax, monitor=monitor)

#####
## Regressor object

class Regressor(SupervisedModel):

    def __init__(self, predictor, monitor=None):
        super(Regressor, self).__init__(predictor=predictor, loss_function=F.mean_squared_error, monitor=monitor)