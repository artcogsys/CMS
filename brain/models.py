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

class Model(Chain):

    def __init__(self, predictor, monitor=None):
        super(Model, self).__init__(predictor=predictor)

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

    def __call__(self, data, train=False):
        """ Compute loss for minibatch of data

        :param data: list of minibatches (e.g. inputs and targets for supervised learning)
        :param train: call predictor in train or test mode
        :return: loss
        """

        raise NotImplementedError

    def predict(self, data, train=False):
        """
        Returns prediction, which can be different than raw output (e.g. for softmax function)

        :param data: minibatch or list of minibatches representing input to the model
        :param train: call predictor in train or test mode
        :return: prediction
        """

        raise NotImplementedError

#####
## Supervised model
#

class SupervisedModel(Model):

    def __init__(self, predictor, loss_function=None, output_function=lambda x:x, monitor=None):
        super(SupervisedModel, self).__init__(predictor=predictor, monitor=monitor)

        self.loss_function = loss_function
        self.output_function = output_function

    def __call__(self, data, train=False):

        x = data[0] if len(data)==2 else data[:-1] # inputs
        t = data[-1] # targets

        if self.monitor:

            if isinstance(x, list):
                self.monitor.set('input', np.hstack(map(lambda z: z.data, x)))
            else:
                self.monitor.set('input', x.data)

            y = self.predictor(x, train=train)

            self.monitor.set('prediction',self.output_function(y).data)

            loss = self.loss_function(y, t)

            self.monitor.set('loss', loss.data)
            self.monitor.set('target', t.data)

            return loss

        else:

            return self.loss_function(self.predictor(x, train), t)

    def predict(self, data, train=False):

        if self.monitor:

            if isinstance(data, list):
                self.monitor.set('input', np.hstack(map(lambda z: z.data, data)))
            else:
                self.monitor.set('input', data.data)

            y = self.predictor(data, train=train)
            output = self.output_function(y).data

            self.monitor.set('prediction', output)

            return output

        else:

            return self.output_function(self.predictor(data, train)).data


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