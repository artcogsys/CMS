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

    def __init__(self, predictor):
        super(Model, self).__init__(predictor=predictor)

        self.monitor = []

    def add_monitor(self, monitor):

        # used to store computed states
        self.monitor.append(monitor)
        self.predictor.add_monitor(monitor)

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

    def __init__(self, predictor, loss_function=None, output_function=lambda x:x):
        super(SupervisedModel, self).__init__(predictor=predictor)

        self.loss_function = loss_function
        self.output_function = output_function

    def __call__(self, data, train=False):

        x = data[0] if len(data)==2 else data[:-1] # inputs
        t = data[-1] # targets

        # check for missing data
        # missing = [np.any(np.isnan(t[i].data)) or (t[i].data.dtype == 'int32' and np.any(t[i].data == -1)) for i in range(len(t.data))]

        if self.monitor:

            if isinstance(x, list):
                map(lambda v: v.set('input', np.hstack(map(lambda z: z.data, x))), self.monitor)
            else:
                map(lambda v: v.set('input', x.data), self.monitor)

            y = self.predictor(x, train=train)

            map(lambda x: x.set('prediction',self.output_function(y).data), self.monitor)

            loss = self.loss_function(y, t)

            map(lambda x: x.set('loss', loss.data), self.monitor)
            map(lambda x: x.set('target', t.data), self.monitor)

            return loss

        else:
            return self.loss_function(self.predictor(x, train), t)

    def predict(self, data, train=False):

        if self.monitor:

            if isinstance(data, list):
                map(lambda v: v.set('input', np.hstack(map(lambda z: z.data, data))), self.monitor)
            else:
                map(lambda v: v.set('input', data.data), self.monitor)

            y = self.predictor(data, train=train)
            output = self.output_function(y).data

            map(lambda x: x.set('prediction', output), self.monitor)

            return output

        else:

            return self.output_function(self.predictor(data, train)).data


#####
## Classifier object

class Classifier(SupervisedModel):

    def __init__(self, predictor):
        super(Classifier, self).__init__(predictor=predictor, loss_function=F.softmax_cross_entropy,
                                         output_function=F.softmax)

#####
## Regressor object

class Regressor(SupervisedModel):

    def __init__(self, predictor):
        super(Regressor, self).__init__(predictor=predictor, loss_function=F.mean_squared_error)


#####
## Reinforcement learning actor-critic model
#

class ActorCriticModel(Model):
    """
    An actor critic model computes the action, policy and value from a predictor
    """

    def __init__(self, predictor):
        super(ActorCriticModel, self).__init__(predictor=predictor)

    def __call__(self, data, train=False):
        """

        :param data: observation
        :param train: True or False
        :return: policy and value
        """

        # separate observation from reward
        x = data[0] if len(data) == 3 else data[:-2]  # inputs (rest is reward and terminal state)

        if self.monitor:
            if isinstance(x, list):
                map(lambda v: v.set('input', np.hstack(map(lambda z: z.data, x))), self.monitor)
            else:
                map(lambda v: v.set('input', x.data), self.monitor)

        # linear outputs reflecting the log action probabilities and the value
        out = self.predictor(x, train)

        policy = out[:,:-1]

        value = out[:,-1]

        # # handle case where we have only one element per batch
        # if value.ndim == 1:
        #     value = F.expand_dims(value, axis=1)

        action = self.get_action(policy)

        if self.monitor:
            map(lambda x: x.set('action', action), self.monitor)

        return action, policy, value

    def predict(self, data, train=False):

        out = self.predictor(data, train)

        policy = out[:, :-1]

        return self.get_action(policy)

    def get_action(self,policy):

        # generate action according to policy
        p = F.softmax(policy).data

        # normalize p in case tiny floating precision problems occur
        row_sums = p.sum(axis=1)
        p /= row_sums[:, np.newaxis]

        # discrete representation
        n_out = self.predictor.n_output-1

        return np.array([np.random.choice(n_out, None, True, p[i]) for i in range(p.shape[0])], dtype='int32')
