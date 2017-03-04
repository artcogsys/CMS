from __future__ import division

import numpy as np
from chainer import cuda

class Agent(object):

    def __init__(self, model, optimizer=None, gpu=-1):
        """

        :param model:
        :param optimizer:
        :param gpu:
        """

        self.model = model

        if not optimizer is None:

            optimizer.setup(model)

            # facilitate setting of hooks
            # optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5))

        self.optimizer = optimizer

        self.xp = np if gpu == -1 else cuda.cupy

        self.reset()

        self.monitor = []

    def add_monitor(self, monitor):

        # used to store computed states
        self.monitor.append(monitor)
        self.model.add_monitor(monitor)

    def reset(self):
        self.model.predictor.reset_state()

    def run(self, batch, train=True, idx=None, final=False):
        """ Process one minibatch

        :param batch: minibatch
        :param train: run agent in train or in test mode
        :param idx: index of current batch
        :param final: flags if we are in the final batch
        :return: loss for this minibatch averaged over number of datapoints
        """

        raise NotImplementedError
