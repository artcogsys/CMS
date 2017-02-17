from __future__ import division

import pickle

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from chainer import cuda, serializers, Variable
import os

#####
## Trainer class - used to run one epoch in training mode - will run forever for life-long learning

class Trainer(object):

    def __init__(self, optimizer, data, gpu=-1, iter_snapshot=None, out='result'):
        """

        :param optimizer: optimizer to train model
        :param data_iterator: iterator over the training data
        :param validation: optional validation iterator (selects optimal model)
        :param gpu: device flag
        :param iter_snapshot: store snapshot every iter_snapshot iterations
        :param path: path to store snapshots
        """

        self.model = optimizer.target

        self.optimizer = optimizer
        self.data = data

        self.xp = np if gpu == -1 else cuda.cupy

        self.iter_snapshot= iter_snapshot

        self.out = out
        try:
            os.makedirs(self.out)
        except OSError:
            pass

        self.iteration = 0

    def run(self, data=None):
        """ Run one training epoch

        :param data: iterator to run on; chooses associated self.data if left unspecified
        :return:
        """

        raise NotImplementedError


#####
## Tester class - used to run one epoch in test mode

class Tester(object):

    def __init__(self, model, data, gpu=-1):
        """

        :param optimizer: optimizer to train model
        :param data: iterator over the training data
        :param validation: optional validation iterator (selects optimal model)
        :param gpu: device flag

        """

        self.model = model

        self.data = data

        self.xp = np if gpu == -1 else cuda.cupy

    def run(self, data=None):
        """ Deafult implementation to run one test epoch

        :param data: iterator to run on; chooses associated self.data if left unspecified
        :return:
        """

        # attach default dataset
        if data is None:
            data = self.data

        cumloss = 0

        self.model.predictor.reset_state()

        for batch in data:

            x = Variable(self.xp.asarray([item[0] for item in batch]), True)
            t = Variable(self.xp.asarray([item[1] for item in batch]), True)

            cumloss += self.model(x, t, train=False).data

        return float(cumloss / data.batch_size)

#####
## Learner class

class Learner(object):

    def __init__(self, trainer, tester=None, epoch_snapshot=None, out='result'):
        """

        :param trainer:
        :param tester:
        :param epoch_snapshot: save snapshot every epoch_snapshot iterations
        :param path:  path to store results (snapshots and loss); however, trainer also should report loss
        """

        self.trainer = trainer
        self.tester = tester

        self.out = out
        try:
            os.makedirs(self.out)
        except OSError:
            pass

        self.epoch_snapshot = epoch_snapshot

        self._fig = None
        self._ax = None
        self._hl = None

    def run(self, n_epochs=1):
        """

        :param n_epochs: number of training epochs
        :return:
        """

        # keep track of minimal validation loss
        min_loss = float('nan')

        self.epoch = self.trainer.optimizer.epoch

        for self.epoch in tqdm.tqdm(xrange(self.trainer.optimizer.epoch, self.trainer.optimizer.epoch + n_epochs)):

            if self.epoch_snapshot and self.epoch % self.epoch_snapshot == 0:
                self.trainer.model.save(os.path.join(self.out, 'epoch-snapshot-' + '{0:04d}'.format(self.epoch)))

            train_loss = self.trainer.run()

            if self.tester:
                val_loss = self.tester.run()
            else:
                val_loss = float('nan')

            self.plot_loss(train_loss, val_loss, self.epoch)

            # store optimal model
            if np.isnan(min_loss):
                optimal_model = self.trainer.optimizer.target.copy()
                min_loss = val_loss
            else:
                if val_loss < min_loss:
                    optimal_model = self.trainer.optimizer.target.copy()
                    min_loss = val_loss

            self.trainer.optimizer.new_epoch() # needed?

        # compute final loss
        self.plot_loss(self.tester.run(self.trainer.data), self.tester.run(), self.epoch+1)

        if self.epoch_snapshot and (self.epoch + 1) % self.epoch_snapshot == 0:
            self.trainer.model.save(os.path.join(self.out, 'epoch-snapshot-' + '{0:04d}'.format(self.epoch+1)))

        # model is set to the optimal model according to validation loss
        # or to last model in case no validation set is used
        self.model = optimal_model

        self.trainer.model.save(os.path.join(self.out, 'optimal-model'))

        self._fig.savefig(os.path.join(self.out, 'loss'))

        plt.close()

    def plot_loss(self, train_loss, val_loss, idx):

        if self._fig is None:

            self._fig, self._ax = plt.subplots()
            self._ax.set_xlabel('epoch')
            self._ax.set_ylabel('loss')
            self._hl = np.empty(2, dtype=object)
            self._hl[0], = self._ax.plot(idx, train_loss)
            self._hl[1], = self._ax.plot(idx, val_loss)
            self._fig.legend(self._hl, ('training', 'validation'))
            self._fig.show()

        else:

            x_data = np.vstack([self._hl[0].get_xdata(), idx])
            self._hl[0].set_xdata(x_data)
            self._hl[1].set_xdata(x_data)
            y_data = np.vstack([self._hl[0].get_ydata(), train_loss])
            self._hl[0].set_ydata(y_data)
            y_data = np.vstack([self._hl[1].get_ydata(), val_loss])
            self._hl[1].set_ydata(y_data)
            self._ax.relim()
            self._ax.autoscale_view()
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()


    def load(self, fname):

        with open('{}_log'.format(fname), 'rb') as f:
            self.log = pickle.load(f)

        serializers.load_npz('{}_optimizer'.format(fname), self.optimizer)
        serializers.load_npz('{}_model'.format(fname), self.model)

    def save(self, fname):

        with open('{}_log'.format(fname), 'wb') as f:
            pickle.dump(self.log, f, -1)

        serializers.save_npz('{}_optimizer'.format(fname), self.optimizer)
        serializers.save_npz('{}_model'.format(fname), self.model)

    def train(self, it, monitor=None):
        # one training epoch

        raise NotImplementedError