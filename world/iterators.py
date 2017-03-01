from __future__ import division

from chainer.iterators import SerialIterator
from chainer import Variable, cuda, serializers
import numpy as np
import pickle
import tqdm
import matplotlib.pyplot as plt

#####
## An iterator generates batches

class Iterator(object):

    def __init__(self, batch_size=None, n_batches=None):

        self.batch_size = batch_size
        self.n_batches = n_batches

        # index of current batch
        self.idx=0

    def __iter__(self):

        self.idx = 0

    def next(self):
        """

        :return: a list of numpy arrays where the first dimension is the minibatch size
        """

        raise NotImplementedError

    def is_final(self):
        """

        :return: boolean if final batch is reached
        """
        return ((self.idx+1)==self.n_batches)

#####
## Random iterator - returns random samples of a chainer TupleDataset

class RandomIterator(Iterator):

    def __init__(self, data, batch_size=None, n_batches=None):

        self.data = data

        if batch_size is None:
            batch_size = 1

        if n_batches is None:
            n_batches = len(self.data) // batch_size

        super(RandomIterator, self).__init__(batch_size=batch_size, n_batches=n_batches)

        assert (self.n_batches * self.batch_size <= len(self.data))

    def __iter__(self):

        self.idx = 0
        self._order = np.random.permutation(len(self.data))[:(self.n_batches * self.batch_size)]

        return self

    def next(self):

        if self.idx == self.n_batches:
            raise StopIteration

        i = self.idx * self.batch_size

        self.idx += 1

        return list(self.data[self._order[i:(i + self.batch_size)]])

#####
## Sequential iterator - returns sequential samples of a chainer TupleDataset

class SequentialIterator(Iterator):

    def __init__(self, data, batch_size=None, n_batches=None):

        self.data = data

        if batch_size is None:
            batch_size = 1

        if n_batches is None:
            n_batches = len(self.data) // batch_size

        super(SequentialIterator, self).__init__(batch_size=batch_size, n_batches=n_batches)

    def __iter__(self):

        self.idx = 0

        offsets = [i * self.n_batches for i in range(self.batch_size)]

        # define custom ordering; we won't process beyond the end of the trial
        self._order = []
        for iter in range(self.n_batches):
            x = [(offset + iter) % len(self.data) for offset in offsets]
            self._order += x

        return self

    def next(self):

        if self.idx == self.n_batches:
            raise StopIteration

        i = self.idx * self.batch_size

        self.idx += 1

        return list(self.data[self._order[i:(i + self.batch_size)]])


#####
## Task iterator - implements a task

class TaskIterator(Iterator):
    """
    Iterator's next function returns an observation based on a previous action
    """

    def reset(self):
        """
        Reset environment to initial state and return observation

        :return: observation
        """

        raise NotImplementedError

    def act(self, action):
        """
        Act upon the environment

        """

        raise NotImplementedError

    def render(self):
        # render the environment

        pass

