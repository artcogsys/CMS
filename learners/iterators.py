from __future__ import division

from chainer.iterators import SerialIterator
from chainer import Variable, cuda, serializers
import numpy as np
import pickle
import tqdm
import matplotlib.pyplot as plt

#####
## Iterator class; specifies how to process data in one epoch

class DataIterator(object):
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        raise NotImplementedError

    def next(self):
        # generator of next batch

        raise NotImplementedError

#####
## Random iterator

class RandomIterator(DataIterator):

    def __init__(self, data, batch_size=32):
        super(RandomIterator, self).__init__(data)

        self.batch_size = batch_size
        self.n_batches = (len(self.data) // self.batch_size)

    def __iter__(self):

        self.batch_idx = 0
        self._order = np.random.permutation(len(self.data))[:(self.n_batches * self.batch_size)]

        return self

    def next(self):

        if self.batch_idx == self.n_batches:
            raise StopIteration

        i = self.batch_idx * self.batch_size

        self.batch_idx += 1

        return [self.data[index] for index in self._order[i:(i + self.batch_size)]]

#####
## Sequential iterator

class SequentialIterator(DataIterator):

    def __init__(self, data, batch_size=32):
        super(SequentialIterator, self).__init__(data)

        self.batch_size = batch_size
        self.n_batches = (len(self.data) // self.batch_size)

    def __iter__(self):

        self.batch_idx = 0

        offsets = [i * self.n_batches for i in range(self.batch_size)]

        # define custom ordering; we won't process beyond the end of the trial
        self._order = []
        for iter in range(self.n_batches):
            x = [(offset + iter) % len(self.data) for offset in offsets]
            self._order += x

        return self


    def next(self):

        if self.batch_idx == self.n_batches:
            raise StopIteration

        i = self.batch_idx * self.batch_size

        self.batch_idx += 1

        return [self.data[index] for index in self._order[i:(i + self.batch_size)]]