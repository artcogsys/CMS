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
        self.idx=0

    def next(self):
        raise NotImplementedError

    def is_final(self):
        """

        :return: boolean if final batch is reached
        """
        return ((self.idx+1)==self.n_batches)

#####
## Random iterator - returns random samples of a chainer dataset

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

        return [self.data[index] for index in self._order[i:(i + self.batch_size)]]

#####
## Sequential iterator - returns sequential samples of a chainer dataset

class SequentialIterator(Iterator):

    def __init__(self, data, batch_size=None, n_batches=None):

        self.data = data

        if batch_size is None:
            batch_size = 1

        if n_batches is None:
            n_batches = len(self.data) // batch_size

        super(SequentialIterator, self).__init__(batch_size=batch_size, n_batches=n_batches)

        assert (self.n_batches * self.batch_size <= len(self.data))

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

        return [self.data[index] for index in self._order[i:(i + self.batch_size)]]



# #####
# ## Trial iterator - operates on consecutive trials
#
# class TrialIterator(SequentialIterator):
#
#     def __init__(self, data, n_batches):
#         """
#
#         :param data:
#         :param n_batches: number of timepoints of which a trial consists
#         """
#
#         batch_size = len(data) // n_batches
#
#         # this must hold for trial-based data
#         assert(n_batches * batch_size == len(data))
#
#         super(TrialIterator, self).__init__(data, batch_size=batch_size, n_batches=n_batches)