from __future__ import division

from base import Iterator
import numpy as np
import random
import chainer
import itertools
from chainer.datasets import *

## Contains iterators that operate on datasets and default datasets

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

        self.idx = -1
        self._order = np.random.permutation(len(self.data))[:(self.n_batches * self.batch_size)]

        return self

    def next(self):

        if self.idx == self.n_batches-1:
            raise StopIteration

        self.idx += 1

        i = self.idx * self.batch_size

        return list(self.data[self._order[i:(i + self.batch_size)]])

    def process(self, agent):
        pass

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

        self.idx = -1

        offsets = [i * self.n_batches for i in range(self.batch_size)]

        # define custom ordering; we won't process beyond the end of the trial
        self._order = []
        for iter in range(self.n_batches):
            x = [(offset + iter) % len(self.data) for offset in offsets]
            self._order += x

        return self

    def next(self):

        if self.idx == self.n_batches-1:
            raise StopIteration

        self.idx += 1

        i = self.idx * self.batch_size

        return list(self.data[self._order[i:(i + self.batch_size)]])

    def process(self, agent):
        pass


#####
## Toy datasets for rapid prototyping

class ClassificationData(TupleDataset):
    """
    Toy dataset for static classification data
    Generates two random inputs and classifies as 0 if their total is smaller than one
    and as 1 otherwise
    """

    def __init__(self):

        X = np.random.rand(1000,2).astype('float32')
        T = (np.sum(X,1) > 1.0).astype('int32')

        super(ClassificationData, self).__init__(X, T)

    def input(self):
        return np.prod(self._datasets[0].shape[1:])

    def output(self):
        return np.max(self._datasets[1].data) + 1


class RegressionData(TupleDataset):
    """
    Toy dataset for static regression data
    Generates two random inputs and outputs their sum and product
    """

    def __init__(self):

        X = np.random.rand(1000,2).astype('float32')
        T = np.vstack([np.sum(X,1), np.prod(X,1)]).transpose().astype('float32')

        super(RegressionData, self).__init__(X, T)

    def input(self):
        return np.prod(self._datasets[0].shape[1:])

    def output(self):
        return np.prod(self._datasets[1].shape[1:])


class ClassificationTimeseries(TupleDataset):
    """
    Toy dataset for dynamic classification data
    Generates two random inputs and classifies output at the next
    time step as 0 if their total is smaller than one and as 1 otherwise

    """

    def __init__(self):

        X = np.array([np.array([random.random(), random.random()], 'float32') for _ in xrange(1000)])
        T = np.array([np.array(0, 'int32')] + [np.array(0, 'int32') if sum(i) < 1.0 else np.array(1, 'int32') for i in X][:-1])

        super(ClassificationTimeseries, self).__init__(X, T)

    def input(self):
        return np.prod(self._datasets[0].shape[1:])

    def output(self):
        return np.max(self._datasets[1].data) + 1

class RegressionTimeseries(TupleDataset):
    """
    Toy dataset for dynamic regression data
    """

    def __init__(self):

        X = np.array([np.array([np.sin(i), random.random()], 'float32') for i in xrange(1000)])
        T = np.array([np.array([1, 0], 'float32')] + [np.array([np.sum(i), np.prod(i)], 'float32') for i in X][:-1])

        super(RegressionTimeseries, self).__init__(X, T)

    def input(self):
        return np.prod(self._datasets[0].shape[1:])

    def output(self):
        return np.prod(self._datasets[1].shape[1:])

#####
## Real-world datasets

class MNISTData(TupleDataset):
    """
    Handwritten character dataset; example of handling convolutional input
    """

    def __init__(self, test=False, convolutional=True, n_samples=None):
        """

        :param test: return test instead of training set
        :param convolutional: return convolutional representation or not
        :param n_samples: return n_samples samples per class
        """

        if test:
            data = get_mnist()[1]
        else:
            data = get_mnist()[0]

        X = data._datasets[0].astype('float32')
        T = data._datasets[1].astype('int32')

        if convolutional:
            X = np.reshape(X,np.concatenate([[X.shape[0]], [1], [28, 28]]))
            self._n_input = [1, 28, 28]
        else:
            self._n_input = X.shape[1]

        self._n_output = (np.max(T) + 1)

        if n_samples:
            idx = [np.where(T==u)[0][:n_samples] for u in np.unique(T)]
            idx = list(itertools.chain(*idx))
            X = X[idx]
            T = T[idx]

        super(MNISTData, self).__init__(X, T)

    def input(self):
        return self._n_input

    def output(self):
        return self._n_output

class CIFARData(TupleDataset):

    def __init__(self, test=False, convolutional=True, n_samples=None):
        """

       :param test: return test instead of training set
       :param convolutional: return convolutional representation or not
       :param n_samples: return n_samples samples per class
       """

        if convolutional:
            train, test = get_cifar10(withlabel=True, ndim=3)
        else:
            train, test = get_cifar10(withlabel=True, ndim=1)

        if test:
            X = test._datasets[0].astype('float32')
            T = test._datasets[1].astype('int32')
        else:
            X = train._datasets[0].astype('float32')
            T = train._datasets[1].astype('int32')

        if convolutional:
            self._n_input = list(X.shape[1:])
        else:
            self._n_input = np.prod(X.shape[1:])
        self._n_output = (np.max(T) + 1)

        if n_samples:
            idx = [np.where(T == u)[0][:n_samples] for u in np.unique(T)]
            idx = list(itertools.chain(*idx))
            X = X[idx]
            T = T[idx]

        super(CIFARData, self).__init__(X, T)

    def input(self):
        return self._n_input

    def output(self):
        return self._n_output


class PTBData(TupleDataset):
    """
    Penn Tree Bank words dataset
    """

    def __init__(self, kind='train'):
        """

        :param kind: 'train', 'validation', 'test'
        """

        train, val, test = chainer.datasets.get_ptb_words()

        if kind == 'train':
            data = train
        elif kind == 'validation':
            data = val
        elif kind == 'test':
            data = test
        else:
            raise ValueError()

        self.word_to_idx = chainer.datasets.get_ptb_words_vocabulary()

        # create reverse vocabulary
        self.idx_to_word = {}
        for k in self.word_to_idx.keys():
            self.idx_to_word[self.word_to_idx[k]] = k

        super(PTBData, self).__init__(data[:-1], data[1:])

        self.n_vocab = len(self.word_to_idx)


class PTBCharData(TupleDataset):
    """
    Penn Tree Bank dataset; character level representation
    """

    def __init__(self, kind='train'):
        """

        :param kind: 'train', 'validation', 'test'
        """

        train, val, test = chainer.datasets.get_ptb_words()

        if kind == 'train':
            word_data = train
        elif kind == 'validation':
            word_data = val
        elif kind == 'test':
            word_data = test
        else:
            raise ValueError()

        word_to_idx = chainer.datasets.get_ptb_words_vocabulary()

        # create reverse word vocabulary
        idx_to_word = {}
        for k in word_to_idx.keys():
            idx_to_word[word_to_idx[k]] = k

        # create unique characters and their conversion
        char_data = ''
        for word in word_to_idx.keys():
            char_data += word
        char_data = set(char_data)
        self.char_to_idx = dict(zip(char_data, np.arange(len(char_data)).tolist()))

        # create reverse character vocabulary
        self.idx_to_char = {}
        for k in self.char_to_idx.keys():
            self.idx_to_char[self.char_to_idx[k]] = k

        # generate dataset
        data = list(itertools.chain(*map(lambda x: map(lambda x: self.char_to_idx[x], list(idx_to_word[x])), word_data)))
        data = np.array(data, 'int32')

        super(PTBCharData, self).__init__(data[:-1], data[1:])

        self.n_vocab = len(self.char_to_idx)

class VIM2(TupleDataset):
    """
    VIM-2 Dataset
    """

    pass