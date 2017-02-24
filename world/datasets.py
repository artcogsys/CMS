import numpy as np
import random
import chainer
import itertools
from chainer.datasets import *

#####
## Example datasets

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

        self._n_input = list(X.shape[1:])
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
