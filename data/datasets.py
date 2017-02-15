import numpy as np
import random
import chainer
import itertools
from chainer.datasets import *

#####
## Example datasets

class ClassificationDataset(TupleDataset):
    """
    Toy dataset for static classification data
    Generates two random inputs and classifies as 0 if their total is smaller than one
    and as 1 otherwise
    """

    def __init__(self):

        X = np.random.rand(1000,2).astype('float32')
        T = (np.sum(X,1) > 1.0).astype('int32')

        super(ClassificationDataset, self).__init__(X, T)

        self.n_input = np.prod(self._datasets[0].shape[1:])
        self.n_output = np.max(self._datasets[1].data) + 1


class RegressionDataset(TupleDataset):
    """
    Toy dataset for static regression data
    Generates two random inputs and outputs their sum and product
    """

    def __init__(self):

        X = np.random.rand(1000,2).astype('float32')
        T = np.vstack([np.sum(X,1), np.prod(X,1)]).transpose().astype('float32')

        super(RegressionDataset, self).__init__(X, T)

        self.n_input = np.prod(self._datasets[0].shape[1:])
        self.n_output = np.prod(self._datasets[1].shape[1:])


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

        self.n_input = np.prod(self._datasets[0].shape[1:])
        self.n_output = np.max(self._datasets[1].data) + 1

class RegressionTimeseries(TupleDataset):
    """
    Toy dataset for dynamic regression data
    """

    def __init__(self):

        X = np.array([np.array([np.sin(i), random.random()], 'float32') for i in xrange(1000)])
        T = np.array([np.array([1, 0], 'float32')] + [np.array([np.sum(i), np.prod(i)], 'float32') for i in X][:-1])

        super(RegressionTimeseries, self).__init__(X, T)

        self.n_input = np.prod(self._datasets[0].shape[1:])
        self.n_output = np.prod(self._datasets[1].shape[1:])

class MNISTData(TupleDataset):
    """
    Handwritten character dataset; example of handling convolutional input
    """


    def __init__(self, validation=False, convolutional=True):

        if validation:
            data = get_mnist()[1]
        else:
            data = get_mnist()[0]

        X = data._datasets[0].astype('float32')
        T = data._datasets[1].astype('int32')

        if convolutional:
            X = np.reshape(X,np.concatenate([[X.shape[0]], [1], [28, 28]]))
            self.n_input = [1, 28, 28]
        else:
            self.n_input = X.shape[1]

        self.n_output = (np.max(T) + 1)

        super(MNISTData, self).__init__(X, T)