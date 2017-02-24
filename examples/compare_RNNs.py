# Example of how to compare two models (or even two optimizers)

from agent.supervised import StatefulAgent
from brain.models import *
from brain.networks import *
from world.iterators import *
from chainer.datasets import TupleDataset
from world.base import World
from brain.links import Elman
import random

# parameters
n_epochs = 150

# define dataset
class MyDataset(TupleDataset):

    def __init__(self):

        X = np.array([np.array([np.sin(i), random.random()], 'float32') for i in xrange(1000)])
        T = np.array([np.array([1, 0], 'float32')] + [np.array([np.sum(i), np.prod(i)], 'float32') for i in X][:-1])

        super(MyDataset, self).__init__(X, T)

    def input(self):
        return np.prod(self._datasets[0].shape[1:])

    def output(self):
        return np.prod(self._datasets[1].shape[1:])

# define training and validation environment
train_iter = SequentialIterator(MyDataset(), batch_size=32)
val_iter = SequentialIterator(MyDataset(), batch_size=32)

# define agent 1
model1 = Regressor(RNN(train_iter.data.input(), train_iter.data.output(), n_hidden=10, n_hidden_layers=1))
agent1 = StatefulAgent(model1, chainer.optimizers.Adam())

# define agent 2
model2 = Regressor(RNN(train_iter.data.input(), train_iter.data.output(), n_hidden=10, n_hidden_layers=1, link=Elman))
agent2 = StatefulAgent(model2, chainer.optimizers.Adam())

# define world
world = World([agent1, agent2])

# add labels to plot - validate first generates training losses and then test losses
world.labels = ['LSTM train', 'Elman train', 'LSTM test', 'Elman test']

# run world in training mode with validation
world.validate(train_iter, val_iter, n_epochs=n_epochs, plot=True)