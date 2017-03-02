# Example of how to compare two models (or even two optimizers)

from agent.supervised import StatefulAgent
from brain.models import *
from brain.networks import *
from world.data import *
from world.base import World
from brain.links import Elman

# parameters
n_epochs = 150

# define training and validation environment
train_iter = SequentialIterator(RegressionTimeseries(), batch_size=32)
val_iter = SequentialIterator(RegressionTimeseries(), batch_size=32)

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
world.validate(train_iter, val_iter, n_epochs=n_epochs, plot=-1)