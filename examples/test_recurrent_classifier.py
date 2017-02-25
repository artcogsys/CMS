# Toy dataset for dynamic classification data

import random

import matplotlib.pyplot as plt
import tools as tools
from agent.supervised import StatefulAgent
from brain.models import *
from brain.monitor import Monitor
from brain.networks import *
from chainer.datasets import TupleDataset
from world.base import World
from world.iterators import *

# parameters
n_epochs = 150

# define dataset
class MyDataset(TupleDataset):

    def __init__(self):

        X = np.array([np.array([random.random(), random.random()], 'float32') for _ in xrange(1000)])
        T = np.array([np.array(0, 'int32')] + [np.array(0, 'int32') if sum(i) < 1.0 else np.array(1, 'int32') for i in X][:-1])

        super(MyDataset, self).__init__(X, T)

    def input(self):
        return np.prod(self._datasets[0].shape[1:])

    def output(self):
        return np.max(self._datasets[1].data) + 1

# define training and validation environment
train_iter = SequentialIterator(MyDataset(), batch_size=32)
val_iter = SequentialIterator(MyDataset(), batch_size=32)

# define brain of agent
model = Classifier(RNN(train_iter.data.input(), train_iter.data.output(), n_hidden=10, n_hidden_layers=1))

# define agent
agent = StatefulAgent(model, chainer.optimizers.Adam())

# add hook
agent.optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5))

# define world
world = World(agent)

# run world in training mode with validation
world.validate(train_iter, val_iter, n_epochs=n_epochs, plot=-1)

# add monitor to model
world.agents[0].model.add_monitor(Monitor())

# run world in test mode
world.test(SequentialIterator(MyDataset(), batch_size=1), n_epochs=1, plot=0)

# get variables
Y = world.agents[0].model.monitor.get('prediction')
T = world.agents[0].model.monitor.get('target')
[n_samples, n_vars] = Y.shape

# plot confusion matrix

conf_mat = tools.confusion_matrix(Y, T)

fig = plt.figure()

plt.imshow(conf_mat, interpolation='nearest')
plt.xlabel('Predicted class')
plt.ylabel('True class')
plt.xticks(np.arange(n_vars)),
plt.gca().set_xticklabels([str(item) for item in 1 + np.arange(n_vars)])
plt.yticks(np.arange(n_vars))
plt.gca().set_yticklabels([str(item) for item in 1 + np.arange(n_vars)])
plt.colorbar()
plt.title('Confusion matrix; accuracy = ' + str(100.0 * np.sum(np.diag(conf_mat))/np.sum(conf_mat[...])) + '%')

tools.save_plot(fig, world.out, 'confusion_matrix')

plt.close()