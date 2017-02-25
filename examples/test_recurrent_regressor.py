# Toy dataset for dynamic regression data

import random

import matplotlib.cm as cm
import scipy.stats as ss
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

# define brain of agent
model = Regressor(RNN(train_iter.data.input(), train_iter.data.output(), n_hidden=10, n_hidden_layers=1))

# define agent
agent = StatefulAgent(model, chainer.optimizers.Adam())

# add hook
agent.optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5))

# define world
world = World(agent)

# run world in training mode with validation
world.validate(train_iter, val_iter, n_epochs=n_epochs, plot=-1)

# add monitor to model
world.agents[0].model.set_monitor(Monitor())

# run world in test mode
world.test(SequentialIterator(MyDataset(), batch_size=1), n_epochs=1, plot=0)

# get variables
Y = world.agents[0].model.monitor.get('prediction')
T = world.agents[0].model.monitor.get('target')
[n_samples, n_vars] = Y.shape


# plot scatterplot

fig = plt.figure()

colors = cm.rainbow(np.linspace(0, 1, n_vars))
regs = []
for i in range(n_vars):
    l = plt.scatter(T[:, i], Y[:, i], c=colors[i, :])
    regs.append(l)
    plt.hold('on')
plt.axis('equal')
plt.grid(True)
plt.xlabel('Observed value')
plt.ylabel('Predicted value')

R = np.zeros([n_vars, 1])
for i in range(n_vars):
    R[i] = ss.pearsonr(np.squeeze(T[:, i]), np.squeeze(Y[:, i]))[0]
plt.title('Scatterplot, <R>={0}'.format(np.mean(R)))

plt.legend(tuple(regs),tuple(1+np.arange(n_vars)))

tools.save_plot(fig, 'result', 'scatterplot')

plt.close()