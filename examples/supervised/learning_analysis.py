# Example of how to perform a learning analysis; derived from test_feedforward_regressor

import matplotlib
matplotlib.use('Qt4Agg') # macosx backend presently lacks blocking show() behavior when matplotlib is in non-interactive mode

import os
import glob
from tools import movie
from agent.supervised import StatelessAgent
from brain.models import *
from brain.networks import *
from world.base import World
from world.data import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# parameters
n_epochs = 10

# define training environment
train_iter = RandomIterator(RegressionData(), batch_size=32)

# define brain of agent
model = Regressor(MLP(train_iter.data.input(), train_iter.data.output(), n_hidden=10, n_hidden_layers=1))

# define agent
agent = StatelessAgent(model, chainer.optimizers.Adam())

# add hook
agent.optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5))

# define world
world = World(agent)

# run world in training mode with validation - shows how to store snapshots each n iterations (instead of epochs)
world.train(train_iter, n_epochs=n_epochs, plot=100, snapshot=100)

# get snapshots
snapshots = glob.glob(os.path.join(world.out, 'agent-0000-snapshot*'))

# set data iterator
test_iter = RandomIterator(RegressionData(), batch_size=32)

# define function to apply to the monitor ran on the model
def foo(monitor):

    # extract variables
    Y = monitor[0]['prediction']
    T = monitor[0]['target']
    [n_samples, n_vars] = Y.shape

    colors = cm.rainbow(np.linspace(0, 1, n_vars))
    for i in range(n_vars):
        l = plt.scatter(T[:, i], Y[:, i], c=colors[i, :])
        plt.hold('on')
    plt.axis('off')

# create movie
movie(snapshots, test_iter, agent, foo, os.path.join(world.out, 'movie.mp4'), dpi=100, fps=1)