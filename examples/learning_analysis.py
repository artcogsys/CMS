# Example of how to perform a learning analysis; derived from test_feedforward_regressor

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
n_epochs = 1

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
world.train(train_iter, n_epochs=n_epochs, plot=-1, snapshot=10)

# get snapshots
snapshots = glob.glob(os.path.join(world.out, 'agent-0000-snapshot*'))

# set data iterator
test_iter = RandomIterator(RegressionData(), batch_size=32)

# define function to apply to the monitor ran on the model
def foo(monitor):

    # extract variables
    Y = monitor['prediction']
    T = monitor['target']
    [n_samples, n_vars] = Y.shape

    colors = cm.rainbow(np.linspace(0, 1, n_vars))
    regs = []
    for i in range(n_vars):
        l = plt.scatter(T[:, i], Y[:, i], c=colors[i, :])
        plt.hold('on')
    plt.axis('off')

# create movie
movie(snapshots, test_iter, agent, foo, os.path.join(world.out, 'movie.mp4'), dpi=100, fps=2)