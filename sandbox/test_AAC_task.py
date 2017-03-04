# Foo task tested with a random RL agent

#### UNFINISHED

from agent.reinforcement import RandomAgent
from brain.models import *
from brain.monitor import Monitor
from brain.networks import *
from world.base import World
from world.data import *
from world.tasks import Foo

# parameters
n_epochs = 150

# define training and validation iterators
train_iter = Foo(batch_size=32, n_batches = 100)
val_iter = Foo(batch_size=32, n_batches = 100)

# define brain of agent
model = ReinforcementModel(MLP(train_iter.n_input, train_iter.n_output, n_hidden=10))

# define agent
agent = RandomAgent(model, chainer.optimizers.Adam())

# define world
world = World(agent)

# run world in training mode with validation
world.validate(train_iter, val_iter, n_epochs=n_epochs, plot=-1)

# add monitor to model
agent.add_monitor(Monitor())

# run world in test mode
world.test(Foo(batch_size=1, n_batches = 100), n_epochs=1, plot=0)

# get variables
Y = agent.monitor['prediction']
T = agent.monitor['target']
[n_samples, n_vars] = Y.shape

