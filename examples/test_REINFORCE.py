# Foo task tested with a random RL agent

from agent.reinforcement import *
from brain.monitor import *
from brain.models import ActorCriticModel
from brain.networks import *
from world.base import World
from world.data import *
from world.tasks import Foo

# parameters
n_epochs = 200

# define iterator
data_iter = Foo(batch_size=32, n_batches = 100)

# an actor-critic model assumes that the predictor's output is number of actions plus one for the value
n_output = data_iter.n_output + 1

# define brain of agent
model = ActorCriticModel(MLP(data_iter.n_input, n_output, n_hidden=10))

# define agent
agent = REINFORCEAgent(model, chainer.optimizers.Adam())

# add oscilloscope
agent.add_monitor(Oscilloscope(names=['return']))

# define world
world = World(agent)

# run world in training mode
world.train(data_iter, n_epochs=n_epochs, plot=-1, monitor=-1)

