# Foo task tested with a random RL agent

import os
from agent.reinforcement import *
from brain.monitor import *
from brain.models import ActorCriticModel
from brain.networks import *
from world.base import World
from world.data import *
from world.tasks import *

# parameters
n_epochs = 1  # a task can run for an infinite time

# define iterator
#data_iter = Foo(batch_size=1, n_batches = 2*10**3)
data_iter = ProbabilisticCategorizationTask(batch_size=1, n_batches = 2*10**4)

# an actor-critic model assumes that the predictor's output is number of actions plus one for the value
n_output = data_iter.n_output + 1

# define brain of agent
model = ActorCriticModel(RNN(data_iter.n_input, n_output, n_hidden=5, link=L.StatefulGRU))

# define agent - update model every 100 steps
agent = ActorCriticAgent(model, chainer.optimizers.Adam(), cutoff=10)

# add gradient clipping
agent.optimizer.add_hook(chainer.optimizer.GradientClipping(5))

# add monitors
agent.add_monitor(Oscilloscope(names=['cumulative reward']))

# define world
world = World(agent)

# run world in training mode - plot cumulative reward every 100 iterations
world.train(data_iter, n_epochs=n_epochs, plot=100, monitor=100)

##  run in test mode

data_iter = ProbabilisticCategorizationTask(batch_size=1, n_batches = 1*10**3)

# reset cumulative reward monitor
agent.monitor[0].reset()

world.test(data_iter, n_epochs=1, plot=0, monitor=1)

# save cumulative reward in test mode
agent.monitor[0].save(os.path.join(world.out, 'cumulative reward'))
