# Foo task tested with a random RL agent

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
data_iter = Foo(batch_size=32, n_batches = np.inf)
#data_iter = ProbabilisticCategorizationTask(batch_size=32, n_batches = np.inf)

# an actor-critic model assumes that the predictor's output is number of actions plus one for the value
n_output = data_iter.n_output + 1

# define brain of agent
model = ActorCriticModel(MLP(data_iter.n_input, n_output, n_hidden=10))

# define agent - update model every 100 steps
agent = AACAgent(model, chainer.optimizers.Adam(), cutoff=10)

# add gradient clipping
agent.optimizer.add_hook(chainer.optimizer.GradientClipping(5))

# add monitors
agent.add_monitor(Oscilloscope(names=['return']))
monitor = Oscilloscope(names=['accuracy'])
data_iter.add_monitor(monitor)
agent.add_monitor(monitor)

# define world
world = World(agent)

# run world in training mode - plot every 100 iterations
world.train(data_iter, n_epochs=n_epochs, plot=100, monitor=100)

