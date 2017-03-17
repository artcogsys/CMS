# Toy dataset for dynamic regression data, test

import matplotlib.cm as cm
import scipy.stats as ss
import tools as tools
from agent.supervised import StatefulAgent
from brain.models import *
from brain.monitor import Monitor
from brain.networks import *
from world.base import World
from world.data import *
import matplotlib.pyplot as plt

# parameters
n_epochs = 150

# define training and validation environment
train_iter = SequentialIterator(RegressionTimeseries(), batch_size=32)
val_iter = SequentialIterator(RegressionTimeseries(), batch_size=32)

# define brain of agent
model = Regressor(RNN(train_iter.data.input(), train_iter.data.output(), n_hidden=10, n_hidden_layers=1))

# define agent
agent = StatefulAgent(model, chainer.optimizers.Adam())

# add hook
agent.optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5))

# define world
world = World(agent)

# run world in training mode with validation
world.validate(train_iter, val_iter, n_epochs=n_epochs, plot=100)

# add monitor to model
agent.add_monitor(Monitor())

# run world in test mode
world.test(SequentialIterator(RegressionTimeseries(), batch_size=1), n_epochs=1, plot=False)

# get variables
Y = agent.monitor[0]['prediction']
T = agent.monitor[0]['target']
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