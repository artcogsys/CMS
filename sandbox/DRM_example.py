# Implementation of dynamic representational modeling. We assume the existence of a recurrent network containing the
# 'representations'. Each neural population has its own (set of) RNN units. Each population projects to one output
# variable (e.g. its BOLD response).

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.stats as ss
import tools as tools
from agent.supervised import StatefulAgent
from brain.models import *
from brain.monitor import Monitor
from brain.networks import *
from world.base import World
from world.data import *

#####
## DRM

class DRM(ChainList, Network):

    def __init__(self, n_input, n_output, n_hidden):
        """Each output gets its own population of n_hidden hidden units

        :param n_input: sensory inputs
        :param n_output: neural measurements
        :param n_hidden: number of hidden units per output
        """

        links = ChainList()

        # Elman layer at which the representations 'emerge'
        links.add_link(Elman(n_input, n_hidden * n_output))

        # add LSTM mechanism per output and a linear readout mechanism
        for i in range(n_output):
            links.add_link(L.LSTM(n_hidden, 1))
            links.add_link(L.Linear(1,1))

        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.monitor = None

        super(DRM, self).__init__(links)

    def __call__(self, x, train=False):

        # recurrent network
        h = self[0][0](x)

        # readout mechanism
        z = []
        for i in range(self.n_output):
            tmp = self[0][1+2*i](h[:,(i*self.n_hidden):((i+1)*self.n_hidden)])
            z.append(self[0][1+2*i+1](tmp))

        # combine all outputs
        y = F.concat(tuple(z), axis=1)

        return y

    def reset_state(self):

        self[0][0].reset_state()
        for i in range(len(self[0])//2):
            self[0][1+2*i].reset_state()

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
model = Regressor(DRM(train_iter.data.input(), train_iter.data.output(), n_hidden=5))

# define agent
agent = StatefulAgent(model, chainer.optimizers.Adam())

# add hook
agent.optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5))

# define world
world = World(agent)

# run world in training mode with validation
world.validate(train_iter, val_iter, n_epochs=n_epochs, plot=-1)

# add monitor to model
agent.add_monitor(Monitor())

# run world in test mode
world.test(SequentialIterator(MyDataset(), batch_size=1), n_epochs=1, plot=0)

# get variables
Y = agent.monitor['prediction']
T = agent.monitor['target']
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