# example of life-long learning with hydranet

import chainer
from world.iterators import *
from agent.supervised import StatefulAgent
from brain.models import Regressor
from chainer.datasets import *
from world.base import World
from brain.monitor import Monitor
from brain.networks import *

#####
## Hydranet RNN

class HydraRNN(Chain, Network):

    def __init__(self, n_input, n_hidden):

        super(HydraRNN, self).__init__(
            l1=Elman(n_input, n_hidden),
            l2=L.Linear(n_hidden, n_input),
        )

    def __call__(self, x, train=False):

        h = self.l1(F.dropout(x, train=True))
        y = self.l2(h)

        return y

    def reset_state(self):
        self.l1.reset_state()

#####
## Hydranet iterator

class HydraIterator(Iterator):

    def __init__(self, batch_size=32, n_batches=None, n_update=1000, noise=0.1):

        batch_size = batch_size # 1 is most reasonable for realistic scenarios

        n_batches = n_batches or np.inf

        super(HydraIterator, self).__init__(batch_size=batch_size, n_batches=n_batches)

        self.data = get_mnist()[0]._datasets[0].astype('float32')

        self.n_update = n_update # update task after so many iterations

        self._order = []

        self.n_input = self.data[0].size

        self.noise = noise

    def __iter__(self):

        self.idx = 0

        return self

    def next(self):

        # never stop iterating during life-long learning
        if self.idx == self.n_batches:
            raise StopIteration

        if self.idx % self.n_update == 0:
            self._order = np.random.permutation(len(self.data))[0:self.batch_size]

        self.idx += 1

        # self.data[self._order[i:(i + self.batch_size)]]
        data = self.data[self._order]

        noise = data.copy()
        mask = (np.random.rand(*data.shape) < self.noise).squeeze()
        noise[mask] = np.random.rand(np.sum(mask==True))

        return [noise, data]

#####
## Create custom monitor

class MyMonitor(Monitor):

    def run(self):

        input = self.get('input')[-1]
        output = self.get('prediction')[-1]

        if not hasattr(self, 'fig'):

            self.fig, self.ax = plt.subplots(2)

            self.hl = [None, None]

            self.hl[0] = self.ax[0].imshow(np.reshape(input, [28, 28]), interpolation='nearest', cmap='gray')
            self.ax[0].set_title('input')
            self.ax[0].axis('off')
            self.ax[0].axis('equal')
            self.hl[1] = self.ax[1].imshow(np.reshape(output, [28, 28]), interpolation='nearest', cmap='gray')
            self.ax[1].set_title('output')
            self.ax[1].axis('off')
            self.ax[1].axis('equal')
            self.fig.show()

        else:

            self.hl[0].set_data(np.reshape(input, [28, 28]))
            self.hl[1].set_data(np.reshape(output, [28, 28]))
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

#####
## Main

# iterator
iter = HydraIterator(batch_size=32, n_update=10)

# define model
model = Regressor(HydraRNN(iter.n_input, n_hidden=100))

# define trainer object
agent = StatefulAgent(model, chainer.optimizers.Adam(), cutoff=50)

# add hook
agent.optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5))

# add monitor to model and define function to apply
agent.model.add_monitor(MyMonitor(names=['input', 'prediction'], append=False))

# define world
world = World(agent)

# run world in training mode
world.train(iter, n_epochs=1, plot=100, monitor=100)


