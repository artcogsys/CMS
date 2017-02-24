# example of life-long learning with hydranet

import chainer
from chainer import Chain
import chainer.functions as F
from brain.links import Elman
import chainer.links as L
from world.iterators import *
from agent.supervised import StatefulAgent
from brain.models import Regressor
from chainer.datasets import *
from world.base import World

#####
## Hydranet RNN

class HydraRNN(Chain):

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

    def __init__(self, batch_size=32, n_batches=None, n_update=1000, noise=1.0):

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

        return [tuple([self.data[index], self.data[index] + self.noise * np.random.randn(*self.data[index].shape).astype('float32')])
                for index in self._order]


#####
## Main

# iterator
iter = HydraIterator(batch_size=32, n_update=100)

# define model
model = Regressor(HydraRNN(iter.n_input, n_hidden=10))

# define trainer object
agent = StatefulAgent(model, chainer.optimizers.Adam(), cutoff=50)

# add hook
agent.optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5))

# define world
world = World(agent)

# run world in training mode
world.train(iter, n_epochs=1, plot=1, per_epoch=False)


