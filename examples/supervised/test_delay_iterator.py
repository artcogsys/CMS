# example of using delayiterator to extend classification of static inputs over time
# also uses masks to implement feedforward versus feedback drive in Elman network

from agent.supervised import *
from brain.models import *
from world.base import World
from world.data import *
from brain.networks import *

#####
## Delay iterator - a trial iterator that spits out the same datapoint n_batch times

class DelayIterator(SequentialIterator):

    def __init__(self, data, n_batches, batch_size=None, noise=0):

        batch_size = batch_size or len(data)

        super(DelayIterator, self).__init__(data, batch_size=batch_size, n_batches=n_batches)

        # flags noise level
        self.noise = noise

    def __iter__(self):

        self.idx = -1

        # generate another random batch in each epoch
        self._order = np.random.permutation(len(self.data))[:self.batch_size]

        return self

    def next(self):

        if self.idx == self.n_batches - 1:
            raise StopIteration

        self.idx += 1

        d_shape = self.data[self._order][0].shape
        d_size = self.data[self._order][0].size

        # create noise component
        noise = np.zeros(d_size)
        n = int(np.ceil(self.noise*d_size))
        noise[np.random.permutation(d_size)[:n]] = np.random.rand(n)
        noise = noise.reshape(d_shape)

        data = self.data[self._order]
        data[0][noise!=0] = noise[noise!=0]

        return list(data)

#####
## Simple RNN

class MyRNN(Chain, Network):

    def __init__(self, n_input, n_output, n_hidden=10):
        """

        :param n_input: number of inputs
        :param n_hidden: number of hidden units
        :param n_output: number of outputs

        """

        super(MyRNN, self).__init__(
            l1=Elman(n_input, n_hidden, initW=chainer.initializers.Identity(scale=0.01)),
            l2=L.Linear(n_hidden, n_output)
        )

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

    def __call__(self, x, train=False):

        # Elman layer incorporates activation function
        return self.l2(self.l1(x))

    def reset_state(self):
            self.l1.reset_state()

# parameters
n_epochs = 500

# get training and validation data - note that we select a subset of datapoints (n_samples is nr of samples per class)
train_data = MNISTData(test=False, convolutional=False, n_samples=1000)
val_data = MNISTData(test=True, convolutional=False, n_samples=1000)

# define training and validation environment; each trial consists of five consecutive frames under a certain noise level
# note that n_batches defines the number of time steps in a trial
train_iter = DelayIterator(train_data, n_batches=5, batch_size=200, noise=0.5)
val_iter = DelayIterator(val_data, n_batches=5, batch_size=200, noise=0.5)

ninput = train_iter.data.input()
nhidden = 10
noutput = train_iter.data.output()

# b_last=True ensures that we only update at the end of a trial
b_last = False

# define two agents; one with a feedforward structure (neurons are linearly ordered) and one recurrent model (all-to-all connectivity)

mU = np.zeros([nhidden, ninput], np.float32)

model1 = Classifier(MyRNN(ninput, noutput, n_hidden=nhidden))
mW1 = np.ones([nhidden, nhidden], np.float32)
model1.predictor.l1.maskW = mW1
agent1 = StatefulAgent(model1, chainer.optimizers.Adam(), last=b_last)
agent1.optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5))

model2 = Classifier(MyRNN(ninput, noutput, n_hidden=nhidden))
#mW2 = np.triu(np.ones([nhidden, nhidden], np.float32), 0)
mW2 = np.zeros([nhidden, nhidden], np.float32)
model2.predictor.l1.maskW = mW2
agent2 = StatefulAgent(model2, chainer.optimizers.Adam(), last=b_last)
agent2.optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5))

model3 = Classifier(MyRNN(ninput, noutput, n_hidden=nhidden))
mW3 = np.zeros([nhidden, nhidden], np.float32)
model3.predictor.l1.maskW = mW3
agent3 = StatefulAgent(model3, chainer.optimizers.Adam(), last=b_last)
agent3.optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5))

# define world
world = World([agent1, agent2, agent3], labels=['recurrent train', 'recurrent test', 'feedforward train', 'feedforward test', 'AR train', 'AR test'])

# run world in training mode with validation
world.validate(train_iter, val_iter, n_epochs=n_epochs, plot=-1)
