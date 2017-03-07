# example of using delayiterator to extend classification of static inputs over time
# also uses masks to implement feedforward versus feedback drive in Elman network

import matplotlib.pyplot as plt
import tools as tools
from agent.supervised import *
from brain.models import *
from brain.monitor import Monitor
from brain.networks import *
from world.base import World
from world.data import *

#####
## Delay iterator - a trial iterator that spits out the same datapoint n_batch times

class DelayIterator(SequentialIterator):

    def __init__(self, data, n_batches, batch_size=None, noise=0):

        batch_size = batch_size or len(data)

        super(DelayIterator, self).__init__(data, batch_size=batch_size, n_batches=n_batches)

        # flags noise level
        self.noise = noise

    def __iter__(self):

        self.idx = 0

        # generate another random batch in each epoch
        self._order = np.random.permutation(len(self.data))[:self.batch_size]

        return self

    def next(self):

        if self.idx == self.n_batches:
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


# parameters
n_epochs = 500

# get training and validation data - note that we select a subset of datapoints
train_data = MNISTData(test=False, convolutional=False, n_samples=100)
val_data = MNISTData(test=True, convolutional=False, n_samples=100)

# define training and validation environment; each trial consists of five consecutive frames under a certain noise level
# note that n_batches defines the number of time steps in a trial
train_iter = DelayIterator(train_data, n_batches=5, batch_size=32, noise=0.5)
val_iter = DelayIterator(val_data, n_batches=5, batch_size=32, noise=0.5)

ninput = train_iter.data.input()
nhidden = 20
noutput = train_iter.data.output()

# define brain of agent
model = Classifier(RNN(ninput, noutput, n_hidden=nhidden, link=Elman))

## define masks for Elman network

# input-hidden mask - only allow first half to watch input
mU = np.ones([nhidden, ninput], np.float32)
mU[(nhidden/2):,:] = 0

# hidden-hidden mask - either recurrent or feedforward
b_recurrent = True
if b_recurrent:
    mW = np.ones([nhidden, nhidden], np.float32)
else:
    mW = np.ones([nhidden, nhidden], np.float32)
    mW[(nhidden/2):,:(nhidden/2)] = 0 # remove feedback connections

# ugly way of setting the mask due to the generic RNN formulation
# removing this mask will yield the fully connected Elman network
model.predictor[0][0].maskW = mW
model.predictor[0][0].maskU = mU

# define agent - last=True ensures that we only update at the end of a trial
b_last = False
agent = StatefulAgent(model, chainer.optimizers.Adam(), last=b_last)

# add hook
agent.optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5))

# define world
world = World(agent)

# run world in training mode with validation
world.validate(train_iter, val_iter, n_epochs=n_epochs, plot=-1)

# add monitor to model
agent.add_monitor(Monitor())

# run world in test mode
world.test(DelayIterator(val_data, n_batches=5, batch_size=200, noise=0.5), n_epochs=1, plot=0)

# get variables
Y = agent.monitor[0]['prediction']
T = agent.monitor[0]['target']

# only take the predictions for the last time step if b_last = True
if b_last:
    Y = Y[-200:]
    T = T[-200:]

[n_samples, n_vars] = Y.shape

# plot confusion matrix

conf_mat = tools.confusion_matrix(Y, T)

fig = plt.figure()

plt.imshow(conf_mat, interpolation='nearest')
plt.xlabel('Predicted class')
plt.ylabel('True class')
plt.xticks(np.arange(n_vars)),
plt.gca().set_xticklabels([str(item) for item in 1 + np.arange(n_vars)])
plt.yticks(np.arange(n_vars))
plt.gca().set_yticklabels([str(item) for item in 1 + np.arange(n_vars)])
plt.colorbar()
plt.title('Confusion matrix; accuracy = ' + str(100.0 * np.sum(np.diag(conf_mat))/np.sum(conf_mat[...])) + '%')

tools.save_plot(fig, world.out, 'confusion_matrix')

plt.close()
