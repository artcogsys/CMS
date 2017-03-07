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
        self.noise = 0

        # get unique values in dataset
        self.values = np.unique(self.data)

    def __iter__(self):

        self.idx = 0

        # generate another random batch in each epoch
        self._order = np.random.permutation(len(self.data))[:self.batch_size]

        return self

    def next(self):

        if self.idx == self.n_batches:
            raise StopIteration

        self.idx += 1

        data = self.data[self._order]

        return list(self.data[self._order])


# parameters
n_epochs = 300

# get training and validation data - note that we select a subset of datapoints
train_data = MNISTData(test=False, convolutional=False, n_samples=100)
val_data = MNISTData(test=True, convolutional=False, n_samples=100)

# define training and validation environment; each trial consists of five consecutive frames
train_iter = DelayIterator(train_data, n_batches=5, batch_size=32)
val_iter = DelayIterator(val_data, n_batches=5, batch_size=32)

ninput = train_iter.data.input()
nhidden = 10
noutput = train_iter.data.output()

# define mask for Elman network
mW = np.ones([nhidden, nhidden], np.float32) # hidden-hidden mask
#mU = np.ones([nhidden, ninput], np.float32) # input-hidden mask

# define brain of agent
model = Classifier(RNN(ninput, noutput, n_hidden=nhidden, link=Elman))

# ugly way of setting the mask due to the generic RNN formulation
model.predictor[0][0].maskW = np.triu(mW,0)

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
world.test(SequentialIterator(val_data, batch_size=1), n_epochs=1, plot=0)

# get variables
Y = agent.monitor[0]['prediction']
T = agent.monitor[0]['target']
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















#
#
#
#
#
#
#
# import chainer
# import tools as an
# from environments.datasets import CIFARData
# from learners.base import Learner, Tester
# from learners.iterators import *
# from learners.supervised_learner import StatefulTrainer
# from models.models import Classifier
# from models.monitor import Monitor
# from models.networks import ConvNet
#
#
# #####
# ## Delay iterator - a trial iterator that spits out the same datapoint n_batch times
#
# class DelayIterator(SequentialIterator):
#
#     def __init__(self, data, trial_length, batch_size=None, noise=None):
#         super(DelayIterator, self).__init__(data)
#
#         self.batch_size = batch_size or len(self.data)
#         self.n_batches = trial_length
#
#         # flags type of noise - not yet implemented
#         self.noise = None
#
#     def __iter__(self):
#
#         self.batch_idx = 0
#
#         # generate another random batch in each epoch
#         self._order = np.random.permutation(len(self.data))[:self.batch_size]
#
#         return self
#
#     def next(self):
#
#         if self.batch_idx == self.n_batches:
#             raise StopIteration
#
#         self.batch_idx += 1
#
#         return [self.data[index] for index in self._order]
#
#
# #####
# ## Main
#
# # parameters
# n_epochs = 10
#
# # get training and validation data
# # note that we select a subset of datapoints
# train_data = CIFARData(test=False, convolutional=True, n_samples=100)
# val_data = CIFARData(test=True, convolutional=True, n_samples=100)
#
# # define model
# model = Classifier(ConvNet(train_data.n_input, train_data.n_output, n_hidden=10))
#
# # Set up an optimizer
# optimizer = chainer.optimizers.Adam()
# optimizer.setup(model)
# optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5))
#
# # define trainer object
# # last=True ensures that we only update at the last time point
# trainer = StatefulTrainer(optimizer, DelayIterator(train_data, trial_length=3), last=True)
#
# # define tester object
# tester = Tester(model, DelayIterator(val_data, trial_length=3))
#
# # define learner to run multiple epochs
# learner = Learner(trainer, tester)
#
# # run the optimization
# learner.run(n_epochs)
#
# # get trained model
# model = learner.model
#
# # add monitor to model
# model.set_monitor(Monitor())
#
# # test some data with the learner - requires target data
# tester = Tester(model, DelayIterator(val_data, trial_length=3))
# tester.run()
#
# # get variables
# Y = model.monitor.get('prediction')
# T = model.monitor.get('target')
# [n_samples, n_vars] = Y.shape
#
# # plot confusion matrix
#
# conf_mat = an.confusion_matrix(Y, T)
#
# fig = plt.figure()
#
# plt.imshow(conf_mat, interpolation='nearest')
# plt.xlabel('Predicted class')
# plt.ylabel('True class')
# plt.xticks(np.arange(n_vars)),
# plt.gca().set_xticklabels([str(item) for item in 1 + np.arange(n_vars)])
# plt.yticks(np.arange(n_vars))
# plt.gca().set_yticklabels([str(item) for item in 1 + np.arange(n_vars)])
# plt.colorbar()
# plt.title('Confusion matrix')
#
# an.save_plot(fig,'result','confusion_matrix')
#
# plt.close()