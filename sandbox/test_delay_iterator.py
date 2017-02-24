# example of using delayiterator to extend classification of static inputs over time

import matplotlib.pyplot as plt
import tools as tools
from agent.supervised import StatelessAgent
from brain.models import *
from brain.monitor import Monitor
from brain.networks import *
from world.base import World
from world.datasets import CIFARData
from world.iterators import *

#####
## Delay iterator - a trial iterator that spits out the same datapoint n_batch times

class DelayIterator(SequentialIterator):

    def __init__(self, data, trial_length, batch_size=None, noise=None):

        batch_size = batch_size or len(data)
        n_batches = trial_length

        super(DelayIterator, self).__init__(data, batch_size=batch_size, n_batches=n_batches)

        # flags type of noise - not yet implemented!
        self.noise = None

    def __iter__(self):

        self.idx = 0

        # generate another random batch in each epoch
        self._order = np.random.permutation(len(self.data))[:self.batch_size]

        return self

    def next(self):

        if self.idx == self.n_batches:
            raise StopIteration

        self.idx += 1

        return [self.data[index] for index in self._order]


# parameters
n_epochs = 50

# get training and validation data - note that we select a subset of datapoints
train_data = CIFARData(test=False, convolutional=True, n_samples=100)
val_data =CIFARData(test=True, convolutional=True, n_samples=100)

# define training and validation environment
train_iter = DelayIterator(train_data, trial_length=3)
val_iter = DelayIterator(val_data, trial_length=3)

# define brain of agent
model = Classifier(ConvNet(train_iter.data.input(), train_iter.data.output(), n_hidden=10))

# define agent
agent = StatelessAgent(model, chainer.optimizers.Adam())

# add hook
agent.optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5))

# define world
world = World(agent)

# run world in training mode with validation
world.validate(train_iter, val_iter, n_epochs=n_epochs, plot=True)

# add monitor to model
world.agents[0].model.set_monitor(Monitor())

# run world in test mode
world.test(SequentialIterator(val_data, batch_size=1), n_epochs=1, plot=False)

# get variables
Y = world.agents[0].model.monitor.get('prediction')
T = world.agents[0].model.monitor.get('target')
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