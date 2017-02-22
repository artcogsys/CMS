# example of using delayiterator to extend classification of static inputs over time

import chainer
import analysis.tools as an
from data.datasets import CIFARData
from learners.base import Learner, Tester
from learners.iterators import *
from learners.supervised_learner import StatefulTrainer
from models.models import Classifier
from models.monitor import Monitor
from models.networks import ConvNet

#####
## Delay iterator - a trial iterator that spits out the same datapoint n_batch times

class DelayIterator(SequentialIterator):

    def __init__(self, data, trial_length, batch_size=None, noise=None):
        super(DelayIterator, self).__init__(data)

        self.batch_size = batch_size or len(self.data)
        self.n_batches = trial_length

        # flags type of noise - not yet implemented
        self.noise = None

    def __iter__(self):

        self.batch_idx = 0

        # generate another random batch in each epoch
        self._order = np.random.permutation(len(self.data))[:self.batch_size]

        return self

    def next(self):

        if self.batch_idx == self.n_batches:
            raise StopIteration

        self.batch_idx += 1

        return [self.data[index] for index in self._order]


#####
## Main

# parameters
n_epochs = 10

# get training and validation data
# note that we select a subset of datapoints
train_data = CIFARData(test=False, convolutional=True, n_samples=100)
val_data = CIFARData(test=True, convolutional=True, n_samples=100)

# define model
model = Classifier(ConvNet(train_data.n_input, train_data.n_output, n_hidden=10))

# Set up an optimizer
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5))

# define trainer object
# last=True ensures that we only update at the last time point
trainer = StatefulTrainer(optimizer, DelayIterator(train_data, trial_length=3), last=True)

# define tester object
tester = Tester(model, DelayIterator(val_data, trial_length=3))

# define learner to run multiple epochs
learner = Learner(trainer, tester)

# run the optimization
learner.run(n_epochs)

# get trained model
model = learner.model

# add monitor to model
model.set_monitor(Monitor())

# test some data with the learner - requires target data
tester = Tester(model, DelayIterator(val_data, trial_length=3))
tester.run()

# get variables
Y = model.monitor.get('prediction')
T = model.monitor.get('target')
[n_samples, n_vars] = Y.shape

# plot confusion matrix

conf_mat = an.confusion_matrix(Y, T)

fig = plt.figure()

plt.imshow(conf_mat, interpolation='nearest')
plt.xlabel('Predicted class')
plt.ylabel('True class')
plt.xticks(np.arange(n_vars)),
plt.gca().set_xticklabels([str(item) for item in 1 + np.arange(n_vars)])
plt.yticks(np.arange(n_vars))
plt.gca().set_yticklabels([str(item) for item in 1 + np.arange(n_vars)])
plt.colorbar()
plt.title('Confusion matrix')

an.save_plot(fig,'result','confusion_matrix')

plt.close()