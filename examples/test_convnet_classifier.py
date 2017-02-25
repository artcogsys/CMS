import matplotlib.pyplot as plt
import tools as tools
from agent.supervised import StatelessAgent
from brain.models import *
from brain.monitor import Monitor
from brain.networks import *
from world.base import World
from world.datasets import MNISTData
from world.iterators import *

# parameters
n_epochs = 50

# get training and validation data - note that we select a subset of datapoints
train_data = MNISTData(test=False, convolutional=True, n_samples=100)
val_data = MNISTData(test=True, convolutional=True, n_samples=100)

# define training and validation environment
train_iter = RandomIterator(train_data, batch_size=32)
val_iter = RandomIterator(val_data, batch_size=32)

# define brain of agent
model = Classifier(ConvNet(train_iter.data.input(), train_iter.data.output(), n_hidden=10))

# define agent
agent = StatelessAgent(model, chainer.optimizers.Adam())

# add hook
agent.optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5))

# define world
world = World(agent)

# run world in training mode with validation
world.validate(train_iter, val_iter, n_epochs=n_epochs, plot=-1)

# add monitor to model
world.agents[0].model.add_monitor(Monitor())

# run world in test mode
world.test(SequentialIterator(val_data, batch_size=1), n_epochs=1, plot=0)

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