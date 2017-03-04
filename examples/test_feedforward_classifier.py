# Toy dataset for static classification data
# Generates two random inputs and classifies as 0 if their total is smaller than one
# and as 1 otherwise

import tools as tools
from agent.supervised import StatelessAgent
from brain.models import *
from brain.monitor import Monitor
from brain.networks import *
from world.base import World
from world.data import *
import matplotlib.pyplot as plt

# parameters
n_epochs = 100

# define training and validation environment
train_iter = RandomIterator(ClassificationData(), batch_size=32)
val_iter = RandomIterator(ClassificationData(), batch_size=32)

# define brain of agent
model = Classifier(MLP(train_iter.data.input(), train_iter.data.output(), n_hidden=10, n_hidden_layers=1))

# define agent
agent = StatelessAgent(model, chainer.optimizers.Adam())

# add hook
agent.optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5))

# define world
world = World(agent)

# run world in training mode with validation
world.validate(train_iter, val_iter, n_epochs=n_epochs, plot=-1)

# add monitor to model
agent.add_monitor(Monitor())

# run world in test mode
world.test(SequentialIterator(ClassificationData(), batch_size=1), n_epochs=1, plot=False)

# get variables
Y = agent.monitor['prediction']
T = agent.monitor['target']
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