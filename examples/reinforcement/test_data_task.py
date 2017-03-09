# test object categorization under noise as an RL task

from agent.reinforcement import *
from brain.monitor import *
from brain.models import ActorCriticModel
from brain.networks import *
from world.base import World
from world.data import *
from world.tasks import *
import tools

# parameters
n_epochs = 1

# get training data - note that we select a subset of datapoints (n_samples is nr of samples per class)
train_data = MNISTData(test=False, convolutional=False, n_samples=1000, classes = [0, 1])

# define iterator
data_iter = DataTask(train_data, batch_size=32, n_batches = 10000, noise=0, rewards=[-1, 50, -50])

# an actor-critic model assumes that the predictor's output is number of actions plus one for the value
n_output = data_iter.n_output + 1

# define brain of agent
model = ActorCriticModel(RNN(data_iter.n_input, n_output, n_hidden=30))

# define agent
agent = AACAgent(model, chainer.optimizers.Adam(), cutoff=10)

# add gradient clipping 
agent.optimizer.add_hook(chainer.optimizer.GradientClipping(5))

# add oscilloscope
agent.add_monitor(Oscilloscope(names=['cumulative reward']))
#agent.add_monitor(Oscilloscope(names=['return']))

monitor = Oscilloscope(names=['accuracy'])
data_iter.add_monitor(monitor)
agent.add_monitor(monitor)

# define world
world = World(agent)

# run world in training mode
world.train(data_iter, n_epochs=n_epochs, plot=100, monitor=100)

# run world in test mode
val_data = MNISTData(test=True, convolutional=False, n_samples=1000, classes=[0, 1])
val_iter = DataTask(val_data, batch_size=1, n_batches = 1000, noise=0, rewards=[-1, 50, -50])

# add monitor to model and iterator
monitor = Monitor()
agent.add_monitor(monitor)
val_iter.add_monitor(monitor)

# run in test mode
world.test(val_iter, n_epochs=1, plot=0)

# get variables
Y = monitor['action']
T = monitor['state']

# only focus on those trials at which a decision is being made
idx = np.where(Y != data_iter.n_output-1)[0]
Y = Y[idx]
T = T[idx]

# plot confusion matrix

conf_mat = tools.confusion_matrix(Y, T)

fig = plt.figure()

# ignore the trials in which the agent asks for more evidence
n_vars = data_iter.n_output - 1

plt.imshow(conf_mat, interpolation='nearest')
plt.xlabel('Predicted class')
plt.ylabel('True class')
plt.xticks(np.arange(n_vars))
plt.gca().set_xticklabels([str(item) for item in 1 + np.arange(n_vars)])
plt.yticks(np.arange(n_vars))
plt.gca().set_yticklabels([str(item) for item in 1 + np.arange(n_vars)])
plt.colorbar()
plt.title('Confusion matrix; accuracy = ' + str(100.0 * np.sum(np.diag(conf_mat))/np.sum(conf_mat[...])) + '%')

tools.save_plot(fig, world.out, 'confusion_matrix')

plt.close()