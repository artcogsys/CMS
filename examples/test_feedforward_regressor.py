import chainer

import analysis.tools as an
from data.datasets import RegressionDataset
from learners.base import Learner, Tester
from learners.iterators import *
from learners.supervised_learner import StatelessTrainer
from models.models import Regressor
from models.monitor import Monitor
from models.networks import MLP
import matplotlib.cm as cm
import scipy.stats as ss

# parameters
n_epochs = 20

# get training and validation data
train_data = RegressionDataset()
val_data  = RegressionDataset()

# define model
model = Regressor(MLP(train_data.n_input, train_data.n_output, n_hidden=10, n_hidden_layers=1))

# Set up an optimizer
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5))

# define trainer object
trainer = StatelessTrainer(optimizer, RandomIterator(train_data, batch_size=32))

# define tester object
tester = Tester(model, SequentialIterator(val_data, batch_size=32))

# define learner to run multiple epochs
learner = Learner(trainer, tester)

# run the optimization
learner.run(n_epochs)

# get trained model
model = learner.model

# add monitor to model
model.set_monitor(Monitor())

# test some data with the learner - requires target data
tester = Tester(model, SequentialIterator(RegressionDataset(), batch_size=1))
tester.run()

# get variables
Y = model.monitor.get('prediction')
T = model.monitor.get('target')
[n_samples, n_vars] = Y.shape

# plot scatterplot

fig = plt.figure()

colors = cm.rainbow(np.linspace(0, 1, n_vars))
regs = []
for i in range(n_vars):
    l = plt.scatter(T[:, i], Y[:, i], c=colors[i, :])
    regs.append(l)
    plt.hold('on')
plt.axis('equal')
plt.grid(True)
plt.xlabel('Observed value')
plt.ylabel('Predicted value')

R = np.zeros([n_vars, 1])
for i in range(n_vars):
    R[i] = ss.pearsonr(np.squeeze(T[:, i]), np.squeeze(Y[:, i]))[0]
plt.title('Scatterplot, <R>={0}'.format(np.mean(R)))

plt.legend(tuple(regs),tuple(1+np.arange(n_vars)))

an.save_plot(fig,'result','scatterplot')

plt.close()