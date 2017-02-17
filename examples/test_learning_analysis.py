# Example of how to perform a learning analysis; derived from test_feedforward_regressor

import glob

import chainer
from data.datasets import RegressionDataset
from learners.base import Learner, Tester
from learners.iterators import *
from learners.supervised_learner import StatelessTrainer
from models.models import Regressor
from models.networks import MLP
import matplotlib.cm as cm
from analysis.tools import movie

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

# define learner to run multiple epochs - save snapshot every epoch
learner = Learner(trainer, tester, epoch_snapshot=1)

# run the optimization
learner.run(n_epochs)

# get trained model
model = learner.model

# store analyses on snapshots as video

# get snapshots
snapshots = glob.glob('./result/epoch-snapshot*')

# set data iterator
data = SequentialIterator(RegressionDataset(), batch_size=1)

# define function to apply to the monitor ran on the model
def foo(monitor):

    # extract variables
    Y = monitor.get('prediction')
    T = monitor.get('target')
    [n_samples, n_vars] = Y.shape

    colors = cm.rainbow(np.linspace(0, 1, n_vars))
    regs = []
    for i in range(n_vars):
        l = plt.scatter(T[:, i], Y[:, i], c=colors[i, :])
        plt.hold('on')
    plt.axis('off')

# create movie
movie(snapshots, data, model, foo, "./result/movie.mp4", dpi=100, fps=1)