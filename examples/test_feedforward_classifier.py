import chainer
import analysis.tools as an
from data.datasets import ClassificationDataset
from learners.base import Learner, Tester
from learners.iterators import *
from learners.supervised_learner import StatelessTrainer
from models.models import Classifier
from models.monitor import Monitor
from models.networks import MLP

# parameters
n_epochs = 50

# get training and validation data
train_data = ClassificationDataset()
val_data  = ClassificationDataset()

# define model
model = Classifier(MLP(train_data.n_input, train_data.n_output, n_hidden=10, n_hidden_layers=1))

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
tester = Tester(model, SequentialIterator(ClassificationDataset(), batch_size=1))
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