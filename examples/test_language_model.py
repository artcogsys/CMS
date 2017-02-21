# test language prediction on PTB data

import chainer

from data.datasets import PTBCharData
from learners.base import Learner, Tester
from learners.iterators import *
from learners.supervised_learner import StatefulTrainer
from models.models import Classifier
from models.networks import RNNForLM

# parameters
n_epochs = 100

# get training and validation data
train_data = PTBCharData(kind='train')
val_data  = PTBCharData(kind='validation')

# define model
model = Classifier(RNNForLM(train_data.n_vocab, n_hidden=20))

# Set up an optimizer
optimizer = chainer.optimizers.SGD(lr=1.0)
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.GradientClipping(5))

# define trainer object - train on sequences of length 100 characters
trainer = StatefulTrainer(optimizer, SequentialIterator(train_data, batch_size=20, n_batches=100), cutoff=100)

# define tester object
tester = Tester(model, SequentialIterator(val_data, batch_size=20, n_batches=100))

# define learner to run multiple epochs
learner = Learner(trainer, tester)

# run the optimization
learner.run(n_epochs)

# get trained model
model = learner.model

# predict characters
C=[]
c = 0
model.predictor.reset_state()
for i in range(100):
    pvals = model.predict(Variable(np.array([c], dtype='int32'), True)).squeeze()
    c = np.where(np.random.multinomial(1, pvals))[0][0]
    C.append(c)

# print message
print(''.join(map(lambda c: train_data.idx_to_char[c], C)))
