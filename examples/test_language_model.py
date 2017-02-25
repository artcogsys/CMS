# test language prediction on character level PTB data

from agent.supervised import StatefulAgent
from brain.models import *
from brain.networks import *
from world.base import World
from world.iterators import *
from world.datasets import PTBCharData

# parameters
n_epochs = 100

# get training and validation data - note that we select a subset of datapoints
train_data = PTBCharData(kind='train')
val_data  = PTBCharData(kind='validation')

# define training and validation environment
train_iter = SequentialIterator(train_data, batch_size=20, n_batches=100)
val_iter = SequentialIterator(val_data, batch_size=20, n_batches=100)

# define brain of agent
model = Classifier(RNNForLM(train_data.n_vocab, n_hidden=20))

# define agent
agent = StatefulAgent(model, chainer.optimizers.SGD(lr=1.0), cutoff=100)

# add hook
agent.optimizer.add_hook(chainer.optimizer.GradientClipping(5))

# define world
world = World(agent)

# run world in training mode with validation
world.validate(train_iter, val_iter, n_epochs=n_epochs, plot=-1)

# get trained model
model = world.agents[0].model

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
