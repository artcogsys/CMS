# Implementation of a dynamic filter layer: De Brabandere, B., Jia, X., Tuytelaars, T., Van Gool, L., 2016. Dynamic Filter Networks.

from agent.supervised import StatelessAgent
from brain.models import *
from brain.networks import *
from world.base import World
from world.data import *
from brain.links import DynamicFilterLinear

#####
## Implementation of a standard MLP

class StandardMLP(Chain, Network):
    """
    Basic convolutional neural network
    """

    def __init__(self, n_input, n_output, n_hidden=10):
        """

        :param n_input: nchannels x height x width
        :param n_output: number of action outputs
        :param n_hidden: number of hidden units
        """

        super(StandardMLP, self).__init__(
            l1=L.Linear(n_input, n_hidden),
            l2=L.Linear(n_hidden, n_output)
        )

        self.ninput = n_input
        self.nhidden = n_hidden
        self.noutput = n_output

    def __call__(self, x, train=False):
        """
        :param x: sensory input (ntrials x nchannels x ninput[0] x ninput[1])
        """

        return self.l2(F.relu(self.l1(x)))

#####
## Dynamic filter implemntation of a standard MLP

class DynamicFilterMLP(Chain, Network):
    """
    Basic convolutional neural network
    """

    def __init__(self, predictors, n_input, n_output, n_hidden=10, constantW=True):
        """

        :param predictors: list of predictors for each of the weight matrices
        :param n_input: nchannels x height x width
        :param n_output: number of action outputs
        :param n_hidden: number of hidden units
        :param constantW: add constant W

        """

        super(DynamicFilterMLP, self).__init__(
            l1=DynamicFilterLinear(predictors[0], n_input, n_hidden, constantW=constantW),
            l2=DynamicFilterLinear(predictors[1], n_hidden, n_output, constantW=constantW)
        )

        self.ninput = n_input
        self.nhidden = n_hidden
        self.noutput = n_output

    def __call__(self, x, train=False):
        """
        :param x: sensory input (ntrials x nchannels x ninput[0] x ninput[1])
        """

        # NOTE: Context is provided by the previous input

        h = F.relu(self.l1(x, x))

        return self.l2(h, h)

#####
## Main

# parameters
n_epochs = 200

# number of hidden units used in basic MLP
n_hidden = 10

# get training and validation data - note that we select a subset of datapoints
# train_data = ClassificationData()
# val_data = ClassificationData()
train_data = MNISTData(test=False, convolutional=False, n_samples=100)
val_data = MNISTData(test=True, convolutional=False, n_samples=100)

# define training and validation environment
train_iter = RandomIterator(train_data, batch_size=32)
val_iter = RandomIterator(val_data, batch_size=32)

n_input = train_iter.data.input()
n_output = train_iter.data.output()

# specify predictors that implement the dynamic filtering
df_hid = 10

predictors = [StandardMLP(n_input, n_input*n_hidden, n_hidden=df_hid), StandardMLP(n_hidden, n_hidden*n_output, n_hidden=df_hid)]

# define agent 1
model1 = Classifier(DynamicFilterMLP(predictors, train_iter.data.input(), train_iter.data.output(), n_hidden=n_hidden, constantW=True))
agent1 = StatelessAgent(model1, chainer.optimizers.Adam())

# define agent 2
model2 = Classifier(StandardMLP(train_iter.data.input(), train_iter.data.output(), n_hidden=n_hidden))
agent2 = StatelessAgent(model2, chainer.optimizers.Adam())

# define world
world = World([agent1, agent2])

# add labels to plot - validate first generates training losses and then test losses
world.labels = ['DF train', 'DF test', 'MLP train', 'MLP test']

# run world in training mode with validation
world.validate(train_iter, val_iter, n_epochs=n_epochs, plot=100)
