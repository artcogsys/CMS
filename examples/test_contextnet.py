# contextnet; testing whether using context-dependent weights improves classification performance

import math

import chainer
from chainer import ChainList
import analysis.tools as an
from data.datasets import ClassificationDataset
from learners.base import Learner, Tester
from learners.iterators import *
from learners.supervised_learner import StatelessTrainer
from models.models import Classifier
from models.monitor import Monitor
import chainer.functions as F
from chainer import initializers
from chainer.functions.connection import linear

# show that with this approach a perceptron can solve an xor problem
# speed up via matrix decompositions
# call function very slow
# generalize to allow arbitrary contexts

#####
## Define custom contextnet layer

class ContextLayer(chainer.Link):

    def __init__(self, in_size, out_size, n_context, wscale=1, bias=0, nobias=False,
                 initialW=None, initial_bias=None):
        # Parameters are initialized as a numpy array of given shape.

        super(ContextLayer, self).__init__()

        self.out_size = out_size
        self.n_context = n_context

        self._W_initializer = initializers._get_initializer(
            initialW, math.sqrt(wscale))

        if in_size is None:
            self.add_uninitialized_param('W')
            for i in range(n_context):
                self.add_uninitialized_param('W' + str(i))
        else:
            self._initialize_params(in_size, n_context)

        if nobias:
            self.b = None
        else:
            if initial_bias is None:
                initial_bias = bias
            bias_initializer = initializers._get_initializer(initial_bias)
            self.add_param('b', out_size, initializer=bias_initializer)

    def _initialize_params(self, in_size, n_context):

        self.add_param('W', (self.out_size, in_size),
                       initializer=self._W_initializer)

        for i in range(n_context):
            self.add_param('W' + str(i), (self.out_size, in_size),
                           initializer=self._W_initializer)

    def __call__(self, x, z):
        """Applies the context layer.

        Args:
            x (~chainer.Variable): Batch of input vectors.
            z (~chainer.Variable): Batch of context vectors.

        Returns:
            ~chainer.Variable: Output of the context layer.

        """
        if self.has_uninitialized_params:
            with cuda.get_device(self._device_id):
                self._initialize_params(x.size // x.shape[0], z.size // z.shape[0])

        batch_size = x.shape[0]

        y = linear.linear(x, self.W)
        for i in range(self.n_context):
            W = getattr(self, 'W'+str(i))
            sz = list(W.shape)
            W = F.broadcast_to(W, [batch_size] + sz)
            zz = F.tile(F.reshape(z[:,i],[batch_size,1,1]),tuple([1] + sz))
            C = zz * W
            y += F.squeeze(F.batch_matmul(C, x), 2)
        y += F.tile(self.b,tuple([batch_size,1]))

        return y


#####
## Define ContextMLP

class ContextMLP(ChainList):

    def __init__(self, n_input, n_output, n_hidden=10, n_hidden_layers=1, actfun=F.relu):
        """

        Now activity determines its own context
        For arbitrary context, we need custom datasets that generate the context
        Here, however, we take the interesting case that the network itself can produce its own context

        :param n_input: number of inputs
        :param n_output: number of outputs
        :param n_hidden: number of hidden units
        :param n_hidden_layers: number of hidden layers (1; standard MLP)
        :param actfun: used activation function (ReLU)
        :param monitor: monitors internal states
        """

        links = ChainList()
        if n_hidden_layers == 0:
            links.add_link(ContextLayer(n_input, n_output, n_input))
        else:
            links.add_link(ContextLayer(n_input, n_hidden, n_input))
            for i in range(n_hidden_layers - 1):
                links.add_link(ContextLayer(n_hidden, n_hidden, n_hidden))
            links.add_link(ContextLayer(n_hidden, n_output, n_hidden))

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_hidden_layers = n_hidden_layers
        self.actfun = actfun
        self.monitor = None

        super(ContextMLP, self).__init__(links)

    def __call__(self, x, train=False):

        if self.n_hidden_layers == 0:

            y = self[0][0](x, x)

        else:

            h = self.actfun(self[0][0](x, x))
            for i in range(1, self.n_hidden_layers):
                h = self.actfun(self[0][i](h, h))
            y = self[0][-1](h, h)

        return y

    def reset_state(self):
        pass


#####
## Main

# parameters
n_epochs = 5

# get training and validation data
train_data = ClassificationDataset()
val_data  = ClassificationDataset()

# define model
model = Classifier(ContextMLP(train_data.n_input, train_data.n_output, n_hidden=10, n_hidden_layers=1))

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