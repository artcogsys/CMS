# contextnet; testing whether using context-dependent weights improves classification performance

import math

from chainer import initializers
from agent.supervised import StatelessAgent
from brain.models import *
from brain.networks import *
from world.base import World
from world.data import *
from chainer.functions.connection import linear

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

class ContextMLP(ChainList, Network):

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


#####
## Main

# parameters
n_epochs = 70

# define training and validation environment
train_iter = RandomIterator(ClassificationData(), batch_size=32)
val_iter = RandomIterator(ClassificationData(), batch_size=32)

# define agent 1
model1 = Classifier(ContextMLP(train_iter.data.input(), train_iter.data.output(), n_hidden=10, n_hidden_layers=1))
agent1 = StatelessAgent(model1, chainer.optimizers.Adam())

# define agent 2
model2 = Classifier(MLP(train_iter.data.input(), train_iter.data.output(), n_hidden=10, n_hidden_layers=1))
agent2 = StatelessAgent(model2, chainer.optimizers.Adam())

# define world
world = World([agent1, agent2])

# add labels to plot - validate first generates training losses and then test losses
world.labels = ['Context train', 'Context test', 'MLP train', 'MLP test']

# run world in training mode with validation
world.validate(train_iter, val_iter, n_epochs=n_epochs, plot=-1)
