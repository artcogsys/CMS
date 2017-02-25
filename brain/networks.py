from chainer import Chain, ChainList
import chainer.links as L
import chainer.functions as F
from links import Elman
import numpy as np

class Network(object):

    def add_monitor(self, monitor):

        # used to store computed states
        self.monitor = monitor

    def reset_state(self):
        pass


class MLP(ChainList, Network):
    """
    Fully connected deep neural network consisting of a chain of layers (weight matrices)
    with a fixed number of nhidden units. Defaults to a standard MLP with one hidden layer.
    If n_hidden_layers=0 then we have a perceptron.

     """

    def __init__(self, n_input, n_output, n_hidden=10, n_hidden_layers=1, actfun=F.relu):
        """

        :param n_input: number of inputs
        :param n_output: number of outputs
        :param n_hidden: number of hidden units
        :param n_hidden_layers: number of hidden layers (1; standard MLP)
        :param actfun: used activation function (ReLU)
        """

        links = ChainList()
        if n_hidden_layers == 0:
            links.add_link(L.Linear(n_input, n_output))
        else:
            links.add_link(L.Linear(n_input, n_hidden))
            for i in range(n_hidden_layers - 1):
                links.add_link(L.Linear(n_hidden, n_hidden))
            links.add_link(L.Linear(n_hidden, n_output))

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_hidden_layers = n_hidden_layers
        self.actfun = actfun
        self.monitor = None

        super(MLP, self).__init__(links)

    def __call__(self, x, train=False):

        if self.n_hidden_layers == 0:

            y = self[0][0](x)

        else:

            if self.monitor:

                h = self.actfun(self[0][0](x))
                self.monitor.set('hidden-1', h.data)
                for i in range(1,self.n_hidden_layers):
                    h = self.actfun(self[0][i](h))
                    self.monitor.set('hidden-'+str(i+1), h.data)
                y = self[0][-1](h)
                self.monitor.set('output', y.data)

            else:

                h = self.actfun(self[0][0](x))
                for i in range(1,self.n_hidden_layers):
                    h = self.actfun(self[0][i](h))
                y = self[0][-1](h)

        return y

#####
## Convolutional Neural Network

class ConvNet(Chain, Network):
    """
    Basic convolutional neural network
    """

    def __init__(self, n_input, n_output, n_hidden=10):
        """

        :param n_input: nchannels x height x width
        :param n_output: number of action outputs
        :param n_hidden: number of hidden units
        :param monitor: monitors internal states
        """

        k = 3 # kernel size
        s = 1 # stride
        p = 1 # padding
        n_linear = n_hidden * np.prod(1 + (np.array(n_input[1:]) - k + 2*p)/s)
        super(ConvNet, self).__init__(
            l1=L.Convolution2D(n_input[0], n_hidden, k, s, p),
            l2=L.Linear(n_linear, n_output)
        )

        self.ninput = n_input
        self.nhidden = n_hidden
        self.noutput = n_output
        self.monitor = None

    def __call__(self, x, train=False):
        """
        :param x: sensory input (ntrials x nchannels x ninput[0] x ninput[1])
        """

        if self.monitor:

            h = F.relu(self.l1(x))
            self.monitor.set('hidden-1', h.data)
            y = self.l2(h)
            self.monitor.set('output', y.data)

        else:

            y = self.l2(F.relu(self.l1(x)))

        return y

#####
## Recurrent Neural Network

class RNN(ChainList, Network):
    """
    Recurrent neural network consisting of a chain of layers (weight matrices)
    with a fixed number of nhidden units

    nlayer determines number of layers. The last layer is always a linear layer. The other layers
    make use of an activation function actfun

    """

    def __init__(self, n_input, n_output, n_hidden=10, n_hidden_layers=1, link=L.LSTM):
        """

        :param n_input: number of inputs
        :param n_hidden: number of hidden units
        :param n_output: number of outputs
        :param n_hidden_layers: number of hidden layers
        :param link: used recurrent link (LSTM)

        """

        links = ChainList()
        if n_hidden_layers == 0:
            links.add_link(L.Linear(n_input, n_output))
        else:
            links.add_link(link(n_input, n_hidden))
            for i in range(n_hidden_layers - 1):
                links.add_link(link(n_hidden, n_hidden))
            links.add_link(L.Linear(n_hidden, n_output))

        self.ninput = n_input
        self.nhidden = n_hidden
        self.noutput = n_output
        self.n_hidden_layers = n_hidden_layers
        self.monitor = None

        super(RNN, self).__init__(links)

    def __call__(self, x, train=False):

        if self.n_hidden_layers == 0:

            y = self[0][0](x)

        else:

            if self.monitor:

                h = self[0][0](x)
                self.monitor.set('hidden-1', h.data)
                for i in range(1,self.n_hidden_layers):
                    h = self[0][i](h)
                    self.monitor.set('hidden-'+str(i+1), h.data)
                y = self[0][-1](h)
                self.monitor.set('output', y.data)

            else:

                h = self[0][0](x)
                for i in range(1, self.n_hidden_layers):
                    h = self[0][i](h)
                y = self[0][-1](h)

        return y

    def reset_state(self):
        for i in range(self.n_hidden_layers):
            self[0][i].reset_state()

#####
## Language model

class RNNForLM(Chain, Network):

    def __init__(self, n_vocab, n_hidden):

        super(RNNForLM, self).__init__(
            embed=L.EmbedID(n_vocab, n_hidden),
            l1=L.LSTM(n_hidden, n_hidden),
            l2=L.Linear(n_hidden, n_vocab),
        )

        for param in self.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)

    def __call__(self, x, train=False):

        h0 = self.embed(x)
        h1 = self.l1(F.dropout(h0, train=train))
        y = self.l2(F.dropout(h1, train=train))

        return y

    def reset_state(self):
        self.l1.reset_state()