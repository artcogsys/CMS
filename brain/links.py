import numpy as np
import chainer
from chainer.functions.activation import relu
from chainer import link
from chainer.links.connection import linear
import chainer.functions as F
from chainer import initializers
import math
from chainer import cuda

###
# Implementation of custom links and layers


###
# Offset link used in Elman layer to learn initial offset

class Offset(link.Link):
    """
    Implementation of offset term to initialize Elman hidden states at t=0
    """

    def __init__(self, n_params):

        super(Offset, self).__init__()

        self.add_param('X', (1, n_params), initializer=chainer.initializers.Constant(0, dtype='float32'))

    def __call__(self, z):
        return F.broadcast_to(self.X, z.shape)

###
# Implementation Elman layer

class Elman(link.Chain):
    """
    Implementation of simple linear Elman layer

    Consider using initW=chainer.initializers.Identity(scale=0.01)
    as in https://arxiv.org/pdf/1504.00941v2.pdf
    (scale=1.0 led to divergence issues in our example)

    """

    def __init__(self, in_size, out_size, initU=None,
                 initW=None, bias_init=0, actfun=relu.relu):

        super(Elman, self).__init__(
            U=linear.Linear(in_size, out_size,
                            initialW=initU, initial_bias=bias_init),
            W=linear.Linear(out_size, out_size,
                            initialW=initW, nobias=True),
            H0=Offset(out_size),
        )

        self.state_size = out_size
        self.reset_state()
        self.actfun = actfun

    def to_cpu(self):
        super(Elman, self).to_cpu()
        if self.h is not None:
            self.h.to_cpu()

    def to_gpu(self, device=None):
        super(Elman, self).to_gpu(device)
        if self.h is not None:
            self.h.to_gpu(device)

    def set_state(self, h):
        assert isinstance(h, chainer.Variable)
        h_ = h
        if self.xp == np:
            h_.to_cpu()
        else:
            h_.to_gpu()
        self.h = h_

    def reset_state(self):
        self.h = None

    def __call__(self, x):

        z = self.U(x)
        if self.h is not None:
            z += self.W(self.h)
        else:
            z += self.H0(z)

        # must be part of layer since the transformed value is part of the
        # representation of the previous hidden state
        self.h = self.actfun(z)

        return self.h


#####
## Dynamic filter implementation of a linear link; can be generalized to all other links (convolutional, LSTM, Elman, ...)

class DynamicFilterLinear(chainer.Link):

    def __init__(self, predictor, in_size, out_size, constantW=True, wscale=1, initialW=None, bias=0, nobias=False, initial_bias=None):
        """

        :param predictor: a neural network which implements the dynamic filter that maps from context inputs to a weight matrix
        :param in_size: size of input
        :param out_size: size of output
        :param constantW: add constant W such that W = C + DF(x)
        :param wscale: scaling factor
        :param initialW: used for weight initialization
        :param bias:
        :param nobias:
        :param initial_bias:
        """

        # Parameters are initialized as a numpy array of given shape.

        super(DynamicFilterLinear, self).__init__()

        self.predictor = predictor
        self.shape = [out_size, in_size]

        self.in_size = in_size
        self.out_size = out_size

        # add bias term
        if nobias:
            self.b = None
        else:
            if initial_bias is None:
                initial_bias = bias
            bias_initializer = initializers._get_initializer(initial_bias)
            self.add_param('b', out_size, initializer=bias_initializer)

        if constantW:
            self._W_initializer = initializers._get_initializer(
                initialW, math.sqrt(wscale))
        self.constantW = constantW

        if in_size is None:
            self.add_uninitialized_param('C')
        else:
            self._initialize_params(in_size)

    def _initialize_params(self, in_size):
        self.add_param('C', (self.out_size, in_size),
                       initializer=self._W_initializer)

    def __call__(self, x, z):
        """

        Args:
            x (~chainer.Variable): Batch of input vectors.
            z (~chainer.Variable): Batch of context vectors.

        Returns:
            ~chainer.Variable: Output of the context layer.

        """

        if self.has_uninitialized_params:
            with cuda.get_device(self._device_id):
                self._initialize_params(x.size // x.shape[0])

        batch_size = x.shape[0]

        # compute adaptive filter
        W = self.predictor(z)

        # reshape linear W to the correct size
        W = F.reshape(W, [batch_size] + self.shape)

        # add constant W if defined
        if self.constantW:
            W += F.tile(self.C,(batch_size,1,1))

        # multiply weights with inputs in batch mode
        y = F.squeeze(F.batch_matmul(W, x), 2)

        # add bias
        y += F.tile(self.b, tuple([batch_size, 1]))

        return y
