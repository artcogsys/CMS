from brain.networks import *
from chainer import Variable, Link

#####
## Default connection object - an identity mapping

class DRMConnection(Chain, Network):
    """
    Mechanism which specifies how a population interacts.
    """

    def __call__(self, x, train=False):
        return x





# class DRMConnection(Link):
#     """
#     Mechanism which specifies how a population interacts. For now just a delayed identity mapping.
#     Note that the delay is necessary in order to make the computation graph acyclic.
#     """
#
#     def __init__(self, delay=1):
#         """
#
#         :param delay: Conduction delay in terms of number of sampling steps
#         """
#
#         self.delay = delay
#         self.history = None
#
#     def __call__(self, x, train=False):
#
#         if self.history is None:
#             self.history = [Variable(np.zeros(x.shape, dtype='float32')) for i in range(self.delay)]
#         else:
#             self.history.append(x)
#             y = self.history[0]
#             self.history = self.history[1:]
#
#         return y
#
#     def reset_state(self):
#         self.history = None

