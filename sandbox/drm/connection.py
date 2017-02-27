# DRM Connections

from brain.networks import *

#####
## Default connection object

class DRMConnection(Chain, Network):
    """
    Mechanism which specifies how a population interacts. For now just an identity mapping
    """

    def __call__(self, x, train=False):
        return x

    def reset_state(self):
        pass

#####
## Default readout mechanism

class DRMReadout(Chain, Network):

    def __init__(self):

        super(DRMReadout, self).__init__(
            l1=L.LSTM(None, 1),
            l2=L.Linear(1, 1)
        )

    def __call__(self, x, train=False):
        raise NotImplementedError


    def reset_state(self):
        raise NotImplementedError
