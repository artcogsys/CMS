# DRM Connections

from brain.networks import *

#####
## Neuronal population

class DRMPopulation(Chain, Network):

    def __init__(self, n_hidden=1, n_output=1):
        """

        :param n_hidden: number of hidden units
        :param n_output: number of outputs that are sent by this model
        """

        super(DRMPopulation, self).__init__(
            l1=Elman(None, n_hidden),
            l2=L.Linear(n_hidden, n_output)
        )

    def __call__(self, x, train=False):
        raise NotImplementedError

    def reset_state(self):
        raise NotImplementedError
