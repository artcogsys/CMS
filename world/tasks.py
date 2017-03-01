from iterators import TaskIterator
import numpy as np

###
# Specific environments

class Foo(TaskIterator):
    """
    Very simple environment for testing fully observed models. The actor gets a reward when it correctly decides
    on the ground truth. Ground truth 0/1 determines probabilistically the number of 0s or 1s as observations
    """

    def __init__(self, n = 2, p = 0.8, batch_size=1, n_batches=np.inf):
        """

        :param n: number of inputs
        :param p: probability of emitting the right sensation at the input
        :param batch_size: one run by default
        :param n_batches: infinite number of time steps by default
        """

        super(Foo, self).__init__(batch_size=batch_size, n_batches=n_batches)

        self.n_input = n
        self.p = p

        self.n_action = 1 # number of action variables
        self.n_output = 2 # number of output variables for the agent (discrete case)
        self.n_states = 1 # number of state variables

        self.reset()

    def reset(self):
        """

        Returns: observation

        """

        self.state = np.random.randint(0, 2)

        p = np.array([1 - self.p, self.p])

        if self.state == 0:
            p = 1 - p

        obs = np.random.choice(2, [1, self.n_input], True, p)

        # the action that was performed by the agent
        self.action = None

        return obs.astype(np.float32)

    def next(self):

        if self.idx == self.n_batches:
            raise StopIteration

        # reward is +1 or -1
        reward = 2 * int(self.action == self.state) - 1

        # this task always produces a new observation after each decision
        obs = self.reset()

        return obs, reward

    # def get_ground_truth(self):
    #     """
    #     Returns: ground truth state of the environment
    #     """
    #
    #     return self.state
    #
    # def set_ground_truth(self, ground_truth):
    #     """
    #     :param: ground_truth : sets ground truth state of the environment
    #     """
    #
    #     self.state = ground_truth