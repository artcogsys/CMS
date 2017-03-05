from base import Iterator
import numpy as np

###
# Simple MDP task

class Foo(Iterator):
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

    def __iter__(self):

        self.idx = 0
        self.state = np.random.choice(2, [self.batch_size, 1], True, [0.5, 0.5]).astype('int32')
        self.reward = 0

        return self

    def next(self):

        if self.idx == self.n_batches:
            raise StopIteration

        self.idx += 1

        # produce a new observation at each step
        obs = np.zeros([self.batch_size, self.n_input]).astype('float32')
        for i in range(self.batch_size):

            if self.state[i] == 0:
                obs[i] = np.random.choice(2, [1, self.n_input], True, np.array([self.p, 1 - self.p]))
            else:
                obs[i] = np.random.choice(2, [1, self.n_input], True, np.array([1 - self.p, self.p]))

        # return new observation and reward associated with agent's choice
        return obs, self.reward

    def process(self, agent):

        if self.is_final():
            self.reward = 0
        else:
            # reward is +1 or -1
            self.reward = (2 * (agent.action == self.state) - 1).astype('float32')

        # this task always produces a new observation after each decision
        self.state = np.random.choice(2, [self.batch_size, 1], True, [0.5, 0.5]).astype('int32')

###
# Probabilistic categorization task

class ProbabilisticCategorizationTask(Iterator):
    """

    Let odds be a vector determining the odds ratio for emitting a certain symbol x = i given state k = j:

        odds = [ P(x = 0 | k = 1) / P(x = 0 | k = 0) ... P(x = n | k = 1) / P(x = n | k = 0) ]

    We define

        p = odds / sum(odds)
        q = (1/odds) / sum(1/odds)

    Let P(x = i | p, q, k) define the probability that the emitted symbol is i given that we have probability vector p and q and
    the true state can be either k = 0 or k = 1. Then

        P(x = i | p, k) = p^k * q^(k-1)

    Note: vector of zeros indicates absence of evidence (starting state)

    Note: a nicer way to generalize this to 2D input is to have the 2D input be a very noisy version of the underlying stimulus
          this makes it an object categorization task. We then have one knob to turn (noise level)

    Note: This code has now been generalized so it can also be used in supervised training. The parameter nsteps when not none indicates
          how many steps are taken in each trial

    """

    def __init__(self, odds = [0.25, 0.75, 1.5, 2.5], nsteps = None, batch_size=1, n_batches=np.inf):
        """

        :param: odds : determines odds ratio

        """

        super(Foo, self).__init__(batch_size=batch_size, n_batches=n_batches)

        self.odds = np.array(odds)

        # keep track of the iteration
        self.idx = 0

        self.p = self.odds/float(np.sum(self.odds))
        self.q = (1.0/self.odds)/float(np.sum(1.0/self.odds))

        self.ninput = len(self.p)
        self.noutput = 3 # number of output variables

        self.rewards = [-1, 15, -100]

        # normalize rewards
        self.rewards = np.array(self.rewards, dtype='float32') / np.max(np.abs(self.rewards)).astype('float32')


    def __iter__(self):

        self.idx = 0
        self.state = np.int32(np.random.randint(1, 3))  # 1 = left, 2 = right
        self.reward = 0

        return self

    def next(self):

        if self.idx == self.n_batches:
            raise StopIteration

        self.idx += 1

        # produce a new observation at each step
        obs = np.zeros([self.batch_size, self.n_input]).astype('float32')

        for i in range(self.batch_size):

            if self.state[i] == 0:
                obs[i] = np.random.choice(2, [1, self.n_input], True, np.array([self.p, 1 - self.p]))
            else:
                obs[i] = np.random.choice(2, [1, self.n_input], True, np.array([1 - self.p, self.p]))

        # return new observation and reward associated with agent's choice
        return obs, self.reward

    def process(self, agent):

        # convert 1-hot encoding to discrete action
        action = np.argmax(agent.action)

        if action == 0:  # wait to get new evidence

            self.reward = self.rewards[0]

            self.terminal = np.float32(0)

        else:  # left or right was chosen

            if action == self.state:
                self.reward = self.rewards[1]
            else:
                self.reward = self.rewards[2]

            self.terminal = np.float32(1)

            obs, target = self.reset()