from base import Iterator
import numpy as np
import copy

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
        self.n_output = 2 # number of output variables (actions) for the agent (discrete case)
        self.n_states = 1 # number of state variables

    def __iter__(self):

        self.idx = -1
        self.state = self.get_state()
        self.obs = self.get_observation()
        self.reward = None

        return self

    def next(self):

        if self.idx == self.n_batches-1:
            raise StopIteration

        self.idx += 1

        # return new observation and reward associated with agent's choice
        return self.obs, self.reward

    def process(self, agent):
        """ Process agent action, compute reward and generate new state and observation

        :param agent:
        :return:
        """

        if self.monitor:
            map(lambda x: x.set('accuracy', 1.0 * np.sum(agent.action == self.state) / agent.action.size), self.monitor)

        self.reward = (2 * (agent.action == self.state) - 1).astype('float32')

        # this task always produces a new observation after each decision
        self.state = self.get_state()

        self.obs = self.get_observation()

    def get_state(self):
        """

        :return: new state
        """

        return np.random.choice(2, self.batch_size, True, [0.5, 0.5]).astype('int32')

    def get_observation(self):
        """

        :return: observation given the state
        """

        # produce a new observation at each step
        obs = np.zeros([self.batch_size, self.n_input]).astype('float32')
        for i in range(self.batch_size):

            if self.state[i] == 0:
                obs[i] = np.random.choice(2, [1, self.n_input], True, np.array([self.p, 1 - self.p]))
            else:
                obs[i] = np.random.choice(2, [1, self.n_input], True, np.array([1 - self.p, self.p]))

        return obs
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

    def __init__(self, odds = [0.25, 0.75, 1.5, 2.5], batch_size=1, n_batches=np.inf):
        """

        :param: odds : determines odds ratio

        """

        super(ProbabilisticCategorizationTask, self).__init__(batch_size=batch_size, n_batches=n_batches)

        self.odds = np.array(odds)

        # keep track of the iteration
        self.idx = 0

        self.p = self.odds/float(np.sum(self.odds))
        self.q = (1.0/self.odds)/float(np.sum(1.0/self.odds))

        self.n_input = len(self.p)
        self.n_output = 3 # number of output variables (actions)

        self.rewards = [-1, 15, -100]

        # normalize rewards
        self.rewards = np.array(self.rewards, dtype='float32') / np.max(np.abs(self.rewards)).astype('float32')

        self.state = None
        self.obs = None
        self.reward = None

    def __iter__(self):

        self.idx = -1
        self.state = np.int32(np.random.randint(1, 3, size=[self.batch_size,1]))
        self.obs = np.zeros([self.batch_size, self.n_input], dtype='float32')
        self.reward = None

        return self

    def next(self):

        if self.idx == self.n_batches-1:
            raise StopIteration

        self.idx += 1

        # return new observation and reward associated with agent's choice
        return self.obs, self.reward

    def process(self, agent):

        if self.monitor:
            map(lambda x: x.set('accuracy', 1.0 * np.sum(agent.action == self.state) / agent.action.size), self.monitor)

        obs = np.zeros([self.batch_size, self.n_input], dtype='float32')

        self.reward = np.zeros(len(agent.action), dtype=np.float32)

        # handle cases where new piece of evidence is requested
        idx = np.where(agent.action == 0)[0]
        self.reward[idx] = self.rewards[0]
        for i in idx:
            if self.state[i]==1:
                evidence = np.random.choice(self.n_input, p=self.p)
            else:
                evidence = np.random.choice(self.n_input, p=self.q)
            obs[i,evidence] = 1

        # handle cases where left or right was chosen
        idx = np.where(agent.action != 0)[0]
        for i in idx:

            if agent.action[i] == self.state[i]:
                self.reward[i] = self.rewards[1]
            else:
                self.reward[i] = self.rewards[2]

            self.state[i] = np.int32(np.random.randint(1, 3))

        self.obs = obs

#####
## Data task - takes a dataset and keeps spitting out the same (corrupted) stimulus until an agent decides on the category

class DataTask(Iterator):

    def __init__(self, data, batch_size=1, n_batches=np.inf, noise=0, rewards=[-1, 10, -10]):

        batch_size = batch_size or len(data)

        super(DataTask, self).__init__(batch_size=batch_size, n_batches=n_batches)

        self.data = data

        self.n_samples = len(data)
        self.n_input = data[0][0].size

        # number of actions. Last action is the decision to accumulate more information
        self.n_output = np.unique(map(lambda x: x[1], data)).size + 1

        # flags noise level
        self.noise = noise

        # rewards/costs for asking for a new observation, deciding on the right category, deciding on the wrong category
        self.rewards = rewards

        # normalize rewards
        self.rewards = np.array(self.rewards, dtype='float32') / np.max(np.abs(self.rewards)).astype('float32')

        self.state = None
        self.obs = None
        self.reward = None

    def __iter__(self):

        self.idx = -1

        # generate another random batch in each epoch
        _order = np.random.permutation(self.n_samples)[:self.batch_size]

        # keep track of true class
        self.state = self.data[_order][1]

        # generate data
        self.obs = self.data[_order][0]

        self.reward = None

        return self

    def next(self):

        if self.idx == self.n_batches-1:
            raise StopIteration

        self.idx += 1

        if self.monitor:
            map(lambda x: x.set('state', copy.copy(self.state)), self.monitor)

        return self.add_noise(self.obs), self.reward

    def add_noise(self, data):

        d_shape = data.shape
        d_size = data.size

        # create noise component
        noise = np.zeros(d_size)
        n = int(np.ceil(self.noise * d_size))
        noise[np.random.permutation(d_size)[:n]] = np.random.rand(n)
        noise = noise.reshape(d_shape)

        data[noise != 0] = noise[noise != 0]

        return data

    def process(self, agent):
        """ Process agent action, compute reward and generate new state and observation

        :param agent:
        :return:
        """

        if self.monitor:
            map(lambda x: x.set('accuracy', 1.0*np.sum(agent.action == self.state)/agent.action.size), self.monitor)

        self.reward = np.zeros(len(agent.action), dtype=np.float32)

        # handle cases in minibatch where action was to choose the correct category
        true_idx = np.where(agent.action.squeeze() == self.state)[0]

        # add rewards
        self.reward[true_idx] = self.rewards[1]

        # handle cases in minibatch where action was to choose the incorrect category
        false_idx = np.where(agent.action.squeeze() != self.state)[0]

        # add rewards
        self.reward[false_idx] = self.rewards[2]

        # handle cases in minibatch where action is to ask for another observation
        wait_idx = np.where(agent.action.squeeze() == self.n_output-1)[0]

        # cost associated with asking for a new observation
        self.reward[wait_idx] = self.rewards[0]

        # create new states and observations for true_idx and false_idx

        update_idx = np.setdiff1d(np.arange(self.batch_size), wait_idx)

        _order = np.random.permutation(self.n_samples)[:update_idx.size]
        self.state[update_idx] = self.data[_order][1]
        self.obs[update_idx] = self.data[_order][0]
