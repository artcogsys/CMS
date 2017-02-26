# Implementation of dynamic representational modeling. We assume the existence of a recurrent network containing the
# 'representations'. Each neural population has its own (set of) RNN units. Each population projects to one output
# variable (e.g. its BOLD response).

from agent.supervised import StatefulAgent
from brain.models import *
from brain.networks import *
from world.base import World
from world.iterators import *

#####
## DRM

# focus: fast/scalable

class DRM(object):

    def __init__(self, data_iter, populations, readout, W_stim=None, W_pop=None, W_resp=None):
        """

        :param data_iter: iterator which generates sensations/responses at some specified resolution
        :param populations: list of populations (neural networks)
        :param readout: list of readout objects (neural networks)
        :param resolution: temporal resolution for simulation in ms
        :param stim_time: times as which stimuli were presented relative to start of simulation t=0
        :param resp_time: times as which responses were observed relative to start of simulation t=0
        :param W_stim: n_pop x n_stim matrix specifying how each input connects to each population (full if None)
        :param W_pop: n_pop x n_pop matrix specifying how each population connects to each population (full if None)
        :param W_resp: n_resp x n_pop matrix specifying how each population connects to each output (full if None)
        """

        self.data_iter = data_iter
        self.populations = populations
        self.n_in = self.data_iter.n_in
        self.n_pop = len(populations)
        self.n_out = self.data_iter.n_out
        self.W_stim = W_stim or np.ones([self.n_pop,self.n_in], dtype=bool)
        self.W_pop = W_pop or np.ones([self.n_pop,self.n_pop], dtype=bool)
        self.W_resp = W_resp or np.ones([self.n_out,self.n_pop], dtype=bool)

        self.model = Regressor(DRMNet(populations, readout))

    def run(self):

        # define agent
        agent = StatefulAgent(self.model, chainer.optimizers.Adam())

        # add hook
        agent.optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5))

        # define world
        world = World(agent)

        # run world in training mode with validation
        world.train(self.data_iter, n_epochs=100, plot=-1)

#####
## Sequential iterator

class DRMIterator(Iterator):

    def __init__(self, stimulus, response, resolution=None, stim_time=None, resp_time=None, batch_size=None, n_batches=None):
        """

        :param stimulus: input stimulus - nsamples x d1 x ... numpy array (float32)
        :param response: output responses - nsamples x d1 x ... numpy array (float32)
        :param resolution: temporal resolution for simulation in ms
        :param stim_time: list of times in ms at which stimuli were presented relative to start of simulation t=0
        :param resp_time: list of times in ms at which responses were observed relative to start of simulation t=0
        :param batch_size: number of batches to process sequentially
        :param n_batches: number of time steps to take per batch
        """

        self.stimulus = stimulus
        self.response = response

        self.n_in = self.stimulus[0].size
        self.n_out = self.response[0].size

        # assume sampling at same rate of 1 ms if nothing is specified
        self.resolution = resolution or 1

        if stim_time is None:
            self.stim_time = np.arange(0, len(stimulus)).tolist()
        else:
            self.stim_time = stim_time

        if resp_time is None:
            self.resp_time = np.arange(0, len(response)).tolist()
        else:
            self.resp_time = resp_time

        # check if lengths agree
        assert(len(self.stimulus) == len(self.stim_time))
        assert(len(self.response) == len(self.resp_time))

        # steps in the simulation should be at least as small as the minimal step in stimulus or response time
        assert(np.min(np.diff(self.stim_time)) >= self.resolution)
        assert(np.min(np.diff(self.resp_time)) >= self.resolution)

        # determine total number of time steps to take according to temporal resolution
        self.n_steps = np.ceil(np.max(self.stim_time + self.resp_time) / self.resolution).astype('int32')

        if batch_size is None:
            batch_size = 1

        if n_batches is None:
            n_batches = self.n_steps // batch_size

        super(DRMIterator, self).__init__(batch_size=batch_size, n_batches=n_batches)

    def __iter__(self):

        self.idx = 0

        offsets = [i * self.n_batches for i in range(self.batch_size)]

        # determine time points to sample
        self._order = []
        for iter in range(self.n_batches):
            x = [(offset + iter) % self.n_steps for offset in offsets]
            self._order += x

        # multiply points by temporal resolution
        self._order *= self.resolution

        return self

    def next(self):

        if self.idx == self.n_batches:
            raise StopIteration

        i = self.idx * self.batch_size

        self.idx += 1

        sample_times = self._order[i:(i + self.batch_size)]

        # find closest time point in sensory stream
        idx = map(lambda t: np.where((self.stim_time >= t - self.resolution/2) & (self.stim_time <= t + self.resolution/2))[0], sample_times)

        # create partially observed data (zeros for no input)
        stim_data = np.array(map(lambda x: np.zeros(self.stimulus[0].shape) if len(x) == 0 else self.stimulus[x[0]], idx)).astype('float32')

        # find closest time point in response stream
        idx = map(lambda t: np.where((self.resp_time >= t - self.resolution/2) & (self.resp_time <= t + self.resolution/2))[0], sample_times)

        # create partially observed data (nans for no output)
        resp_data = np.array(map(lambda x: np.full(self.response[0].shape, np.nan) if len(x) == 0 else self.response[x[0]], idx)).astype('float32')

        return [stim_data, resp_data]


class DRMNet(ChainList, Network):

    def __init__(self, populations, readout):

        # build model
        links = ChainList()

        # add populations
        for i in range(len(populations)):
            links.add_link(populations[i])

        # add readout mechanisms
        for i in range(len(readout)):
            links.add_link(readout[i])

        super(DRMNet, self).__init__(links)

    def __call__(self, x, train=False):
        pass

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

        pass

    def reset_state(self):
        pass

#####
## Readout mechanism

class DRMReadout(Chain, Network):

    def __init__(self):

        super(DRMReadout, self).__init__(
            l1=L.LSTM(None, 1),
            l2=L.Linear(1, 1)
        )

    def __call__(self, x, train=False):
        pass


    def reset_state(self):
        pass
