from __future__ import division

import os
import numpy as np
import tqdm
from brain.monitor import Oscilloscope

#####
## An iterator generates batches

class Iterator(object):

    def __init__(self, batch_size=None, n_batches=None):

        self.batch_size = batch_size
        self.n_batches = n_batches

        # index of current batch
        self.idx=0

    def __iter__(self):

        self.idx = 0

        return self

    def next(self):
        """

        :return: a list of numpy arrays where the first dimension is the minibatch size
        """

        raise NotImplementedError

    def is_final(self):
        """

        :return: boolean if final batch is reached
        """
        return (self.idx==self.n_batches)

    def process(self, agent):
        """
        Processing the possible actions of an agent
        By default this has no effect

        :param agent:
        :return:
        """

        raise NotImplementedError

    def render(self, agent):
        """TO DO: Rendering function to track input-output over time

        :param agent:
        :return:
        """
        pass

#####
## World class

class World(object):

    def __init__(self, agents, out='result'):
        """ A world is inhabited by one or more agents

        :param agents:
        :param out: output folder
        """

        if not isinstance(agents, list):
            self.agents = [agents]
        else:
            self.agents = agents

        self.out = out
        if not self.out is None:
            try:
                os.makedirs(self.out)
            except OSError:
                pass

        self.n_agents = len(self.agents)

        # optional labels for plotting
        # Note that validate first plots the training losses and then the validation losses
        self.labels = None

    def save_snapshot(self, idx):
        for i in range(self.n_agents):
            self.agents[i].model.save(os.path.join(self.out, 'agent-{0:04d}-snapshot-{1:04d}'.format(i, idx)))

    def train(self, data_iter, n_epochs=1, plot=0, snapshot=0, monitor=0):
        self.run(data_iter, validation=None, train=True, n_epochs=n_epochs, plot=plot, snapshot=snapshot, monitor=monitor)

    def test(self, data_iter, n_epochs=1, plot=0, snapshot=0, monitor=0):
        self.run(data_iter, validation=None, train=False, n_epochs=n_epochs, plot=plot, snapshot=snapshot, monitor=monitor)

    def validate(self, data_iter, validation, n_epochs=1, plot=0, snapshot=0, monitor=0):
        self.run(data_iter, validation=validation, train=True, n_epochs=n_epochs, plot=plot, snapshot=snapshot, monitor=monitor)

    def run(self, data_iter, validation=None, train=False, n_epochs=1, plot=0, snapshot=0, monitor=0):
        """ Used to train, test and validate a model. Generalizes to the use of multiple agents

        :param data_iter: environment to train on
        :param validation: environment to validate on
        :param train: run in train or test mode
        :param n_epochs: number of epochs to run an environment
        :param plot: plot change in loss - -1 : per epoch; > 0 : after this many iterations
        :param snapshot: save snapshot - -1 : per epoch; > 0 : after this many iterations
        :param monitor: execute monitor.run() - -1 : per epoch; > 0 : after this many iterations
        :return:
        """

        if plot:
            if not validation is None:
                loss_monitor = Oscilloscope(['training', 'validation'], ylabel='loss')
            else:
                loss_monitor = Oscilloscope(['training'], ylabel='loss')

        # initialization for validation
        min_loss = [None] * self.n_agents
        optimal_model = [None] * self.n_agents

        # iterate over epochs
        for epoch in tqdm.tqdm(xrange(0, n_epochs)):

            # reset agents at start of each epoch
            map(lambda x: x.reset(), self.agents)

            cum_loss = np.zeros(self.n_agents)

            # iterate over batches
            for data in data_iter:

                losses = map(lambda x: x.run(data, train=train, idx=data_iter.idx, final=data_iter.is_final()),
                             self.agents)

                # the iterator can process actions of the agents that change the state of the iterator
                # used in case of processing of tasks by RL agents
                map(lambda x: data_iter.process(x), self.agents)

                idx = data_iter.idx if np.isinf(
                    data_iter.n_batches) else data_iter.idx + epoch * data_iter.n_batches

                if plot > 0 and idx % plot == 0:
                    loss_monitor.set('training', losses)
                    loss_monitor.run()

                if snapshot > 0 and idx % snapshot == 0:
                    self.save_snapshot(idx)

                # if monitor is defined then run optional monitoring function
                if monitor > 0 and idx % monitor == 0:
                    map(lambda x: map(lambda z: z.run(), x.monitor) if x.monitor else None, self.agents)

                cum_loss += losses

            # validate model (only possible for environments that run for a fixed number of steps)
            if not validation is None:

                map(lambda x: x.reset(), self.agents)

                val_loss = np.zeros(self.n_agents)

                for data in validation:

                    losses = map(lambda x: x.run(data, train=False), self.agents)

                    map(lambda x: data_iter.process(x), self.agents)

                    val_loss += losses

                # store best models in case we are validating
                for i in range(self.n_agents):

                    if min_loss[i] is None:
                        optimal_model[i] = self.agents[i].optimizer.target.copy()
                        min_loss[i] = val_loss[i]
                    else:
                        if val_loss[i] < min_loss[i]:
                            optimal_model[i] = self.agents[i].optimizer.target.copy()
                            min_loss[i] = val_loss[i]

            # plot cumulative loss averaged over number of batches of each agent
            if plot == -1:
                loss_monitor.set('training', cum_loss / data_iter.n_batches)
                if not validation is None:
                    loss_monitor.set('validation', val_loss / validation.n_batches)
                loss_monitor.run()

            # store snapshot
            if snapshot == -1:
                self.save_snapshot(epoch)

            # if monitor is defined then run optional monitoring function
            if monitor == -1:
                map(lambda x: map(lambda z: z.run(), x.monitor) if x.monitor else None, self.agents)

        # each agent is assigned the best 'brain' according to validation loss
        if not validation is None:
            for i in range(self.n_agents):
                self.agents[i].model = optimal_model[i]
                self.agents[i].optimizer.target = optimal_model[i]

        if plot:
            loss_monitor.save(os.path.join(self.out, 'loss'))
