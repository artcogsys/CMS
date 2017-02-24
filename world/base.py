from __future__ import division

import os
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import tools

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

    def run(self, data_iter, train, n_epochs=1, plot=False, snapshot=False, per_epoch=True):
        """ run in training or test mode; training supports life-long learning given infinite environment

        :param data_iter: environment to run on
        :param train: run in train mode if True and test mode if False
        :param n_epochs: number of epochs to run an environment
        :param plot: plot change in loss
        :param snapshot: save snapshot every epoch
        :param per_epoch: count per epoch if True and per iteration if False
        :return:
        """

        gfx = None

        # iterate over epochs
        for epoch in tqdm.tqdm(xrange(0, n_epochs)):

            # reset agents at start of each epoch
            map(lambda x: x.reset(), self.agents)

            cum_loss = np.zeros(self.n_agents)

            # iterate over batches
            for data in data_iter:

                losses = map(lambda x: x.run(data, train=train, idx=data_iter.idx, final=data_iter.is_final()), self.agents)

                if not per_epoch:

                    idx = data_iter.idx if np.isinf(data_iter.n_batches) else data_iter.idx + epoch * data_iter.n_batches

                    if plot:
                        gfx = tools.plot_loss(gfx, idx, losses, self.labels)

                    if snapshot and data_iter.idx % snapshot == 0:
                        self.save_snapshot(idx)

                cum_loss += losses

            # plot cumulative loss averaged over number of batches of each agent and store snapshot
            if per_epoch:
                if plot:
                    gfx = tools.plot_loss(gfx, epoch, cum_loss / data_iter.n_batches, self.labels)
                if snapshot and epoch % snapshot == 0:
                    self.save_snapshot(epoch)

        if plot and gfx[0]:
            gfx[0].savefig(os.path.join(self.out, 'loss'))
            plt.close()

    def save_snapshot(self, idx):
        for i in range(self.n_agents):
            self.agents[i].model.save(os.path.join(self.out, 'agent-{0:04d}-snapshot-{1:04d}'.format(i, idx)))

    def train(self, data_iter, n_epochs=1, plot=False, snapshot=False, per_epoch=True):
        self.run(data_iter, True, n_epochs=n_epochs, plot=plot, snapshot=snapshot, per_epoch=per_epoch)

    def test(self, data_iter, n_epochs=1, plot=False, snapshot=False, per_epoch=True):
        self.run(data_iter, False, n_epochs=n_epochs, plot=plot, snapshot=snapshot, per_epoch=per_epoch)

    def validate(self, data_iter, validation, n_epochs=1, plot=False, snapshot=False, per_epoch=True):
        """ Used to train a model and test it such as to determine the best model; this model will become
        the brain of the agent. Generalizes to the use of multiple agents

        :param data_iter: environment to train on
        :param validation: environment to validate on
        :param n_epochs: number of epochs to run an environment
        :param plot: plot change in loss every nth step indicated by this parameter
        :param snapshot: save snapshot every epoch
        :param per_epoch: count per epoch if True and per iteration if False
        :return:
        """

        gfx = None

        # initialization
        min_loss = [None] * self.n_agents
        optimal_model = [None] * self.n_agents

        # iterate over epochs
        for epoch in tqdm.tqdm(xrange(0, n_epochs)):

            # reset agents at start of each epoch
            map(lambda x: x.reset(), self.agents)

            cum_loss = np.zeros(self.n_agents)

            # iterate over batches
            for data in data_iter:

                losses = map(lambda x: x.run(data, train=True, idx=data_iter.idx, final=data_iter.is_final()), self.agents)

                if not per_epoch:

                    idx = data_iter.idx if np.isinf(data_iter.n_batches) else data_iter.idx + epoch * data_iter.n_batches

                    if plot:
                        gfx = tools.plot_loss(gfx, idx, losses, self.labels)

                    if snapshot and data_iter.idx % snapshot == 0:
                        self.save_snapshot(idx)

                cum_loss += losses

            # validate model (only possible for environments that run for a fixed number of steps)

            map(lambda x: x.reset(), self.agents)

            val_loss = np.zeros(self.n_agents)

            for data in validation:
                losses = map(lambda x: x.run(data, train=False), self.agents)
                val_loss += losses

            # plot cumulative loss averaged over number of batches of each agent
            if per_epoch:
                if plot:
                    gfx = tools.plot_loss(gfx, epoch,
                                    np.concatenate([cum_loss / data_iter.n_batches, val_loss / validation.n_batches]),
                                    self.labels)
                if snapshot and epoch % snapshot == 0:
                    self.save_snapshot(epoch)

            # store best models in case we are validating
            for i in range(self.n_agents):

                if min_loss[i] is None:
                    optimal_model[i] = self.agents[i].optimizer.target.copy()
                    min_loss[i] = val_loss[i]
                else:
                    if val_loss[i] < min_loss[i]:
                        optimal_model[i] = self.agents[i].optimizer.target.copy()
                        min_loss[i] = val_loss[i]

        # each agent is assigned the best 'brain' according to validation loss
        for i in range(self.n_agents):
            self.agents[i].model = optimal_model[i]
            self.agents[i].optimizer.target = optimal_model[i]

        if plot and gfx[0]:
            gfx[0].savefig(os.path.join(self.out, 'loss'))
            plt.close()
