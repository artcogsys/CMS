from chainer import Variable
from base import Trainer
from analysis.tools import plot_loss

#####
## Stateless trainer

class StatelessTrainer(Trainer):
    """
    Used to train stateless models that have no persistent variables
    """

    def run(self, data=None, plot=False):
        """One training epoch

        :param data: iterator
        :param plot: plot loss
        :return: loss
        """

        if data is None:
            data = self.data

        if plot:
            gfx = [None, None, None]

        cumloss = 0

        for batch in data:

            self.snapshot(data.batch_idx)

            x = Variable(self.xp.asarray([item[0] for item in batch]))
            t = Variable(self.xp.asarray([item[1] for item in batch]))

            loss = self.model(x, t, train=True)

            if plot:
                gfx = plot_loss(gfx[0], gfx[1], gfx[2], data.batch_idx, [loss.data])

            self.optimizer.zero_grads()
            loss.backward()
            self.optimizer.update()

            cumloss += loss.data

        self.snapshot(data.batch_idx+1)

        return float(cumloss / data.batch_size)


class StatefulTrainer(Trainer):
    """
   Used to train stateful models that have persistent variables
   """

    def __init__(self, optimizer, data=None, gpu=-1, cutoff=None, last=False):
        """

        :param optimizer:
        :param data:
        :param gpu:
        :param cutoff: cutoff for BPTT (n_batches if none)
        :param last: whether to only update when the cutoff is reached (False)
        """

        super(StatefulTrainer, self).__init__(optimizer, data, gpu)

        # cutoff for BPTT
        self.cutoff = cutoff

        # whether to update from loss in last step only
        self.last = last

    def run(self, data=None, plot=False):
        """One training epoch

        :param data: iterator
        :return: loss
        """

        if data is None:
            data = self.data

        if plot:
            old_loss = 0
            gfx = [None, None, None]

        cumloss = 0

        loss = Variable(self.xp.zeros((), 'float32'))

        self.model.predictor.reset_state()

        for batch in data:

            self.snapshot(data.batch_idx)

            x = Variable(self.xp.asarray([item[0] for item in batch]))
            t = Variable(self.xp.asarray([item[1] for item in batch]))

            if self.last: # used in case of propagating back at end of trials only
                loss = self.model(x, t, train=True)
            else:
                loss += self.model(x, t, train=True)

            # backpropagate if we reach the cutoff for truncated backprop or if we processed the last batch
            if (self.cutoff and (data.batch_idx % self.cutoff) == 0) or \
                    (data.batch_idx+1 == data.n_batches): # replace with end_of_iter flag

                self.optimizer.zero_grads()
                loss.backward()
                loss.unchain_backward()
                self.optimizer.update()

                cumloss += loss.data

                if plot:
                    gfx = plot_loss(gfx[0], gfx[1], gfx[2], data.batch_idx, [cumloss - old_loss])
                    old_loss = cumloss

                loss = Variable(self.xp.zeros((), 'float32'))

        self.snapshot(data.batch_idx+1)

        return float(cumloss / data.batch_size)