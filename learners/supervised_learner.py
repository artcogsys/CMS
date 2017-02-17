from chainer import Variable
from base import Trainer
import os

#####
## Stateless trainer

class StatelessTrainer(Trainer):
    """
    Used to train stateless models that have no persistent variables
    """

    def run(self, data=None):
        """One training epoch

        :param data: iterator
        :return: loss
        """

        if data is None:
            data = self.data

        cumloss = 0

        for batch in data:

            if self.iter_snapshot and self.iteration % self.iter_snapshot == 0:
                self.model.save(os.path.join(self.out, 'iter-snapshot-' + '{0:04d}'.format(self.iteration)))

            x = Variable(self.xp.asarray([item[0] for item in batch]))
            t = Variable(self.xp.asarray([item[1] for item in batch]))

            loss = self.model(x, t, train=True)

            self.optimizer.zero_grads()
            loss.backward()
            self.optimizer.update()

            self.iteration += 1

            cumloss += loss.data

        if self.iter_snapshot and (self.iteration+1) % self.iter_snapshot == 0:
            self.model.save(os.path.join(self.out, 'iter-snapshot-' + '{0:04d}'.format(self.iteration+1)))

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

    def run(self, data=None):
        """One training epoch

        :param data: iterator
        :return: loss
        """

        if data is None:
            data = self.data

        cumloss = 0

        loss = Variable(self.xp.zeros((), 'float32'))

        self.model.predictor.reset_state()

        for batch in data:

            if self.iter_snapshot and self.iteration % self.iter_snapshot == 0:
                self.model.save(os.path.join(self.out, 'iter-snapshot-' + '{0:04d}'.format(self.iteration)))

            x = Variable(self.xp.asarray([item[0] for item in batch]))
            t = Variable(self.xp.asarray([item[1] for item in batch]))

            if self.last:
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

                loss = Variable(self.xp.zeros((), 'float32'))

                self.iteration += 1

        if self.iter_snapshot and (self.iteration + 1) % self.iter_snapshot == 0:
            self.model.save(os.path.join(self.out, 'iter-snapshot-' + '{0:04d}'.format(self.iteration+1)))

        return float(cumloss / data.batch_size)