from __future__ import division

from chainer import Variable
from agent.base import Agent
from brain.monitor import Monitor
import numpy as np
import chainer.functions as F

#####
## Actor-critic agent base class
#

class ActorCriticAgent(Agent):

    def score_function(self, action, pi):

        logp = F.log_softmax(pi)

        score = F.select_item(logp, Variable(action.squeeze()))

        # handle case where we have only one element per batch
        if score.ndim == 1:
            score = F.expand_dims(score, axis=1)

        return score

    def entropy(self, pi):

        p = F.softmax(pi)
        logp = F.log_softmax(pi)

        return - F.sum(p * logp, axis=1)


#####
## REINFORCE algorithm
#

class REINFORCEAgent(ActorCriticAgent):
    """
    Note that REINFORCE is a policy gradient method which does not use a critic.
    Instead the return is computed as a running estimate

    https://webdocs.cs.ualberta.ca/%7Esutton/book/bookdraft2016sep.pdf
    https://github.com/dennybritz/reinforcement-learning/tree/master/PolicyGradient
    http://blog.shakirm.com/2015/11/machine-learning-trick-of-the-day-5-log-derivative-trick/
    http://www.1-4-5.net/~dmm/ml/log_derivative_trick.pdf
    """

    def __init__(self, model, optimizer=None, gpu=-1, cutoff=None, gamma=0.99, beta=1e-2):

        super(REINFORCEAgent, self).__init__(model, optimizer=optimizer, gpu=gpu)

        # cutoff for truncated BPTT
        self.cutoff = cutoff

        # discounting factor
        self.gamma = gamma

        # contribution of entropy term
        self.beta = beta

        # monitor score, entropy and reward
        self.buffer = Monitor()

        # reset state
        self.reset()

    def run(self, data, train=True, idx=None, final=False):
        """

        :param data: a new observation and the reward associated with the previous observation and action
        :param train:
        :param idx:
        :param final:
        :return:
        """

        # get reward associated with taking the previous action in the previous state
        reward = data[-1]
        if not reward is None:
            self.buffer.set('reward', reward)

        # compute policy and take new action based on observations
        self.action, policy, _ = self.model(map(lambda x: Variable(self.xp.asarray(x)), data), train=train)

        # recompute score function: grad_theta log pi_theta (s_t, a_t) * v_t
        self.buffer.set('score', self.score_function(self.action, policy))

        # compute entropy
        self.buffer.set('entropy', F.sum(self.entropy(policy)))

        # backpropagate if we reach the cutoff for truncated backprop or if we processed the last batch
        if train and ((self.cutoff and (idx % self.cutoff) == 0) or final):

            # remove last results since we have no reward associated
            self.buffer.dict['score'].pop()
            self.buffer.dict['entropy'].pop()

            # return associated with last state
            _return = 0

            loss = Variable(self.xp.zeros((), 'float32'))
            for i in range(len(self.buffer.dict['reward'])-1,-1,-1):

                _return = self.buffer.dict['reward'].pop() + self.gamma * _return

                loss -= F.squeeze(F.sum(F.batch_matmul(self.buffer.dict['score'].pop(), _return, transa=True), axis=0))

                loss -= self.beta * self.buffer.dict['entropy'].pop()

                _loss = loss.data

                self.optimizer.zero_grads()
                loss.backward()
                loss.unchain_backward()
                self.optimizer.update()

            # store return
            if self.monitor:
                map(lambda x: x.set('return', np.mean(_return)), self.monitor)

        else:

            _loss = 0

        return _loss


    def reset(self):

        self.model.predictor.reset_state()

        # keep track of loss for BPTT
        self.loss = Variable(self.xp.zeros((), 'float32'))

        # keep track of action by agent
        self.action = None

        ## keep track of history

        self._score = None
        self._entropy = None
        self._return = None

        self.buffer = Monitor()



#####
## Advantage Actor-Critic algorithm
#
