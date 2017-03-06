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

        # reset state
        self.reset()

    def run(self, data, train=True, idx=None, final=False):

        # get reward associated with taking the previous action in the previous state
        reward = data[-1]

        # determine if we already completed a perception-action cycle
        if not self._score is None:

            # update return using reward
            if self._return is None:
                self._return = self.gamma * Variable(reward)
            else:
                self._return += self.gamma * Variable(reward)

            # add minus the score function times the value based on (s_t-1, a_t-1, r_t-1) to loss
            loss = - F.squeeze(F.sum(F.batch_matmul(self._score, self._return, transa=True), axis=0))

            # add entropy term
            loss -= self.beta * self._entropy

            self.loss += loss

            # normalize by number of datapoints in minibatch
            _loss = float(loss.data / data[0].shape[0])

        else:
            _loss = 0.0

        # compute policy and take new action based on observations
        self.action, policy, _ = self.model(map(lambda x: Variable(self.xp.asarray(x)), data), train=train)

        # recompute score function: grad_theta log pi_theta (s_t, a_t) * v_t
        self._score = self.score_function(self.action, policy)

        # compute entropy
        self._entropy = F.sum(self.entropy(policy))

        # backpropagate if we reach the cutoff for truncated backprop or if we processed the last batch
        if train and ((self.cutoff and (idx % self.cutoff) == 0) or final):

            self.optimizer.zero_grads()
            self.loss.backward()
            self.loss.unchain_backward()
            self.optimizer.update()

            # store return
            if self.monitor:
                map(lambda x: x.set('return', np.mean(self._return.data)), self.monitor)

            self._return = None

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


#####
## Advantage Actor-Critic algorithm
#
