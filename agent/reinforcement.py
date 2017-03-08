from __future__ import division

from chainer import Variable
from agent.base import Agent
from brain.monitor import Monitor
import numpy as np
import chainer.functions as F
import copy

#####
## Actor-critic agent base class
#

class ActorCriticAgent(Agent):

    def score_function(self, action, pi):

        logp = F.log_softmax(pi)

        score = F.select_item(logp, Variable(action))

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

        # cutoff for truncated BPTT
        self.cutoff = cutoff

        # discounting factor
        self.gamma = gamma

        # contribution of entropy term
        self.beta = beta

        # monitor score, entropy and reward
        self.buffer = Monitor()

        super(REINFORCEAgent, self).__init__(model, optimizer=optimizer, gpu=gpu)

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

        # backpropagate if we reach the cutoff for truncated backprop or if we processed the last batch
        if train and idx > 0 and ((self.cutoff and (idx % self.cutoff) == 0) or final):

            # return associated with last state
            _return = 0

            loss = Variable(self.xp.zeros((), 'float32'))
            for i in range(len(self.buffer.dict['reward'])-1,-1,-1):

                _return = self.buffer.dict['reward'].pop() + self.gamma * _return

                _ss = F.squeeze(self.buffer.dict['score'].pop()) * F.squeeze(_return)
                if _ss.size > 1:
                    _ss = F.sum(_ss, axis=0)

                loss -= F.squeeze(_ss)

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

        # recompute score function: grad_theta log pi_theta (s_t, a_t) * v_t
        self.buffer.set('score', self.score_function(self.action, policy))

        # compute entropy
        self.buffer.set('entropy', F.sum(self.entropy(policy)))

        return _loss


    def reset(self):

        self.model.predictor.reset_state()

        # keep track of action by agent
        self.action = None

        # clean buffer
        self.buffer.reset()



#####
## Advantage Actor-Critic algorithm
#


class AACAgent(ActorCriticAgent):
    """
    Advantage Actor-Critic
    """

    def __init__(self, model, optimizer=None, gpu=-1, cutoff=None, gamma=0.99, beta=1e-2):

        # cutoff for truncated BPTT
        self.cutoff = cutoff

        # discounting factor
        self.gamma = gamma

        # contribution of entropy term
        self.beta = beta

        # monitor score, entropy and reward
        self.buffer = Monitor()

        super(AACAgent, self).__init__(model, optimizer=optimizer, gpu=gpu)

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
        self.action, policy, value = self.model(map(lambda x: Variable(self.xp.asarray(x)), data), train=train)

        # backpropagate if we reach the cutoff for truncated backprop or if we processed the last batch
        if train and idx > 0 and ((self.cutoff and (idx % self.cutoff) == 0) or final):

            # return value associated with last state
            _return = value

            pi_loss = Variable(self.xp.zeros((), 'float32'))
            v_loss = Variable(self.xp.zeros((), 'float32'))
            for i in range(len(self.buffer.dict['reward'])-1,-1,-1):

                _return = self.buffer.dict['reward'].pop() + self.gamma * _return

                advantage = _return - self.buffer.dict['value'].pop()

                _ss = F.squeeze(self.buffer.dict['score'].pop()) * advantage.data
                if _ss.size > 1:
                    _ss = F.sum(_ss, axis=0)

                pi_loss -= F.squeeze(_ss)

                pi_loss -= self.beta * self.buffer.dict['entropy'].pop()

                v_loss += F.sum(advantage ** 2)

            v_loss = F.reshape(v_loss, pi_loss.data.shape)

            # Compute total loss; 0.5 supposedly used by Mnih et al
            loss = pi_loss + 0.5 * v_loss

            _loss = loss.data

            self.optimizer.zero_grads()
            loss.backward()
            loss.unchain_backward()
            self.optimizer.update()

            # store return
            if self.monitor:
                map(lambda x: x.set('return', np.mean(_return.data)), self.monitor)

        else:

            _loss = 0

        # recompute score function: grad_theta log pi_theta (s_t, a_t) * v_t
        self.buffer.set('score', self.score_function(self.action, policy))

        # compute entropy
        self.buffer.set('entropy', F.sum(self.entropy(policy)))

        # add value
        self.buffer.set('value', value)

        return _loss


    def reset(self):

        self.model.predictor.reset_state()

        # keep track of action by agent
        self.action = None

        # clean buffer
        self.buffer.reset()
