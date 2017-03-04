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

class REINFORCEAgent(ActorCriticAgent):
    """
    Note that REINFORCE is a policy gradient method which does not use a critic.
    Instead the return is computed as a running estimate

    https://webdocs.cs.ualberta.ca/%7Esutton/book/bookdraft2016sep.pdf
    https://github.com/dennybritz/reinforcement-learning/tree/master/PolicyGradient
    http://blog.shakirm.com/2015/11/machine-learning-trick-of-the-day-5-log-derivative-trick/
    http://www.1-4-5.net/~dmm/ml/log_derivative_trick.pdf
    """

    def __init__(self, model, optimizer=None, gpu=-1, cutoff=None, gamma=0.99):

        super(REINFORCEAgent, self).__init__(model, optimizer=optimizer, gpu=gpu)

        # cutoff for truncated BPTT
        self.cutoff = cutoff

        # discounting factor
        self.gamma = gamma

        # reset state
        self.reset()

    def run(self, data, train=True, idx=None, final=False):

        # get reward associated with taking the previous action in the previous state
        reward = data[-1]

        # determine if we already completed a perception-action cycle
        if not self.score is None:

            # update return using reward
            if self._return is None:
                self._return = self.gamma * Variable(reward)
            else:
                self._return += self.gamma * Variable(reward)

            # add minus the score function times the value based on (s_t-1, a_t-1, r_t-1) to loss
            self.loss -= F.squeeze(F.sum(F.batch_matmul(self.score, self._return, transa=True), axis=0))

        # compute policy and take new action based on observations
        self.action, policy, _ = self.model(map(lambda x: Variable(self.xp.asarray(x)), data), train=train)

        # recompute score function: grad_theta log pi_theta (s_t, a_t) * v_t
        self.score = self.score_function(self.action, policy)

        # backpropagate if we reach the cutoff for truncated backprop or if we processed the last batch
        if train and ((self.cutoff and (idx % self.cutoff) == 0) or final):

            self.optimizer.zero_grads()
            self.loss.backward()
            self.loss.unchain_backward()
            self.optimizer.update()

            self.loss = Variable(self.xp.zeros((), 'float32'))

            # store return
            if self.monitor:
                self.monitor.set('return', np.mean(self._return.data))

            self._return = None

        # reset if we reached the end of an episode
        if final:

            self.action = None
            self.score = None
            self._return = None

        return self.loss.data

    def reset(self):

        self.model.predictor.reset_state()

        # keep track of loss for BPTT
        self.loss = Variable(self.xp.zeros((), 'float32'))

        # keep track of action by agent
        self.action = None

        ## keep track of history

        self.score = None
        self._return = None




# class REINFORCEAgent(ActorCriticAgent):
#
#     def __init__(self, model, optimizer=None, gpu=-1, cutoff=None):
#         super(REINFORCEAgent, self).__init__(model, optimizer=optimizer, gpu=gpu)
#
#         # cutoff for truncated BPTT
#         self.cutoff = cutoff
#
#         # keep track of current index for BPTT
#         self.idx = 0
#
#         # keep track of loss for BPTT
#         self.loss = Variable(self.xp.zeros((), 'float32'))
#
#     def run(self, data, train=True, idx=None, final=False):
#
#         # separate observation from reward
#         obs = data[0] if len(data) == 2 else data[:-1]  # inputs
#
#         # reward associated with taking the previous action
#         reward = data[-1]
#
#         # provide observation to RL model. Returns action, policy and value
#         action, policy, value = self.model(map(lambda x: Variable(self.xp.asarray(x)), obs), train=train)
#
#         self.action = action
#
#         score = self.score_function(action, policy)
#
#         self.loss += F.squeeze(F.batch_matmul(score, value, transa=True))
#
#         # backpropagate if we reach the cutoff for truncated backprop or if we processed the last batch
#         if train and ((self.cutoff and (idx % self.cutoff) == 0) or final):
#             self.optimizer.zero_grads()
#             self.loss.backward()
#             self.loss.unchain_backward()
#             self.optimizer.update()
#
#             self.loss = Variable(self.xp.zeros((), 'float32'))
#
#         return float(score.data[0])


                        # class StatelessAgent(Agent):
#
#     def __init__(self, model, optimizer=None, gpu=-1):
#         """
#
#         :param model:
#         :param optimizer:
#         :param gpu:
#         """
#
#         super(StatelessAgent, self).__init__(model, optimizer, gpu)
#
#         # add monitor needed for updating the RL model
#         self.monitor = Monitor(names=['reward', 'value', 'score_function', 'entropy'])
#
#
#     def run(self, data, train=True, idx=None, final=False):
#
#
#         # generate action using RL model - , pi, v
#         action = self.model(map(lambda x: Variable(self.xp.asarray(x)), data), train=train)
#
#         # # store log policy data
#         # _score_function = self.score_function(action, pi)
#         #
#         # # compute entropy
#         # _entropy = self.entropy(pi)
#         #
#         # # get feedback from environment
#         # obs, reward, terminal, target = task.step(action)
#         #
#         # # add to results
#         # result.add('score_function', _score_function.data, 'entropy', _entropy.data,
#         #            'value', v.data, 'action', action, 'policy', pi.data, 'reward', reward, 'terminal', terminal)
#         # for key in internal.keys():
#         #     result.add(key, internal[key])
#         #
#         # # add to past buffer
#         # past.add('score_function', _score_function, 'entropy', _entropy, 'value', v, 'reward', reward)
#         #
#         # result.increment()
#         # past.increment()






        # loss = self.model(map(lambda x: Variable(self.xp.asarray(x)), data), train=train)
        #
        # # normalize by number of datapoints in minibatch
        # _loss = float(loss.data/len(data))
        #
        # if train:
        #     self.optimizer.zero_grads()
        #     loss.backward()
        #     self.optimizer.update()
        #
        # return _loss

# class StatefulAgent(Agent):
#
#     def __init__(self, model, optimizer=None, gpu=-1, cutoff=None, last=False):
#
#         super(StatefulAgent, self).__init__(model, optimizer=optimizer, gpu=gpu)
#
#         # cutoff for BPTT
#         self.cutoff = cutoff
#
#         # whether to update from loss in last step only
#         self.last = last
#
#         # required for BPTT
#
#         # keep track of current index for
#         self.idx = 0
#
#         # keep track of loss for truncated BP
#         self.loss = Variable(self.xp.zeros((), 'float32'))
#
#     def run(self, data, train=True, idx=None, final=False):
#
#         loss = self.model(map(lambda x: Variable(self.xp.asarray(x)), data), train=True)
#
#         if self.last:  # used in case we propagate back at end of trials only
#             self.loss = loss
#         else:
#             self.loss += loss
#
#         _loss = float(loss.data / len(data))
#
#         # backpropagate if we reach the cutoff for truncated backprop or if we processed the last batch
#         if train and ((self.cutoff and (idx % self.cutoff) == 0) or final):
#
#             self.optimizer.zero_grads()
#             self.loss.backward()
#             self.loss.unchain_backward()
#             self.optimizer.update()
#
#             self.loss = Variable(self.xp.zeros((), 'float32'))
#
#         return _loss