from collections import defaultdict
import numpy as np

class Monitor(object):

    def __init__(self, names=None, function=None):
        """

        :param names: if defined then the name or list of names indicates keys to store
        :param function: optional function that is applied after each processing cycle
        """

        self._dict = defaultdict(list)

        self._names = [names] if names is str else names

        self.function = function


    def append(self, name, value):
        """

        :param name: dictionary key
        :param value: dictionary value
        :return:
        """

        if self._names is None or name in self._names:
            self._dict[name].append(value)

    def keys(self):
        return self._dict.keys()

    def get(self, name):
        """Returns dict[name] as numpy array

        :param name: list of key names
        :return: numpy array
        """

        assert(name in self.keys())
        data = np.asarray(self._dict[name])
        if data.ndim <=2:
            return data.reshape([np.prod(data.shape)], order='F')
        else:
            return data.reshape([np.prod(data.shape[:2])] + list(data.shape[2:]), order='F')

    def run(self):
        if not self.function is None:
            self.function()
