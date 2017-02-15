from collections import defaultdict
import numpy as np

class Monitor(object):

    def __init__(self, names=None):
        """

        :param names: if defined then the name or list of names indicates keys to store
        """

        self._dict = defaultdict(list)

        self._names = [names] if names is str else names


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

    def get(self, names):
        """Returns each value for key in keys list as numpy array

        :param names: list of key names
        :return: list of numpy arrays
        """

        if type(names) is str:
            assert(names in self.keys())
            data = np.asarray(self._dict[names])
            return data if data.ndim <= 1 else data.squeeze(axis=1)
        else:
            assert(np.all(map(lambda x: x in self.keys(), names)))
            data = [np.asarray(self._dict[k]) for k in names]
            return map(lambda x: x if x.ndim <= 1 else x.squeeze(axis=1), data)