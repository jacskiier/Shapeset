from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six

import copy


class Curridata(object):

    def __init__(self, datagenerator, genparams, funclist, dependencies, funcparams={}, batchsize=1, seed=0,
                 generatorReturnsBatch=False, feature_input=0, feature_output=1):
        """A generator that wraps another generator and processes that data into formats based on given functions

        :param datagenerator: the class to use as the data generator
        :param genparams: the parameters to give the data generator class on initialization
        :param funclist: the list of feature functions
        :param dependencies: a list of dictionaries that corresponds to funclist.
            Each dictionary has as a key the parameter name that will be given to the func in funclist.
            The value is the index of the func in funclist which that parameter will be set to.
            Allows the arguments of a func in funclist to be the output of some other func in funclist
        :param funcparams: a dictionary of parameters to give all functions in the funclist or a list of dicts to give each function
        :param batchsize: the size of how many samples to generate at once
        :param seed: the random seed to use when generating samples
        :param generatorReturnsBatch: if True when generating new batches it will return a tuple of (input features, outputs)
        :param feature_input: the index of which function in funclist to use as the input when returning a batch tuple
        :param feature_output: the index of which function in funclist to use as the output when returning a batch tuple
        """

        self.genparams = genparams
        self.batchsize = batchsize
        self.seed = seed
        self.nfeatures = len(funclist)
        if len(funclist) != len(dependencies):
            raise ValueError("funclist and dependencies must have same length")
        self.funclist = funclist
        self.dependencies = dependencies
        if isinstance(funcparams, dict):
            self.funcparams = [funcparams] * self.nfeatures
        elif isinstance(funcparams, list) or isinstance(funcparams, tuple):
            if len(funcparams) == 1:
                self.funcparams = funcparams * self.nfeatures
            elif len(funcparams) == self.nfeatures:
                self.funcparams = funcparams
            else:
                raise ValueError("funcparams length is {0} but must have same length as funclist which is {1}".format(len(funcparams), self.nfeatures))
        else:
            raise ValueError("funcparams must be a dict, list or tuple")

        self.__data = None
        self.__features = [None for _ in range(self.nfeatures)]

        self.gen = datagenerator(**self.genparams)
        self.genit = self.gen.iterator(batchsize, seed)

        self.generatorReturnsBatch = generatorReturnsBatch
        self.feature_input = feature_input
        self.feature_output = feature_output

    def _iterate(self):
        for i in self.genit:
            return i

    def send(self, sendArg):
        self.__data = self._iterate()
        self.__features = [None for _ in range(self.nfeatures)]
        if self.generatorReturnsBatch:
            return self.getter(self.feature_input), self.getter(self.feature_output)
        else:
            return

    def throw(self, type=None, value=None, traceback=None):
        raise StopIteration

    def __iter__(self):
        return self

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def next(self):
        return self.send(None)

    def close(self):
        """Raise GeneratorExit inside generator.
        """
        try:
            self.throw(GeneratorExit)
        except (GeneratorExit, StopIteration):
            # cleanup before generator exits
            pass
        else:
            raise RuntimeError("generator ignored GeneratorExit")

    def changegenparam(self, genparams):
        for t, u in six.iteritems(genparams):
            setattr(self.gen, t, u)

    # definition of the get property functions
    def getter(self, i):
        if i >= len(self.__features):
            raise ValueError("i is too large for the available list of features from funclist and dependencies")
        if self.__data is None:
            self.next()
        if self.__features[i] is not None:
            return self.__features[i]
        else:
            data_tmp = copy.copy(self.__data)
            data_tmp.update(self.funcparams[i])
            if self.dependencies[i] is not None:
                for t, u in six.iteritems(self.dependencies[i]):
                    self.getter(u)
                    data_tmp.update({t: self.__features[u]})
                self.__features[i] = self.funclist[i](**data_tmp)
            else:
                self.__features[i] = self.funclist[i](**data_tmp)
        return self.__features[i]
