import copy
import functools


class Curridata(object):

    def __init__(self, nfeatures, datagenerator, genparams, funclist, dependencies, funcparams={}, batchsize=1, seed=0):

        self.genparams = genparams
        self.batchsize = batchsize
        self.seed = seed
        self.nfeatures = nfeatures
        self.funclist = funclist
        self.dependencies = dependencies
        self.funcparams = funcparams

        self.__data = None
        self.__features = [None for _ in range(self.nfeatures + 1)]

        self.gen = datagenerator(**genparams)
        self.genit = self.gen.iterator(batchsize, seed)

    def _iterate(self):
        for i in self.genit:
            return i

    def next(self):
        self.__data = self._iterate()
        self.__data.update(self.funcparams)
        self.__features = [None for _ in range(self.nfeatures + 1)]

    def changegenparam(self, genparams):
        for t, u in genparams.iteritems():
            setattr(self.gen, t, u)

    # definition of the get property functions
    def getter(self, i):
        if self.__data is None:
            self.next()
        if self.__features[i] is not None:
            return self.__features[i]
        else:
            if self.dependencies[i] is not None:
                tmp = copy.copy(self.__data)
                for t, u in self.dependencies[i].iteritems():
                    self.getter(u)
                    tmp.update({t: self.__features[u]})
                self.__features[i] = self.funclist[i](**tmp)
            else:
                self.__features[i] = self.funclist[i](**self.__data)
        return self.__features[i]

    # here you need to hard code the max features number and the targets and inputs property field
    image = property(functools.partial(getter, i=0))
    edges = property(functools.partial(getter, i=1))
    depth = property(functools.partial(getter, i=2))
    identity = property(functools.partial(getter, i=3))
    segmentation = property(functools.partial(getter, i=4))
    output = property(functools.partial(getter, i=5))
    edgesc = property(functools.partial(getter, i=6))
