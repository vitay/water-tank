import numpy as np

from .Projections import DenseProjection, SparseProjection
from .RandomDistributions import RandomDistribution

class LinearReadout(object):

    def __init__(self, size):
        self.size = size
        self.r = np.zeros((self.size,))

    def output(self):
        return self.r

    def step(self, force=None):

        self.x = self.projection.step()

        if force is None:
            self.r = self.x
        else:
            self.r = force

    def connect(self, population, weights, bias=None, sparseness=1.0):

        if sparseness == 1.0:
            self.projection = DenseProjection(pre=population, post=self, weights=weights, bias=bias)
        else:
            self.projection = SparseProjection(pre=population, post=self, weights=weights, bias=bias, sparseness=sparseness)

        return self.projection