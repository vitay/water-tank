import numpy as np

from .Projections import DenseProjection, SparseProjection
from .RandomDistributions import RandomDistribution

class TanhReservoir(object):

    def __init__(self, size, g=1.0, tau=10.0, W=None, std_W=None, sparseness=1.0):
        
        self.size = size
        self.g = g
        self.tau = tau
        self.std_W = std_W
        self.sparseness = sparseness

        # Vectors
        self.x = np.zeros((self.size,))
        self.r = np.zeros((self.size,))

        # Weight matrix
        if W is None:
            if self.std_W is None:
                self.std_W = self.g / np.sqrt(self.size)

            rng = np.random.default_rng()
            self.W = rng.normal(0.0, self.std_W, (self.size, self.size))
            np.fill_diagonal(self.W, 0.0)
        
        elif isinstance(W, RandomDistribution):
            self.W = W.sample((self.size, self.size))
            np.fill_diagonal(self.W, 0.0)
        
        else: # numpy array
            self.W = W

        # Projections
        self.projections = []

    def output(self):
        return self.r

    def step(self):

        inputs = self._collect_inputs()
        self.x += (inputs + self.W @ self.r - self.x)/self.tau
        self.r = np.tanh(self.x)
    
    def connect(self, population, weights, sparseness=1.0):

        if sparseness == 1.0:
            proj = DenseProjection(pre=population, post=self, weights=weights)
        else:
            proj = SparseProjection(pre=population, post=self, weights=weights, sparseness=sparseness)

        self.projections.append(proj)

        return proj

    def _collect_inputs(self):

        inp = 0.0
        for proj in self.projections:
            inp += proj.step()

        return inp
    