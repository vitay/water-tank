import numpy as np
import numba as nb

from .RandomDistributions import RandomDistribution

class DenseProjection(object):

    def __init__(self, pre, post, weights, bias=None):

        self.pre = pre
        self.post = post

        # Weights
        if isinstance(weights, RandomDistribution):
            self.W = weights.sample((self.post.size, self.pre.size))
        elif isinstance(weights, (float, int)):
            self.W = float(weights) * np.ones((self.post.size, self.pre.size))
        else: # TODO: Check dimensions
            self.W = weights

        # Bias
        self._has_bias = True
        if bias is None:
            self._has_bias = False
            self.bias = 0.0
        elif isinstance(bias, (float, int)):
            self.bias = float(bias) * np.ones(self.post.size)
        elif isinstance(bias, RandomDistribution):
            self.bias = bias.sample((self.post.size,))
        else: # bias=True works
            self.bias = np.zeros(self.post.size)

    def step(self):
        return self.W @ self.pre.output() + self.bias
    
    def nb_connections(self, i):
        return self.pre.size


class SparseProjection(object):

    def __init__(self, pre, post, weights, bias, sparseness):

        self.pre = pre
        self.post = post
        self.weights_initializer = weights
        self.sparseness = sparseness

        self.nb_post = post.size
        self.nb_pre = pre.size

        # Weight matrix
        if not isinstance(weights, RandomDistribution):
            print("Sparse projections can only accept random distributions for the weights argument.")
            raise ZeroDivisionError

        rng = np.random.default_rng()
        self.connectivity = []
        self.nb_weights = []
        self.W = []

        for i in range(self.nb_post):
            nb = np.random.binomial(self.nb_pre, self.sparseness, 1)[0]
            self.nb_weights.append(nb)
            self.connectivity.append(
                sorted(
                    rng.choice(
                        pre.size, 
                        size=nb, 
                        replace=False, 
                    )
                ) 
            )
            self.W.append(self.weights_initializer.sample(nb))

        # Bias
        self._has_bias = True
        if bias is None:
            self._has_bias = False
            self.bias = 0.0
        elif isinstance(bias, (float, int)):
            self.bias = float(bias) * np.ones(self.post.size)
        elif isinstance(bias, RandomDistribution):
            self.bias = bias.sample((self.post.size,))
        else: # bias=True works
            self.bias = np.zeros(self.post.size)

    def nb_connections(self, i):
        return self.nb_weights[i]

    def step(self):
        return _spmv(self.nb_post, self.W, self.bias, self.connectivity, self.pre.output())
    

#@nb.jit(nopython=True)
def _spmv(nb_post, W, bias, connectivity, r):
    if isinstance(bias, (float,)):
        res = np.array([bias for i in range(nb_post)])
    else:
        res = bias.copy()

    for i in range(nb_post):
        r_sub = r[connectivity[i]]
        tmp = 0.0
        for j in range(len(r_sub)):
            tmp += r_sub[j] * W[i][j]
        res[i] = tmp

    return res