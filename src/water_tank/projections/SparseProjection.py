import numpy as np
import scipy.sparse as sp

from .Projection import Projection

from ..layers import Layer
from ..random import RandomDistribution, Const
from .sparse import LIL

class SparseProjection(Projection):
    """
    Sparse weight matrix.  Created and returned by `connect()`.

    Parameters:
        pre: input layer.
        post: output layer.
        weights: float or `RandomDistribution` to create the weight matrix.
        bias: bias per post neuron. If `None` or `False`, no bias is used. Otherwise, can be a float or `RandomDistribution`.
        sparseness: density of the weight matrix.
    """

    def __init__(self,  
                 pre:Layer, post:Layer, 
                 weights: float | RandomDistribution, 
                 bias: float | RandomDistribution=None, 
                 sparseness:float=0.1) -> None:

        self.pre = pre
        self.post = post
        self.sparseness = sparseness

        self.nb_post = post.size
        self.nb_pre = pre.size

        # Weight matrix
        if isinstance(weights, (float, int)):
            weights = Const(float(weights))
        if isinstance(weights, (RandomDistribution,)):
            self.W = LIL(self.post.size, self.pre.size)
            self.W.fill_random(proba=self.sparseness, weights=weights, self_connections=(not pre==post))
        elif isinstance(weights, (sp.sparray,)):
            self.W = LIL.from_scipy(weights)

        # Bias
        self._has_bias = True
        if bias is None or False:
            self._has_bias = False
            self.bias = 0.0
        elif isinstance(bias, (float, int)):
            self.bias = float(bias) * np.ones(self.post.size)
        elif isinstance(bias, RandomDistribution):
            self.bias = bias.sample((self.post.size,))
        elif isinstance(bias, np.ndarray):
            self.bias = bias
        else: # bias=True works
            self.bias = np.zeros(self.post.size)


    def nb_connections(self, idx)  -> int:
        """
        Returns:
            the number of weights received by the neuron of index `idx`.
        """
        return len(self.W.ranks[idx])
    
    def input(self, idx) -> np.ndarray:
        """
        Returns:
            the vector of inputs received by the neuron of index `idx`.
        """
        return self.pre.output()[self.W.ranks[idx]]

    def step(self) -> None:
        "Performs a weighted sum of inputs plus bias."

        return self.W @ self.pre.output() + self.bias
    

    def _empty_copy(self) -> sp.sparray:
        """
        A matrix of the same shape with the same connections, but the values are filled with zeros.
        """
        return self.W.uniform_copy(0.0)
    

    def save(self) -> dict:
        """
        Returns a dictionary of learnable parameters.
        """
        return {
            'type': 'sparse',
            'W': self.W, # TODO
            'bias': self.bias if self._has_bias else None,
        }

    def load(self, data: dict) -> None:
        """
        Loads a dictionary of learnable parameters.

        TODO
        """

        # Load weight matrix
        self.W = sp.csr_matrix(data['W']) # TODO

        # Load bias
        self._has_bias = False if data['bias'] is None or False else True

        if not self._has_bias:
            self.bias = 0.0
        else:
            self.bias = np.array(data['bias'])