import numpy as np
import scipy.sparse as sp

from .Projection import Projection
from ..layers import Layer
from ..random import RandomDistribution, Const


class DenseProjection(Projection):
    """
    Dense weight matrix. Created and returned by `connect()`.

    Parameters:
        pre: input layer.
        post: output layer.
        weights: float or `RandomDistribution` to create the weight matrix.
        bias: bias per post neuron. If `None` or `False`, no bias is used. Otherwise, can be a float or `RandomDistribution`.
    """

    def __init__(self, 
                 pre:Layer, 
                 post:Layer, 
                 weights: float | RandomDistribution, 
                 bias: float | RandomDistribution=None) -> None:

        self.pre = pre
        self.post = post

        # Weights
        if isinstance(weights, RandomDistribution):
            self.W = weights.sample((self.post.size, self.pre.size))
        elif isinstance(weights, (float, int)):
            self.W = float(weights) * np.ones((self.post.size, self.pre.size))
        elif isinstance(weights, np.ndarray):
            self.W = weights # TODO: Check dimensions
        else: 
            self.W = weights

        # No self-connection
        if pre == post:
            np.fill_diagonal(self.W, 0.0)

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

    def step(self):
        "Performs a weighted sum of inputs plus bias."
        return self.W @ self.pre.output() + self.bias
    
    def nb_connections(self, idx) -> int:
        """
        Returns:
            the number of weights received by the neuron of index `idx`.
        """
        return self.pre.size

    def input(self, idx)  -> np.ndarray:
        """
        Returns:
            the vector of inputs received by the neuron of index `idx`.
        """
        return self.pre.output()
    
    def _empty_copy(self):
        """
        A matrix of the same shape, filled with zeros.
        """
        return np.zeros(self.W.shape)
    
    def save(self):
        """
        Returns a dictionary of learnable parameters.
        """
        return {
            'type': 'dense',
            'W': self.W,
            'bias': self.bias if self._has_bias else None,
        }

    def load(self, data: dict) -> None:
        """
        Loads a dictionary of learnable parameters.
        """
        # Load weight matrix
        self.W = np.array(data['W'])

        # Load bias
        self._has_bias = False if data['bias'] is None or False else True

        if not self._has_bias:
            self.bias = 0.0
        else:
            self.bias = np.array(data['bias'])
