import numpy as np
import scipy.sparse as sp
from typing import Union

from .Layers import Layer
from .RandomDistributions import RandomDistribution, Const

class Projection(object):
    pass


    
def connect(
        pre:Layer, 
        post:Layer, 
        weights:Union[float, RandomDistribution], 
        bias:Union[float, RandomDistribution]=None, 
        sparseness:float=1.0) -> Projection:
    """
    Connects two layers with a (sparse) weight matrix and optionally a bias vector. 

    Parameters:
        pre: input layer.
        post: output layer.
        weights: float or `RandomDistribution` to create the weight matrix.
        bias: bias per post neuron. If `None` or `False`, no bias is used. Otherwise, can be a float or `RandomDistribution`.
        sparseness: density of the weight matrix.

    Returns:
        a `DenseProjection` or `SparseProjection` instance.
    """

    if sparseness == 1.0:
        proj = DenseProjection(
            pre=pre, 
            post=post, 
            weights=weights, 
            bias=bias
        )
    else:
        proj = SparseProjection(
            pre=pre, 
            post=post, 
            weights=weights, 
            bias=bias, 
            sparseness=sparseness
        )

    post.projections.append(proj)

    return proj



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
                 pre:Layer, post:Layer, 
                 weights:Union[float, RandomDistribution], 
                 bias:Union[float, RandomDistribution]=None) -> None:

        self.pre = pre
        self.post = post

        # Weights
        if isinstance(weights, RandomDistribution):
            self.W = weights.sample((self.post.size, self.pre.size))
        elif isinstance(weights, (float, int)):
            self.W = float(weights) * np.ones((self.post.size, self.pre.size))
        else: # TODO: Check dimensions
            self.W = weights

        # No self-connection
        if pre == post:
            np.fill_diagonal(self.W.diagonal, 0.0)

        # Bias
        self._has_bias = True
        if bias is None or False:
            self._has_bias = False
            self.bias = 0.0
        elif isinstance(bias, (float, int)):
            self.bias = float(bias) * np.ones(self.post.size)
        elif isinstance(bias, RandomDistribution):
            self.bias = bias.sample((self.post.size,))
        else: # bias=True works
            self.bias = np.zeros(self.post.size)

    def step(self):
        "Performs a weighted sum of inputs plus bias."
        return self.W @ self.pre.output() + self.bias
    
    def _update_parameters(self, weights, bias=None):

        self.W += weights
        if self._has_bias and bias is not None:
            self.bias += bias
    
    def _set_parameters(self, weights, bias=None):

        self.W = weights
        if self._has_bias and bias is not None:
            self.bias = bias
    
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
                 weights:Union[float, RandomDistribution], 
                 bias:Union[float, RandomDistribution]=None, 
                 sparseness:float=0.1) -> None:

        self.pre = pre
        self.post = post
        self.weights_initializer = weights
        self.bias_initializer = bias
        self.sparseness = sparseness

        self.nb_post = post.size
        self.nb_pre = pre.size

        # Weight matrix
        if isinstance(self.weights_initializer, (float, int)):
            self.weights_initializer = Const(float(self.weights_initializer))
        self.W = sp.random(self.post.size, self.pre.size, density=self.sparseness, format='csr', data_rvs=self.weights_initializer.sample)
        #self.W.setdiag(0.0)
        
        self.nb_weights = np.diff(self.W.indptr)
        self.connectivity = np.split(self.W.indices, self.W.indptr)[1:-1]

        # Bias
        self._has_bias = True
        if self.bias_initializer is None or False:
            self._has_bias = False
            self.bias = 0.0
        elif isinstance(self.bias_initializer, (float, int)):
            self.bias = float(self.bias_initializer) * np.ones(self.post.size)
        elif isinstance(self.bias_initializer, RandomDistribution):
            self.bias = self.bias_initializer.sample((self.post.size,))
        else: # bias=True works
            self.bias = np.zeros(self.post.size)
            
    def nb_connections(self, idx)  -> int:
        """
        Returns:
            the number of weights received by the neuron of index `idx`.
        """
        return self.nb_weights[idx]
    
    def input(self, idx) -> np.ndarray:
        """
        Returns:
            the vector of inputs received by the neuron of index `idx`.
        """
        return self.pre.output()[self.connectivity[idx]]

    def step(self) -> None:
        "Performs a weighted sum of inputs plus bias."
        return self.W @ self.pre.output() + self.bias
    
    
    def _update_parameters(self, weights:np.ndarray, bias:np.ndarray=None):
        
        weights = [val for row in weights for val in row]
        self.W += sp.csr_matrix((weights, self.W.indices, self.W.indptr), shape=self.W.shape) 
        if self._has_bias and bias is not None:
            self.bias += bias
    
    def _set_parameters(self, weights:np.ndarray, bias:np.ndarray=None):

        weights = [val for row in weights for val in row]
        self.W = sp.csr_matrix((weights, self.W.indices, self.W.indptr), shape=self.W.shape) 
        if self._has_bias and bias is not None:
            self.bias = bias

