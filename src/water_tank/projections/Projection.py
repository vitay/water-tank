from ..layers import Layer
from ..random import RandomDistribution, Const

class Projection(object):
    pass

    
def connect(
        pre:Layer, 
        post:Layer, 
        weights:float | RandomDistribution, 
        bias:float | RandomDistribution=None, 
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
        from .DenseProjection import DenseProjection
        proj = DenseProjection(
            pre=pre, 
            post=post, 
            weights=weights, 
            bias=bias
        )
    else:
        from .SparseProjection import SparseProjection
        proj = SparseProjection(
            pre=pre, 
            post=post, 
            weights=weights, 
            bias=bias, 
            sparseness=sparseness
        )

    post.projections.append(proj)

    return proj
