import numpy as np

from ..projections import Projection, SparseProjection

class DeltaLearningRule(object):
    r"""
    Delta learning rule (online linear regression).

    Equation:

    $$\Delta W = \eta \, (\mathbf{t} - \mathbf{y}) \times \mathbf{x}^T$$

    Parameters:
        projection: projection on which to apply the learning rule.
        learning_rate: learning rate.
    """

    def __init__(self, projection:Projection, learning_rate:float):

        self.projection = projection
        self.learning_rate = learning_rate

        self.nb_post = self.projection.post.size

        self.sparse = isinstance(self.projection, (SparseProjection))
        self._has_bias = self.projection._has_bias

    def train(self, target):

        error = target - self.projection.post.output()

        if not self.sparse:

            self.projection.W += self.learning_rate * np.outer(error, self.projection.pre.output()),
            
            if self._has_bias:
                self.projection.bias += self.learning_rate * error 

        else:
            self.projection.W += self.learning_rate * self.projection.W.outer(error, self.projection.pre.output())

            if self._has_bias:
                self.projection.bias += self.learning_rate * error 


