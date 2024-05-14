import numpy as np

from ..projections.Projection import Projection
from ..projections.SparseProjection import SparseProjection

class RLS(object):
    """
    Recursive least-squares (RLS) learning rule for FORCE learning.

    Parameters:
        projection: projection on which to apply the learning rule.
        delta: initial diagonal value of the correlation matrix.
    """

    def __init__(self, projection:Projection, delta:float=1e-6) -> None:

        self.projection = projection
        self.delta = delta

        self.nb_post = self.projection.post.size

        self.sparse = isinstance(self.projection, (SparseProjection, ))

        self._has_bias = self.projection._has_bias

        self.P = [
            np.identity(
                self.projection.nb_connections(i) + (1 if self._has_bias else 0)
            ) / self.delta 
            for i in range(self.nb_post)
        ]
        

    def train(self, error:np.ndarray) -> None:
        """
        Applies one step of the RLS learning rule.

        Parameters:
            error: error vector at the post-synaptic level.
        """

        # Apply the FORCE learning rule to the readout weights
        dW = []
        db = []

        for i in range(self.nb_post):

            r = self.projection.input(i).reshape((-1, ))
            
            if self._has_bias:
                r = np.concatenate((r, np.ones((1, ))))

            # Multiply the rates with the inverse correlation matrix P*R
            PxR =  self.P[i] @ r

            # Normalization term 1 + R'*P*R
            RxPxR =  (r.T @  PxR + 1.)
            
            # Update the inverse correlation matrix P <- P - ((P*R)*(P*R)')/(1+R'*P*R)
            self.P[i] -= np.outer(PxR, PxR)/RxPxR
            
            # Learning rule W <- W + e * (P*R)/(1+R'*P*R)
            diff = error[i] * (PxR/RxPxR).reshape((-1,))

            if self._has_bias:
                dW.append(diff[:-1])
                db.append(diff[-1])
            else:
                dW.append(diff)

        self.projection.W += np.array(dW)
        if self._has_bias:
            self.projection.bias += np.array(db)

