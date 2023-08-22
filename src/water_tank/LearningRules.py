import numpy as np
#import numba as nb 

from .Projections import SparseProjection

class DeltaLearningRule(object):

    def __init__(self, projection, target, learning_rate):

        self.projection = projection
        self.target = target
        self.learning_rate = learning_rate

    def step(self):

        self.projection.W +=  self.learning_rate * np.outer(self.target.output() - self.projection.post.output(), self.projection.pre.output())
        

class RLS(object):

    def __init__(self, projection, target, delta):

        self.projection = projection
        self.target = target
        self.delta = delta

        self.nb_post = self.projection.post.size

        self.sparse = isinstance(self.projection, (SparseProjection))

        self._has_bias = self.projection._has_bias

        self.P = [
            np.identity(
                self.projection.nb_connections(i) + (1 if self._has_bias else 0)
            )/self.delta \
                for i in range(self.nb_post)
            ]

    def step(self):

        # Compute the error of the output neurons
        error =  self.target.output() - self.projection.post.output()
        
        r = self.projection.pre.output().reshape((-1, 1))

        # Apply the FORCE learning rule to the readout weights
        for i in range(self.nb_post): # for each readout neuron

            if self.sparse:
                r_local = r[self.projection.connectivity[i]]
            else:
                r_local = r

            if self._has_bias:
                r_local = np.concatenate((r_local, np.ones((1, 1))))

            # Multiply the rates with the inverse correlation matrix P*R
            PxR =  self.P[i] @ r_local

            # Normalization term 1 + R'*P*R
            RxPxR =  (1. + r_local.T @  PxR)[0]
            
            # Update the inverse correlation matrix P <- P - ((P*R)*(P*R)')/(1+R'*P*R)
            self.P[i] -= np.outer(PxR, PxR)/RxPxR
            
            # Learning rule W <- W + e * (P*R)/(1+R'*P*R)
            dW = (error[i] * (PxR/RxPxR))[:, 0]

            if self._has_bias:
                self.projection.W[i] += dW[:-1]
                self.projection.bias[i] += dW[-1]
            else:
                self.projection.W[i] += dW



