import numpy as np

# Allowed transfer functions
transfer_functions = {
    'tanh': np.tanh
}


class Layer(object):

    def output(self)  -> None:
        """
        Returns:
            a vector of activities.
        """
        raise NotImplementedError
    

    def _collect_inputs(self):
        """
        Calls each projection of `self.projections` and accumulates the result of their `step()` method in a numpy array of the same size as the layer.
        """

        inp = np.zeros(self.size)
        for proj in self.projections:
            inp += proj.step().reshape((self.size,))

        return inp