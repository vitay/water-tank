# distutils: language = c++
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

from libcpp.vector cimport vector
from libcpp cimport bool
from cython.operator cimport dereference

import numpy as np
cimport numpy as np

import water_tank as wt

cdef extern from "LIL.hpp" :
    cdef cppclass LILMatrix:
        LILMatrix(int, int) except +

        # Attributes
        const int nb_post;
        const int nb_pre;

        vector [ vector[int] ] ranks;
        vector [ vector[float] ] values;

        # Size
        long get_size();

        # Construction
        void fill_row(int, vector[int], vector[float])

        # Operations returning a new matrix
        LILMatrix* uniform_copy(float)
        LILMatrix* add_scalar_copy(float)
        LILMatrix* multiply_scalar_copy(float)
        vector[float] multiply_vector_copy(vector[float])

        LILMatrix* outer_product(vector[float], vector[float])

        # In-place operations
        void add_scalar_inplace(float)
        void multiply_scalar_inplace(float)
        void add_matrix_inplace(LILMatrix*)
        void substract_matrix_inplace(LILMatrix*)

cdef class ConnectionMatrix:
    pass

cdef class LIL(ConnectionMatrix):
    cdef LILMatrix *matrix
    cdef int nb_post, nb_pre
    cdef bool _instantiated

    def __cinit__(self, int nb_post, int nb_pre):
        self.nb_post = nb_post
        self.nb_pre = nb_pre
        self._instantiated = False

    def __dealloc__(self):
        del self.matrix

    def _instantiate(self):
        self._instantiated = True
        self.matrix = new LILMatrix(self.nb_post, self.nb_pre)

    @staticmethod
    cdef create_matrix(int nb_post, int nb_pre, LILMatrix *matrix):
        cdef LIL result = LIL(nb_post, nb_pre)
        result.matrix = matrix
        result._instantiated = True
        return result

    @property
    def shape(self):
        return (self.nb_post, self.nb_pre)

    @property
    def size(self):
        # Instantiate if necessary
        if not self._instantiated: self._instantiate()
        return self.matrix.get_size()

    @property
    def values(self):
        # Instantiate if necessary
        if not self._instantiated: self._instantiate()
        return self.matrix.values

    @property
    def ranks(self):
        # Instantiate if necessary
        if not self._instantiated: self._instantiate()
        return self.matrix.ranks

    # Methods
    def to_dense(self):
        """
        Returns a 2-dimensional numpy array equivalent to the sparse matrix, with zeros for the non-existing values.
        """
        # Instantiate if necessary
        if not self._instantiated: self._instantiate()

        # Create the numpy matrix
        res = np.zeros((self.nb_post, self.nb_pre))
        cdef Py_ssize_t idx_post, idx_pre
        cdef vector[int] ranks
        cdef vector[float] values
        for idx_post in range(self.nb_post):
            ranks = self.ranks[idx_post]
            values = self.values[idx_post]
            for idx_pre, rank in enumerate(ranks):
                res[idx_post, rank] = values[idx_pre]
        return res


    def fill_row(self, int idx, vector[int] ranks, vector[float] values):
        """
        Sets a row of the connectivity matrix (for a single post-synaptic neurons).

        Existing values for this neuron are erased.
        """
        # Instantiate if necessary
        if not self._instantiated: self._instantiate()

        # Fill the row
        self.matrix.fill_row(idx, ranks, values)

    def fill_random(self, proba:float, weights, diagonal=False):
        """
        Fills the sparse matrix with weights according to the given probability. Weights are sampled from the provided random distribution.

        Parameters:
            proba: Probability that a connections is connected.
        """
        # Instantiate if necessary
        if not self._instantiated: self._instantiate()

        # RNG
        rng = np.random.default_rng()

        # Iterate over all pre-post pairs
        cdef Py_ssize_t idx_post, idx_pre
        cdef list indices = []

        for idx_post in range(self.nb_post):
            indices.clear()

            for idx_pre in range(self.nb_pre):
                if not diagonal and idx_post == idx_pre: continue # self.connections
                if rng.random() < proba: indices.append(idx_pre)

            self.matrix.fill_row(idx_post, indices, weights.sample(len(indices)))


    def uniform_copy(self, value):
        """
        Returns a LIL matrix with the same ranks, but the weights are all set to the same value (e.g. 0 or 1)
        """
        return LIL.create_matrix(
            self.nb_post, self.nb_pre, 
            self.matrix.uniform_copy(value)
        )
        

    def add_scalar(self, value):
        """
        Returns a LIL matrix with the same ranks, but the values are incremented by the same value.
        """
        return LIL.create_matrix(
            self.nb_post, self.nb_pre, 
            self.matrix.add_scalar_copy(value)
        )

    def multiply_scalar(self, value):
        """
        Returns a LIL matrix with the same ranks, but the values are multiplied by the same value.
        """
        return LIL.create_matrix(
            self.nb_post, self.nb_pre, 
            self.matrix.multiply_scalar_copy(value)
        )

    def multiply_vector(self, value):
        """
        Returns a LIL matrix with the same ranks, but the values are from a right-side multiplication with a dense vector.
        """
        if value.ndim != 1:
            raise Exception("Sparse matrices can only be multiplied by vectors.")
        if value.size != self.nb_pre:
            raise Exception("The vector must have the same size as the second dimension of the sparse matrix.")
        
        return np.array(self.matrix.multiply_vector_copy(value))

    def outer(self, left, right):
        """
        Performs the outer product between two dense vectors, but only at locations where connections actually exist.

        ```python
        W = LIL(20, 10)
        pre = np.ones(10)
        error = np.ones(20)

        W += 0.1 * W.outer(error, pre)
        ```
        """
        if left.ndim != 1:
            raise Exception("Inputs must be 1D vectors.")
        if left.size != self.nb_post:
            raise Exception("The vector must have the same size as the first dimension of the sparse matrix.")
        if right.ndim != 1:
            raise Exception("Inputs must be 1D vectors.")
        if right.size != self.nb_pre:
            raise Exception("The vector must have the same size as the second dimension of the sparse matrix.")

        return LIL.create_matrix(
            self.nb_post, self.nb_pre, 
            self.matrix.outer_product(left, right)
        )
        
    ####################
    # Dunder methods
    ####################

    # Addition is quite straightforward in both directions
    def __add__(self, other): 
        if isinstance(other, (float, int, )):
            return self.add_scalar(float(other))
        else:
            raise Exception("Cannot add anything other than a scalar.")

    def __radd__(self, other): 
        if isinstance(other, (float, int, )):
            return self.add_scalar(float(other))
        else:
            raise Exception("Cannot add anything other than a scalar.")

    # Substraction: left is W - a (just add -a), right is a - W (multiply W by -1, then add a)
    def __sub__(self, other): 
        if isinstance(other, (float, int, )):
            return self.add_scalar(- float(other))
        else:
            raise Exception("Cannot substract anything other than a scalar.")

    def __rsub__(self, other): 
        if isinstance(other, (float, int, )):
            return self.multiply_scalar(-1).add_scalar(float(other))
        else:
            raise Exception("Cannot substract anything other than a scalar.")

    # Multiplication: straightforward with scalars in both directions
    def __mul__(self, other): 
        if isinstance(other, (float, int, )):
            return self.multiply_scalar(float(other))
        else:
            raise Exception("Cannot multiply by anything other than a scalar.")

    def __rmul__(self, other): 
        if isinstance(other, (float, int, )):
            return self.multiply_scalar(float(other))
        else:
            raise Exception("Cannot multiply by anything other than a scalar.")

    # Matrix multiplication: only accept W @ r, not r @ W
    def __matmul__(self, other): 
        if isinstance(other, (list, np.ndarray, )):
            return self.multiply_vector(np.array(other))
        else:
            raise Exception("Cannot multiply by anything other than a vector.")

    def __rmatmul__(self, other): 
        raise Exception("Sparse matrices cannot be left-multiplied by a vector.")


    # True division: only accept W / a, not a / W
    def __truediv__(self, other): 
        if isinstance(other, (float, int, )):
            return self.multiply_scalar(1./float(other))
        else:
            raise Exception("Cannot divide by anything other than a scalar.")

    def __rmatmul__(self, other): 
        raise Exception("Sparse matrices cannot be used in the denominator.")

    # In-place add
    def __iadd__(self, other): 
        if isinstance(other, (float, int, )):
            self.matrix.add_scalar_inplace(float(other))
        elif isinstance(other, (LIL, )):
            self.add_matrix(other)
        else:
            raise Exception("Cannot add this value.")
        return self

    def __isub__(self, other): 
        if isinstance(other, (float, int, )):
            self.matrix.add_scalar_inplace(-float(other))
        elif isinstance(other, (LIL, )):
            self.substract_matrix(other)
        else:
            raise Exception("Cannot substract this value.")
        return self

    cdef add_matrix(self, LIL other):
        self.matrix.add_matrix_inplace(other.matrix)
    cdef substract_matrix(self, LIL other):
        self.matrix.substract_matrix_inplace(other.matrix)