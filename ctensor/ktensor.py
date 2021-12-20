
import numpy as np
from numba.cuda import jit
from numpy import dot, ones, array, outer, zeros, prod, sum
from .core import khatrirao, tensor_mixin
from .dtensor import dtensor

__all__ = [
    'ktensor',
    'vectorized_ktensor',
]


class ktensor(object):
    """
    Tensor stored in decomposed form as a Kruskal operator.

    Intended Usage
        The Kruskal operator is particularly useful to store
        the results of a CP decomposition.

    Parameters
    ----------
    U : list of ndarrays
        Factor matrices from which the tensor representation
        is created. All factor matrices ``U[i]`` must have the
        same number of columns, but can have different
        number of rows.
    lmbda : array_like of floats, optional
        Weights for each dimension of the Kruskal operator.
        ``len(lambda)`` must be equal to ``U[i].shape[1]``

    See also
    --------
    sktensor.dtensor : Dense tensors
    sktensor.sptensor : Sparse tensors
    sktensor.ttensor : Tensors stored in form of the Tucker operator

    References
    ----------
    .. [1] B.W. Bader, T.G. Kolda
           Efficient Matlab Computations With Sparse and Factored Tensors
           SIAM J. Sci. Comput, Vol 30, No. 1, pp. 205--231, 2007
    """

    def __init__(self, U, lmbda=None):
        self.U = U
        self.shape = tuple(Ui.shape[0] for Ui in U)
        self.ndim = len(self.shape)
        self.rank = U[0].shape[1]
        self.lmbda = lmbda
        if not all(array([Ui.shape[1] for Ui in U]) == self.rank):
            raise ValueError('Dimension mismatch of factor matrices')
        if lmbda is None:
            self.lmbda = ones(self.rank)
    #@jit(cache=True)
    def __eq__(self, other):
        if isinstance(other, ktensor):
            # avoid costly elementwise comparison for obvious cases
            if self.ndim != other.ndim or self.shape != other.shape:
                return False
            # do elementwise comparison
            return all(
                [(self.U[i] == other.U[i]).all() for i in range(self.ndim)] +
                [(self.lmbda == other.lmbda).all()]
            )
        else:
            # TODO implement __eq__ for tensor_mixins and ndarrays
            raise NotImplementedError()
    #@jit(cache=True)
    def uttkrp(self, U, mode):

        """
        Unfolded tensor times Khatri-Rao product for Kruskal tensors

        Parameters
        ----------
        X : tensor_mixin
            Tensor whose unfolding should be multiplied.
        U : list of array_like
            Matrices whose Khatri-Rao product should be multiplied.
        mode : int
            Mode in which X should be unfolded.

        See also
        --------
        sktensor.sptensor.uttkrp : Efficient computation of uttkrp for sparse tensors
        ttensor.uttkrp : Efficient computation of uttkrp for Tucker operators
        """
        N = self.ndim
        if mode == 1:
            R = U[1].shape[1]
        else:
            R = U[0].shape[1]
        W = np.tile(self.lmbda, 1, R)
        for i in range(mode) + range(mode + 1, N):
            W = W * dot(self.U[i].T, U[i])
        return dot(self.U[mode], W)
    #@jit(cache=True)
    def norm(self):
        """
        Efficient computation of the Frobenius norm for ktensors

        Returns
        -------
        norm : float
            Frobenius norm of the ktensor
        """
        N = len(self.shape)
        coef = outer(self.lmbda, self.lmbda)
        for i in range(N):
            coef = coef * dot(self.U[i].T, self.U[i])
        return np.sqrt(coef.sum())
    #@jit(cache=True)
    def innerprod(self, X):
        """
        Efficient computation of the inner product of a ktensor with another tensor

        Parameters
        ----------
        X : tensor_mixin
            Tensor to compute the inner product with.

        Returns
        -------
        p : float
            Inner product between ktensor and X.
        """
        N = len(self.shape)
        R = len(self.lmbda)
        res = 0
        for r in range(R):
            vecs = []
            for n in range(N):
                vecs.append(self.U[n][:, r])
            res += self.lmbda[r] * X.ttv(tuple(vecs))
        return res
    #@jit(cache=True)
    def toarray(self):
        """
        Converts a ktensor into a dense multidimensional ndarray

        Returns
        -------
        arr : np.ndarray
            Fully computed multidimensional array whose shape matches
            the original ktensor.
        """
        A = dot(self.lmbda, khatrirao(tuple(self.U)).T)
        return A.reshape(self.shape)
    #@jit(cache=True)
    def totensor(self):
        """
        Converts a ktensor into a dense tensor

        Returns
        -------
        arr : dtensor
            Fully computed multidimensional array whose shape matches
            the original ktensor.
        """
        return dtensor(self.toarray())
    #@jit(cache=True)
    def tovec(self):
        v = zeros(sum([s * self.rank for s in self.shape]))
        offset = 0
        for M in self.U:
            noff = offset + prod(M.shape)
            v[offset:noff] = M.flatten()
            offset = noff
        return vectorized_ktensor(v, self.shape, self.lmbda)


class vectorized_ktensor(object):
    #@jit(cache=True)
    def __init__(self, v, shape, lmbda):
        self.v = v
        self.shape = shape
        self.lmbda = lmbda
    #@jit(cache=True)
    def toktensor(self):
        order = len(self.shape)
        rank = len(self.v) / sum(self.shape)
        U = [None for _ in range(order)]
        offset = 0
        for i in range(order):
            noff = offset + self.shape[i] * rank
            U[i] = self.v[offset:noff].reshape((self.shape[i], rank))
            offset = noff
        return ktensor(U, self.lmbda)

# vim: set et:
