
import numpy as np
from numpy import array, prod, argsort
from .core import tensor_mixin, khatrirao
from .pyutils import inherit_docstring_from, from_to_without
from numba.cuda import jit

__all__ = [
    'dtensor',
    'unfolded_dtensor',
]


class dtensor(tensor_mixin, np.ndarray):
    """
    Class to store **dense** tensors

    Parameters
    ----------
    input_array : np.ndarray
        Multidimenional numpy array which holds the entries of the tensor

    """
    #@jit(cache=True)
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj
    #@jit(cache=True)
    def __array_wrap__(self, out_arr, context=None):
        return np.ndarray.__array_wrap__(self, out_arr, context)
    #@jit(cache=True)
    def __eq__(self, other):
        return np.equal(self, other)
    #@jit(cache=True)
    def _ttm_compute(self, V, mode, transp):
        sz = array(self.shape)
        r1, r2 = from_to_without(0, self.ndim, mode, separate=True)
        #r1 = list(range(0, mode))
        #r2 = list(range(mode + 1, self.ndim))
        order = [mode] + r1 + r2
        newT = self.transpose(axes=order)
        newT = newT.reshape(sz[mode], prod(sz[r1 + list(range(mode + 1, len(sz)))]))
        if transp:
            newT = V.T.dot(newT)
            p = V.shape[1]
        else:
            newT = V.dot(newT)
            p = V.shape[0]
        newsz = [p] + list(sz[:mode]) + list(sz[mode + 1:])
        newT = newT.reshape(newsz)
        # transpose + argsort(order) equals ipermute
        newT = newT.transpose(argsort(order))
        return dtensor(newT)
    #@jit(cache=True)
    def _ttv_compute(self, v, dims, vidx, remdims):
        """
        Tensor times vector product

        Parameter
        ---------
        """
        if not isinstance(v, tuple):
            raise ValueError('v must be a tuple of vectors')
        ndim = self.ndim
        order = list(remdims) + list(dims)
        if ndim > 1:
            T = self.transpose(order)
        sz = array(self.shape)[order]
        for i in np.arange(len(dims), 0, -1):
            T = T.reshape((sz[:ndim - 1].prod(), sz[ndim - 1]))
            T = T.dot(v[vidx[i - 1]])
            ndim -= 1
        if ndim > 0:
            T = T.reshape(sz[:ndim])
        return T

    def ttt(self, other, modes=None):
        pass

    #@jit(cache=True)
    def unfold(self, mode):
        """
        Unfolds a dense tensor in mode n.

        Parameters
        ----------
        mode : int
            Mode in which tensor is unfolded

        Returns
        -------
        unfolded_dtensor : unfolded_dtensor object
            Tensor unfolded along mode
        """

        sz = array(self.shape)
        N = len(sz)
        order = ([mode], from_to_without(N - 1, -1, mode, step=-1, skip=-1))
        newsz = (sz[order[0]][0], prod(sz[order[1]]))
        arr = self.transpose(axes=(order[0] + order[1]))
        arr = arr.reshape(newsz)
        return unfolded_dtensor(arr, mode, self.shape)

    #@jit(cache=True)
    def norm(self):
        """
        Computes the Frobenius norm for dense tensors
        :math:`norm(X) = \sqrt{\sum_{i_1,\ldots,i_N} x_{i_1,\ldots,i_N}^2}`

        """
        return np.linalg.norm(self)

    @inherit_docstring_from(tensor_mixin)
    def uttkrp(self, U, n):
        order = list(range(n)) + list(range(n + 1, self.ndim))
        Z = khatrirao(tuple(U[i] for i in order), reverse=True)
        return self.unfold(n).dot(Z)

    @inherit_docstring_from(tensor_mixin)
    def transpose(self, axes=None):
        return dtensor(np.transpose(array(self), axes=axes))


class unfolded_dtensor(np.ndarray):

    def __new__(cls, input_array, mode, ten_shape):
        obj = np.asarray(input_array).view(cls)
        obj.ten_shape = ten_shape
        obj.mode = mode
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.ten_shape = getattr(obj, 'ten_shape', None)
        self.mode = getattr(obj, 'mode', None)
    #@jit(cache=True)
    def fold(self):
        shape = array(self.ten_shape)
        N = len(shape)
        order = ([self.mode], from_to_without(0, N, self.mode, reverse=True))
        arr = self.reshape(tuple(shape[order[0]],) + tuple(shape[order[1]]))
        arr = np.transpose(arr, argsort(order[0] + order[1]))
        return dtensor(arr)
