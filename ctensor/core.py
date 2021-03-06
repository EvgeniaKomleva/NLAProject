
import numpy as np
from numpy import array, dot, zeros, ones, arange, kron
from numpy import setdiff1d
from scipy.linalg import eigh
from scipy.sparse import issparse as issparse_mat
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from abc import ABCMeta, abstractmethod
from .pyutils import is_sequence, func_attr
#from coremod import khatrirao
from numba.cuda import jit
import sys
import types

module_funs = []

#@jit(cache=True)
def modulefunction(func):
    module_funs.append(func_attr(func, 'name'))


class tensor_mixin(object):
    """
    Base tensor class from which all tensor classes are subclasses.
    Can not be instaniated

    See also
    --------
    sktensor.dtensor : Subclass for *dense* tensors.
    sktensor.sptensor : Subclass for *sparse* tensors.
    """

    __metaclass__ = ABCMeta
    #@jit(cache=True)
    def ttm(self, V, mode=None, transp=False, without=False):
        """
        Tensor times matrix product

        Parameters
        ----------
        V : M x N array_like or list of M_i x N_i array_likes
            Matrix or list of matrices for which the tensor times matrix
            products should be performed
        mode : int or list of int's, optional
            Modes along which the tensor times matrix products should be
            performed
        transp: boolean, optional
            If True, tensor times matrix products are computed with
            transpositions of matrices
        without: boolean, optional
            It True, tensor times matrix products are performed along all
            modes **except** the modes specified via parameter ``mode``


        """
        if mode is None:
            mode = range(self.ndim)
        if isinstance(V, np.ndarray):
            Y = self._ttm_compute(V, mode, transp)
        elif is_sequence(V):
            dims, vidx = check_multiplication_dims(mode, self.ndim, len(V), vidx=True, without=without)
            Y = self._ttm_compute(V[vidx[0]], dims[0], transp)
            for i in range(1, len(dims)):
                Y = Y._ttm_compute(V[vidx[i]], dims[i], transp)
        return Y
    #@jit(cache=True)
    def ttv(self, v, modes=[], without=False):
        """
        Tensor times vector product

        Parameters
        ----------
        v : 1-d array or tuple of 1-d arrays
            Vector to be multiplied with tensor.
        modes : array_like of integers, optional
            Modes in which the vectors should be multiplied.
        without : boolean, optional
            If True, vectors are multiplied in all modes **except** the
            modes specified in ``modes``.

        """
        if not isinstance(v, tuple):
            v = (v, )
        dims, vidx = check_multiplication_dims(modes, self.ndim, len(v), vidx=True, without=without)
        for i in range(len(dims)):
            if not len(v[vidx[i]]) == self.shape[dims[i]]:
                raise ValueError('Multiplicant is wrong size')
        remdims = np.setdiff1d(range(self.ndim), dims)
        return self._ttv_compute(v, dims, vidx, remdims)

    #@abstractmethod
    #def ttt(self, other, modes=None):
    #    pass

    @abstractmethod
    def _ttm_compute(self, V, mode, transp):
        pass

    @abstractmethod
    def _ttv_compute(self, v, dims, vidx, remdims):
        pass

    @abstractmethod
    def unfold(self, rdims, cdims=None, transp=False):
        pass

    @abstractmethod
    def uttkrp(self, U, mode):
        """
        Computes the _matrix_ product of the unfolding
        of a tensor and the Khatri-Rao product of multiple matrices.
        Efficient computations are perfomed by the respective
        tensor implementations.

        Parameters
        ----------
        U : list of array-likes
            Matrices for which the Khatri-Rao product is computed and
            which are multiplied with the tensor in mode ``mode``.
        mode: int
            Mode in which the Khatri-Rao product of ``U`` is multiplied
            with the tensor.

        Returns
        -------
        M : np.ndarray
            Matrix which is the result of the matrix product of the unfolding of
            the tensor and the Khatri-Rao product of ``U``

        See also
        --------
        For efficient computations of unfolded tensor times Khatri-Rao products
        for specialiized tensors see also
        dtensor.uttkrp, sptensor.uttkrp, ktensor.uttkrp, ttensor.uttkrp

        References
        ----------
        .. [1] B.W. Bader, T.G. Kolda
               Efficient Matlab Computations With Sparse and Factored Tensors
               SIAM J. Sci. Comput, Vol 30, No. 1, pp. 205--231, 2007
        """
        pass

    @abstractmethod
    def transpose(self, axes=None):
        """
        Compute transpose of tensors.

        Parameters
        ----------
        axes : array_like of ints, optional
            Permute the axes according to the values given.

        Returns
        -------
        d : tensor_mixin
            tensor with axes permuted.

        See also
        --------
        dtensor.transpose, sptensor.transpose
        """
        pass


def istensor(X):
    return isinstance(X, tensor_mixin)


# dynamically create module level functions
conv_funcs = [
    'norm',
    'transpose',
    'ttm',
    'ttv',
    'unfold',
]

for fname in conv_funcs:
    def call_on_me(obj, *args, **kwargs):
        if not istensor(obj):
            raise ValueError('%s() object must be tensor (%s)' % (fname, type(obj)))
        func = getattr(obj, fname)
        return func(*args, **kwargs)

    nfunc = types.FunctionType(
        func_attr(call_on_me, 'code'),
        {
            'getattr': getattr,
            'fname': fname,
            'istensor': istensor,
            'ValueError': ValueError,
            'type': type
        },
        name=fname,
        argdefs=func_attr(call_on_me, 'defaults'),
        closure=func_attr(call_on_me, 'closure')
    )
    setattr(sys.modules[__name__], fname, nfunc)


def check_multiplication_dims(dims, N, M, vidx=False, without=False):
    dims = array(dims, ndmin=1)
    if len(dims) == 0:
        dims = arange(N)
    if without:
        dims = setdiff1d(range(N), dims)
    if not np.in1d(dims, arange(N)).all():
        raise ValueError('Invalid dimensions')
    P = len(dims)
    sidx = np.argsort(dims)
    sdims = dims[sidx]
    if vidx:
        if M > N:
            raise ValueError('More multiplicants than dimensions')
        if M != N and M != P:
            raise ValueError('Invalid number of multiplicants')
        if P == M:
            vidx = sidx
        else:
            vidx = sdims
        return sdims, vidx
    else:
        return sdims

#@jit(cache=True)
def innerprod(X, Y):
    """
    Inner prodcut with a Tensor
    """
    return dot(X.flatten(), Y.flatten())

#@jit(cache=True)
def nvecs(X, n, rank, do_flipsign=True, dtype=np.float):
    """
    Eigendecomposition of mode-n unfolding of a tensor
    """
    Xn = X.unfold(n)
    if issparse_mat(Xn):
        Xn = csr_matrix(Xn, dtype=dtype)
        Y = Xn.dot(Xn.T)
        _, U = eigsh(Y, rank, which='LM')
    else:
        Y = Xn.dot(Xn.T)
        N = Y.shape[0]
        _, U = eigh(Y, eigvals=(N - rank, N - 1))
        #_, U = eigsh(Y, rank, which='LM')
    # reverse order of eigenvectors such that eigenvalues are decreasing
    U = array(U[:, ::-1])
    # flip sign
    if do_flipsign:
        U = flipsign(U)
    return U

#@jit(cache=True)
def flipsign(U):
    """
    Flip sign of factor matrices such that largest magnitude
    element will be positive
    """
    midx = abs(U).argmax(axis=0)
    for i in range(U.shape[1]):
        if U[midx[i], i] < 0:
            U[:, i] = -U[:, i]
    return U

#@jit(cache=True)
def center(X, n):
    Xn = unfold(X, n)
    N = Xn.shape[0]
    m = Xn.sum(axis=0) / N
    m = kron(m, ones((N, 1)))
    Xn = Xn - m
    return fold(Xn, n)

#@jit(cache=True)
def center_matrix(X):
    m = X.mean(axis=0)
    return X - m

#@jit(cache=True)
def scale(X, n):
    Xn = unfold(X, n)
    m = np.float_(np.sqrt((Xn ** 2).sum(axis=1)))
    m[m == 0] = 1
    for i in range(Xn.shape[0]):
        Xn[i, :] = Xn[i] / m[i]
    return fold(Xn, n, X.shape)


# TODO more efficient cython implementation
#@jit(cache=True)
def khatrirao(A, reverse=False):
    """
    Compute the columnwise Khatri-Rao product.

    Parameters
    ----------
    A : tuple of ndarrays
        Matrices for which the columnwise Khatri-Rao product should be computed

    reverse : boolean
        Compute Khatri-Rao product in reverse order

    Examples
    --------
    >>> A = np.random.randn(5, 2)
    >>> B = np.random.randn(4, 2)
    >>> C = khatrirao((A, B))
    >>> C.shape
    (20, 2)
    >>> (C[:, 0] == np.kron(A[:, 0], B[:, 0])).all()
    true
    >>> (C[:, 1] == np.kron(A[:, 1], B[:, 1])).all()
    true
    """

    if not isinstance(A, tuple):
        raise ValueError('A must be a tuple of array likes')
    N = A[0].shape[1]
    M = 1
    for i in range(len(A)):
        if A[i].ndim != 2:
            raise ValueError('A must be a tuple of matrices (A[%d].ndim = %d)' % (i, A[i].ndim))
        elif N != A[i].shape[1]:
            raise ValueError('All matrices must have same number of columns')
        M *= A[i].shape[0]
    matorder = arange(len(A))
    if reverse:
        matorder = matorder[::-1]
    # preallocate
    P = np.zeros((M, N), dtype=A[0].dtype)
    for n in range(N):
        ab = A[matorder[0]][:, n]
        for j in range(1, len(matorder)):
            ab = np.kron(ab, A[matorder[j]][:, n])
        P[:, n] = ab
    return P

#@jit(cache=True)
def teneye(dim, order):
    """
    Create tensor with superdiagonal all one, rest zeros
    """
    I = zeros(dim ** order)
    for f in range(dim):
        idd = f
        for i in range(1, order):
            idd = idd + dim ** (i - 1) * (f - 1)
        I[idd] = 1
    return I.reshape(ones(order) * dim)

#@jit(cache=True)
def tvecmat(m, n):
    d = m * n
    i2 = arange(d).reshape(m, n).T.flatten()
    Tmn = zeros((d, d))
    Tmn[arange(d), i2] = 1
    return Tmn

    #i = arange(d);
    #rI = m * (i-1)-(m*n-1) * floor((i-1)/n)
    #print rI
    #I1s = s2i((d,d), rI, arange(d))
    #print I1s
    #Tmn[I1s] = 1
    #return Tmn.reshape((d,d)).T

# vim: set et:
