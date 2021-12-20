import numpy as np
import scipy as sci
import timeit
import functools

from numba.cuda import jit

from .ktensor import ktensor
from .dtensor import dtensor, unfolded_dtensor
from .ctools import kr, _rT, _cT, _compress, _sign_flip, _eiginit, _normalization, _arrange

#@jit(cache=True)
def ccp_als(X, r=None, c=True, p=10, q=1, tol=1E-5, maxiter=500, trace=True):
    """
    Randomized CP Decomposition using the Alternating Least Squares Method.
    
    Given a tensor X, the best rank-R CP model is estimated using the 
    alternating least-squares method.
    If `c=True` the input tensor is compressed using the randomized 
    QB-decomposition.

    
    Parameters
    ----------
    X : array_like or dtensor
        Real tensor `X` with dimensions `(I, J, K)`.
    
    r : int
        `r` denotes the number of components to compute.

    c : bool `{'True', 'False'}`, optional (default `c=True`)
        Whether or not to compress the tensor.         

    p : int, optional (default `p=10`)
        `p` sets the oversampling parameter.

    q : int, optional (default `q=2`)
        `q` sets the number of normalized power iterations.        
        
    tol : float, optional (default `tol=1E-5`)
        Stopping tolerance for reconstruction error.
        
    maxiter : int, optional (default `maxiter=500`)
        Maximum number of iterations to perform before exiting.

    trace : bool `{'True', 'False'}`, optional (default `trace=True`)
        Display progress.


    Returns
    -------
    P : ktensor
        Tensor stored in decomposed form as a Kruskal operator.

    
    Notes
    -----  
    
    
    References
    ----------
    Kolda, T. G. & Bader, B. W.
    "Tensor Decompositions and Applications." 
    SIAM Rev. 51 (2009): 455-500
    http://epubs.siam.org/doi/pdf/10.1137/07070111X

    Comon, Pierre & Xavier Luciani & Andre De Almeida. 
    "Tensor decompositions, alternating least squares and other tales."
    Journal of chemometrics 23 (2009): 393-405.
    http://onlinelibrary.wiley.com/doi/10.1002/cem.1236/abstract

    """

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Error catching
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
    
    if X.ndim < 3:
        raise ValueError("Array with ndim >= 3 expected.")

    if r is None:
        raise ValueError("Rank 'r' not given.")

    if r < 0 or r > np.min(X.shape):
        raise ValueError("Rank 'r' is invalid.")
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Init
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    t0 = timeit.default_timer() # start timer
    fit_out = []
    fit = 0
    rdiff = 1
    N = X.ndim
    normX = sci.linalg.norm(X) # Fro norm


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compress Tensor
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    if c==True:    
        Q , X = _compress(X, r=r, p=p, q=q)

        # Compare fro norm between the compressed and full tensor  
        normXc = sci.linalg.norm(X) # Fro norm
        rdiff =  normXc/normX   
        if trace==True: print('Shape of cTensor: ', X.shape )         
        if trace==True: print('Fro. norm of Tensor: %s,  cTensor: %s' %(normX, normXc) )
        if trace==True: print('Rel. difference of the Fro. norm: %s' %( round(1-rdiff,2) ))
        normX = normXc 


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize components [U_1, U_2, ... U_N] using the eig. decomposition
    # Note that only N-1 components are required for initialization
    # Hence, U_1 is assigned an empty list, i.e., U_1 = []
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    U = [_eiginit(X.unfold(n), r, n) for n in range(N)]
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Iterate the ALS algorithm until convergence or maxiter is reached
    # i)   compute the N gram matrices and multiply   
    # ii)  Compute Khatri-Rao Pseudoinverse
    # iii) Update component U_1, U_2, ... U_N
    # iv) Normalize columns of U_1, U_2, ... U_N to length 1
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for itr in range(maxiter):
        fitold = fit

        for n in range(N):
            
            # Select all components, but U_n
            components = [U[j] for j in range(N) if j != n]

            # i) compute the N-1 gram matrices 
            grams = [ arr.T.dot(arr) for arr in components ]             

            # ii)  Compute Khatri-Rao Pseudoinverse
            p1 = functools.reduce(kr, components[:-1][::-1], components[-1])
            p2 = sci.linalg.pinv(functools.reduce(sci.multiply, grams, 1.))

            # iii) Update component U_n            
            U[n] = X.unfold(n).dot( p1.dot(p2) )           

            # iv) normalize U_n to prevent singularities
            lamb = _normalization(U[n], itr)
            U[n] = U[n] / lamb

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute fit of the approximation,
        # defined as 1 - norm(X - full(P)) / norm(X) 
        # This is loosely the proportion of the data described by the CP model, 
        # i.e., fit of 1 is perfect
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
        P = ktensor(U, lamb)
        normresidual = normX**2 + P.norm()**2 - 2*P.innerprod(X)
        fit = (1 - (normresidual / normX ** 2)) * rdiff**2
        fit_out.append( fit )        
        
        fitchange = abs(fitold - fit)
        print('Iteration: %s fit: %s, fitchange: %s' %(itr, fit, fitchange))

        if itr > 0 and fitchange < tol:
            break

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Recover full-state components 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    P = ktensor(U, lamb)
    
    if c==True:
        for n in range(len(P.U)):
            if Q[n] is not None:
              P.U[n] = np.array(Q[n]).dot(P.U[n])
              
        P.shape = tuple([arr.shape[0] for arr in P.U])
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Normalize and sort components and store as ktensor
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    P = _arrange(P, None)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
    if trace==True: print('Compute time: %s seconds' %(timeit.default_timer()  - t0))
    return P

