import numpy as np
from ctensor import ccp_als, ccp_bcd
from ctensor import dtensor, ktensor
from ctensor import toydata

X = toydata(m=250, t=150, background=0, display=1)
print(X)
T = dtensor(X)
P = ccp_als(T , r=4, c=True, p=10, q=2, maxiter=500)
print(P.lmbda)
A,B,C = P.U