import numpy as np
from scipy.linalg import schur, eigvals
from scipy.linalg import eig

A = np.array([[1,0,0], [0,2,0], [0,0,3]])

eig, v = eig(A)
print(eig, v)

eig, v = np.linalg.eig(A)
print(eig, v)

 = schur(A)