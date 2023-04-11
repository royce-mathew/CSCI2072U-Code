# LU decomposition without pivoting (not for practical use)
#  Program for demonstrative purposes only 
#      - will only work in practice when each A(1:k, 1:k) is invertible (k=1:N)
#  See pseudo-code from Lecture 5 slides from class

import numpy as np

def LU_NoPivot(A):
    N = np.shape(A)[0]                              # extract number of rows/columns (note: A should be square)
    U = np.copy(A)                                  # copy contents of A (avoid "U=A" to avoid changing A)
    L = np.identity(N)                              # initialize L as the identoty matrix
    
    for jj in range(N-1):                           # loop over columns (i.e. pivots), excluding the last
        for ii in range(jj+1,N):                    # loop over column elements (i.e. rows) below the pivot
            L[ii,jj] = U[ii,jj]/U[jj,jj]            # compute and store the multiplier in appropriate spot in L
            U[ii,:] = U[ii,:] - L[ii,jj] * U[jj,:]  # elementary row operation that zeros column element below pivot
            
    return L,U

A = np.array([[2,1,1,0],[4,3,3,1],[8,7,9,5],[6,7,9,8]])   # does work
#A = np.array([[2,1,1,0],[4,3,3,1],[8,5,5,1],[6,7,9,8]])  # doesn't work

L,U = LU_NoPivot(A)

print(L)
print('\n')

print(U)
print('\n')

print(np.matmul(L,U)-A)



