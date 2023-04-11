# CSCI / MATH 2072U -- Computational Science 1
# Tutorial 3
# PA=LU decomposition 
#                    GNU GENERAL PUBLIC LICENSE
#                       Version 3, 29 June 2007

import numpy as np

def LUP(A):
    n = np.shape(A)[0]                         # extract matrix size
    U = np.copy(A).astype(float)                # copy content of A (avoid linking U and A)
    L = np.zeros((n, n))                         # initialize L and P
    P = np.identity(n)

    for j in range(n):
        # We add j here to account for the pivot column getting smaller
        k = np.argmax(np.abs(U[j:n, j])) + j; # Find pivot element
        
        if k != j:  # if the pivot is not on the diagonal...
            swap(U, j, k, start=j, end=n) # Swap rows of U
            if j > 0: # Check if we are on the first iteration
               swap(L, j, k, end=j) # swap rows of L left of diagonal element
            swap(P, j, k) # Swap rows of P

        L[j, j] = 1;
        for i in range(j + 1, n):       # Gauss elimination of rows below pivot
            L[i, j] = U[i, j] / U[j, j] # Store multiplier in L matrix
            U[i, j:n] = U[i, j:n] - L[i, j] * U[j, j:n] # Update row j in U matrix
    return L,U,P


"""
    M: Matrix we want to swap
    P: P is the row
    Q: Another row we want to swap
    k: number of elements of the rows that get swapped
"""
def swap(M, p, q, start=0, end=None):
    assert 0 <= p, "p must be lesser than or equal to 0"
    assert q <= np.shape(A)[0], "q must be lesser than or equal to the n"
    
    if end is None:
        end = np.shape(M)[0]
    
    # Swap rows p and q 
    M[[p, q], start:end] = M[[q, p], start:end]
    return M   


A = np.array([
    [7, 3, -1, 2],
    [3, 8, 1, -4],
    [-1, 1, 4, -1],
    [2, -4, -1, 6]
])


L, U, P = LUP(A)
print(f"L: {L}\n\nU: {U}\n\nP: {P}")
PA = P.dot(A)
LU = L.dot(U)
print(PA)
print(LU)
