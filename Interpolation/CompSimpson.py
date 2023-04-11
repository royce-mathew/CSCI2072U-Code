import numpy as np

def CompSimpson(f,a,b,n):
    # Code for Composite Simpson's rule
    h = (b-a) / n
    x = np.linspace(a, b, n+1)

    I = 0
    for k in range(1, n+1):
        xk = (x[k-1] + x[k])/2
        I += f(x[k-1]) + 4*f(xk) + f(x[k]) 
    
    I = h/6 * I

    return I