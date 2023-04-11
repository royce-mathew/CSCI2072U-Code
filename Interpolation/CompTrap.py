import numpy as np

def CompTrap(f,a,b,n):

    # Code for composite trapezoidal rule
    # Calculate step size
    xs = np.linspace(a, b, n+1)
    h = (b - a) / float(n)

    # Find sum
    I = (f(a) + f(b)) / 2.0

    for i in range(1, n):
        I += f(xs[i])

    # Finding final integrations value
    I *= h
    
    return I