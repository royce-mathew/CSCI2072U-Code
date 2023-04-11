#  This file contains a function that evaluates the Jacobian of f(x)
#    (used in Newton's method for solving the system of nonlinear equations f(x) = 0 ) 

import numpy as np

def Jacobian(x):     # input numpy array containing elements of the vector x          
    x1 = x[0]
    x2 = x[1]
    J11 = (x2**2) + (2*x1*x2) + 3*(x1**2)
    J12 = (2*x1*x2) + (x1**2)
    J21 = (3*(x1**2)*(x2**2)) - (10*(x1**4)*x2) - (2*x1)
    J22 = (2*(x1**3)*x2) - (2*(x1**5))
    return np.array([[J11, J12], [J21, J22]])      # return numpy array with Jacobian matrix