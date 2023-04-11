from typing import Callable
import numpy as np;

def EulerCheb_method(f: Callable, df: Callable, df2: Callable, x0: float, k_max: int, eps_x: float, eps_f: float):
   # order of arguments:  function f(x), derivative, 2nd derivative, initial guess, max iterations, tolerance on approximate error, tolerance on residual
    x: float = x0 # Initial Guess 
    iteration_history = [] # Stores the iteration history

    for k in range(k_max):
        fx: float = f(x)                 # current function value
        dx: float = fx / df(x)  
        ddx: float = -dx - (df2(x) / (2 * df(x))) * (dx**2) # Update Step
        err: float = abs(ddx)              # current error estimate
        res: float = abs(fx)              # current residual
    
        print(f'Iteration {k+1}: x={x} err={err:.4e}, res={res:.4e}')   
        iteration_history.append([k+1, x, err, res]) 

        if err < eps_x and res < eps_f:       # If converged ...
            break;
        x += ddx
    else:
        raise Exception(f"No convergence after {k_max} iterations")

    iteration_history = np.array(iteration_history);
    return x, iteration_history, True
