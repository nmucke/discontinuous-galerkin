import numpy as np
import pdb

#import autograd.numpy as np
from autograd import grad, jacobian

import matplotlib.pyplot as plt

class NewtonSolver(object):
        """Newton solver for nonlinear problems."""    
        def __init__(self, ):
            pass

        def _compute_jacobian(self, f, x, dx=1e-12):
            """Compute the Jacobian of a function."""

            x = x.flatten('F')

            n = len(x)
            func = f(x)
            jac = np.zeros((n, n))

            Dx = np.abs(x) * dx
            Dx[x==0] = dx

            '''
            x_plus = np.tile(np.expand_dims(x, 1), (1, n))

            np.fill_diagonal(x_plus, x+Dx)

            f_plus = np.apply_along_axis(f, 0, x_plus)

            jac = (f_plus - func[:, None])/Dx
            '''
            #Dx_diag = np.diag(Dx)
            #x_plus = np.zeros((n, n))
            for j in range(n):  # through columns to allow for vector addition
                x_plus = x.copy()
                x_plus[j] = x_plus[j] + Dx[j]
                #jac[:, j] = (f(x + Dx_diag[:,j]) - func)/Dx[j]
                jac[:, j] = (f(x_plus) - func)/Dx[j]
                #jac[:, j] = (f_plus[:,j] - func)/Dx[j]
            return jac

        def _compute_residual(self, q, t, step_size):
            """Compute the residual of the nonlinear problem."""
            
            # Compute the right hand side
            rhs = self.rhs(t, q)
            
            # Compute the residual
            residual = q - step_size*rhs
            
            return residual
        
        def _compute_update(self, q, t, step_size):
            # Compute the Jacobian
            jacobian = self._compute_jacobian(q, t, step_size)
            
            # Compute the residual
            residual = self._compute_residual(q, t, step_size)
            
            # Compute the update
            update = np.linalg.solve(jacobian, residual)
            
            return update
    
