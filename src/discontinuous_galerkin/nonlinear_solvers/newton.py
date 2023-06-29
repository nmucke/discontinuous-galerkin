import numpy as np
import pdb

from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt

from scipy.optimize import newton_krylov
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu, spsolve


from scipy.optimize import BroydenFirst, KrylovJacobian
from scipy.optimize import InverseJacobian


class NewtonSolver(object):
        """Newton solver for nonlinear problems."""    
        def __init__(
            self,
            solver: str = 'krylov',
            max_newton_iter: int = 100,
            newton_tol: float = 1e-5,
            num_jacobian_reuses: int = 500,
            reuse_jacobian: bool = True,
            use_jacobian_inverse: bool = True,
            ):

            self.max_newton_iter = max_newton_iter
            self.newton_tol = newton_tol
            self.solver = solver

            self.reuse_jacobian = reuse_jacobian
            self.num_jacobian_reuses = num_jacobian_reuses
            self.jacobian_reuse_counter = 0

            self.use_jacobian_inverse = use_jacobian_inverse

            self.first_solve = True
        
        def _compute_jacobian(self, func, x, dx=1e-8):
            """Compute the Jacobian of a function."""
            n = len(x)
            f = func(x)
            self.jac = np.zeros((n, n))

            Dx = np.abs(x) * dx
            Dx[x==0] = dx

            #x_plus = np.tile(np.expand_dims(x, 1), (1, n))
            #np.fill_diagonal(x_plus, x+Dx)
            #f_plus = np.apply_along_axis(f, 0, x_plus)
            #jac = (f_plus - func[:, None])/Dx

            for idx in range(n):  # through columns to allow for vector addition
                x_plus = x.copy()
                x_plus[idx] += Dx[idx]
                self.jac[:, idx] = (func(x_plus) - f)/Dx[idx]

            return self.jac
        '''
        def _compute_jacobian(self, func, x, dx=1e-8):
            """Compute the Jacobian of a function."""
            n = len(x)
            f = func(cp.asarray(x))
            self.jac = cp.zeros((n, n))

            Dx = cp.abs(x) * dx
            Dx[x == 0] = dx

            for j in range(n):  # through columns to allow for vector addition
                x_plus = x.copy()
                x_plus[j] += Dx[j]
                self.jac[:, j] = (func(cp.asarray(x_plus)) - f) / Dx[j]

            return cp.asnumpy(self.jac)
        '''

        def _compute_residual(self, func, q):
            """Compute the residual of the nonlinear problem."""
                        
            return func(q)
        
        def _solve_direct(self, func, q):
            """Solve the nonlinear problem using a direct solver."""

            # Compute the Jacobian
            if self.first_solve:
                self.jacobian = self._compute_jacobian(func, q)
                #self.jac_lu, self.jac_piv = lu_factor(self.jacobian)
                self.jacobian = csc_matrix(self.jacobian)
                self.jac_lu = splu(csc_matrix(self.jacobian))
                self.first_solve = False

            if self.reuse_jacobian and self.jacobian_reuse_counter < self.num_jacobian_reuses:
                self.jacobian_reuse_counter += 1
            else:
                self.jacobian = self._compute_jacobian(func, q)
                #self.jac_lu, self.jac_piv = lu_factor(self.jacobian)
                self.jacobian = csc_matrix(self.jacobian)
                self.jac_lu = splu(csc_matrix(self.jacobian))
                self.jacobian_reuse_counter = 0
            
            for _ in range(self.max_newton_iter):
                # Compute the residual
                residual = -self._compute_residual(func, q)

                #q = np.reshape(q, (2, 3*200), order='F')
                #residual = np.reshape(residual, (2, 3*200), order='F')
                
                # Compute the update
                #delta_q = np.linalg.solve(self.jacobian, residual)
                #delta_q = lu_solve((self.jac_lu, self.jac_piv), residual)
                delta_q = self.jac_lu.solve(residual)
                
                # Update the solution
                q = q + delta_q

                # Check for convergence
                #if np.max(np.abs(delta_q)) < self.newton_tol:
                #   return q
                if np.max(np.abs(residual)) < self.newton_tol:
                    return q
            
            raise Exception('Newton solver did not converge.')
        
        def _solve_krylov(self, func, q):
            """Solve the nonlinear problem using a Krylov solver."""

            if self.use_jacobian_inverse:
                # Compute the Jacobian
                if self.first_solve:
                    jacobian = self._compute_jacobian(func, q)
                    self.inv_jacobian = np.linalg.inv(jacobian)
                    self.first_solve = False

                if self.reuse_jacobian and self.jacobian_reuse_counter < self.num_jacobian_reuses:
                    self.jacobian_reuse_counter += 1
                else:
                    jacobian = self._compute_jacobian(func, q)
                    self.inv_jacobian = np.linalg.inv(jacobian)
                    self.jacobian_reuse_counter = 0
                    
                q = newton_krylov(
                    F=func,
                    xin=q, 
                    f_tol=self.newton_tol, 
                    maxiter=self.max_newton_iter,
                    inner_M=self.inv_jacobian
                    )
            else:
                q = newton_krylov(
                    F=func,
                    xin=q, 
                    f_tol=self.newton_tol, 
                    maxiter=self.max_newton_iter,
                    )

            return q

        def solve(self, func, q):
            
            if self.solver == 'direct':
                return self._solve_direct(func, q)
            elif self.solver == 'krylov':
                return self._solve_krylov(func, q)
            else:
                raise Exception(f'Unknown solver: {self.solver}')
