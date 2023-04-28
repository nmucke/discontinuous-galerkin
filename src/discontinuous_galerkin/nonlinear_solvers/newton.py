import numpy as np
import pdb


import matplotlib.pyplot as plt


from scipy.optimize import newton_krylov

class NewtonSolver(object):
        """Newton solver for nonlinear problems."""    
        def __init__(
            self,
            solver='krylov',
            max_newton_iter=200,
            newton_tol=1e-5,
            reuse_jacobian=True
            ):
            
            self.max_newton_iter = max_newton_iter
            self.newton_tol = newton_tol
            self.solver = solver

            self.reuse_jacobian = reuse_jacobian
            self.num_jacobian_reuses = 5
            self.jacobian_reuse_counter = 0

        def _compute_jacobian(self, func, x, dx=1e-8):
            """Compute the Jacobian of a function."""
            n = len(x)
            f = func(x)
            self.jac = np.zeros((n, n))

            Dx = np.abs(x) * dx
            Dx[x==0] = dx

            '''
            x_plus = np.tile(np.expand_dims(x, 1), (1, n))

            np.fill_diagonal(x_plus, x+Dx)

            f_plus = np.apply_along_axis(f, 0, x_plus)

            jac = (f_plus - func[:, None])/Dx
            '''
            for j in range(n):  # through columns to allow for vector addition
                x_plus = x.copy()
                x_plus[j] += Dx[j]
                self.jac[:, j] = (func(x_plus) - f)/Dx[j]
            
            return self.jac

        def _compute_residual(self, func, q):
            """Compute the residual of the nonlinear problem."""
                        
            return func(q)
        
        def _solve_direct(self, func, q):
            """Solve the nonlinear problem using a direct solver."""

            # Compute the Jacobian
            if self.reuse_jacobian and self.jacobian_reuse_counter == 0:
                self.jac = self._compute_jacobian(func, q)

            if self.reuse_jacobian and self.jacobian_reuse_counter < self.num_jacobian_reuses:
                jacobian = self.jac
                self.jacobian_reuse_counter += 1
            else:
                jacobian = self._compute_jacobian(func, q)
                self.jacobian_reuse_counter = 0
            
            for _ in range(self.max_newton_iter):
                # Compute the residual
                residual = -self._compute_residual(func, q)
                
                # Compute the update
                delta_q = np.linalg.solve(jacobian, residual)
                
                # Update the solution
                q = q + delta_q

                # Check for convergence
                if np.max(np.abs(delta_q)) < self.newton_tol:
                    return q
            
            raise Exception('Newton solver did not converge.')
        
        def _solve_krylov(self, func, q):
            """Solve the nonlinear problem using a Krylov solver."""

            q = newton_krylov(
                F = func,
                xin = q, 
                f_tol=self.newton_tol, 
                maxiter=self.max_newton_iter
                )

            return q

        def solve(self, func, q):

            if self.solver == 'direct':
                return self._solve_direct(func, q)
            elif self.solver == 'krylov':
                return self._solve_krylov(func, q)
            else:
                raise Exception(f'Unknown solver: {self.solver}')
