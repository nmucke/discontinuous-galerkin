import numpy as np
import pdb

import matplotlib.pyplot as plt

from functools import partial

from discontinuous_galerkin.nonlinear_solvers.newton import NewtonSolver
from discontinuous_galerkin.base.base_time_integrator import BaseTimeIntegrator


class ImplicitEuler(BaseTimeIntegrator):
    """Implicit Euler time integrator."""

    def __init__(
        self, 
        DG_vars, 
        stabilizer,
        newton_params={
            'solver': 'krylov',
            'max_newton_iter': 200,
            'newton_tol': 1e-5
            }
        ):

        self.DG_vars = DG_vars
        self.stabilizer = stabilizer

        self.newton_solver = NewtonSolver(**newton_params)

    def _implicit_euler_rhs(self, q, t_new, q_old, rhs, step_size):
        """Compute the RHS of the implicit Euler equation."""
        time_derivative = (q - q_old)/step_size
        pde_rhs = rhs(
            t = t_new, 
            q = np.reshape(
                q, 
                (self.DG_vars.num_states, self.DG_vars.Np*self.DG_vars.K), 
                order='F'
                )
            ).flatten('F')
        return (time_derivative - pde_rhs)

    def time_step(self, t, q, step_size, rhs):
        """Take a time step."""

        q_init = q.copy()
        q_init = q_init.flatten('F')

        t_new = t + step_size

        implicit_euler_rhs = partial(
            self._implicit_euler_rhs,
            t_new = t_new,
            q_old = q.flatten('F'),
            rhs = rhs,
            step_size = step_size
            )

        q = self.newton_solver.solve(
            func = implicit_euler_rhs,
            q = q_init,
            )

        q = q.reshape(
            (self.DG_vars.num_states, self.DG_vars.Np*self.DG_vars.K),
            order='F'
            )
        q = self.stabilizer(q)
            
        return q, t_new