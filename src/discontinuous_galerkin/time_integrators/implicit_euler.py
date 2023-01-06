import numpy as np
from discontinuous_galerkin.base.base_time_integrator import BaseTimeIntegrator
import pdb
from discontinuous_galerkin.nonlinear_solvers.newton import NewtonSolver

import matplotlib.pyplot as plt

from scipy.optimize import newton_krylov, newton

class ImplicitEuler(BaseTimeIntegrator):
    """Implicit Euler time integrator."""

    def __init__(
        self, 
        DG_vars, 
        stabilizer,
        max_newton_iter=200, 
        newton_tol=1e-5, 
        step_size=1e-1
        ):

        self.DG_vars = DG_vars
        self.stabilizer = stabilizer
        self.time = 0
        self.max_newton_iter = max_newton_iter
        self.newton_tol = newton_tol
        self.step_size = step_size

        self.newton_solver = NewtonSolver()

        self.num_DOFs = self.DG_vars.num_states*self.DG_vars.Np*self.DG_vars.K

        self.idx = 0

        self.indices = np.arange(self.num_DOFs)
        self.indices = self.indices % (self.DG_vars.Np*self.DG_vars.K)
        self.state_indices = np.zeros(self.num_DOFs)
        for i in range(2*self.DG_vars.num_states):
            self.state_indices[i*self.num_DOFs:(i+1)*self.num_DOFs] = i
        self.state_indices = self.state_indices.astype(int)

    def time_step(self, t, q, step_size, rhs):
        """Take a time step."""

        q_init = q.copy()
        q_init = q_init.flatten('F')

        time=t

        RHS = lambda q: (1/step_size*(q - q_init) - rhs(time+step_size, np.reshape(q, (self.DG_vars.num_states, self.DG_vars.Np*self.DG_vars.K), 'F')).flatten('F'))

        q_old = newton_krylov(
            F = RHS,
            xin = q_init,
        )

        q_old = q_old.reshape(
            (self.DG_vars.num_states, self.DG_vars.Np*self.DG_vars.K),
            order='F'
            )
        '''



        #if self.idx%5 == 0:
        #    self.J = self.newton_solver._compute_jacobian(time,q_init,rhs)

        func = lambda q: rhs(
            time + step_size, 
            np.reshape(
                q, 
                (self.DG_vars.num_states, self.DG_vars.Np*self.DG_vars.K),
                'F'
            )).flatten('F')

        self.J = self.newton_solver._compute_jacobian(
            f = func,
            x=q_init
            )

        LHS = 1/step_size*np.eye(self.num_DOFs) - self.J

        newton_error = 1e8
        iterations = 0

        q_old = q_init.copy()
        while newton_error > self.newton_tol and \
            iterations < self.max_newton_iter:
            
            RHS = -(1/step_size*(q_old - q_init) - rhs(time+step_size, q_old)).flatten('F')

            delta_q = np.linalg.solve(LHS,RHS)
            delta_q = delta_q.reshape(
                (self.DG_vars.num_states, self.DG_vars.Np*self.DG_vars.K),
                 order='F'
                 )

            alpha = 1.
            q_old = q_old + alpha*delta_q

            newton_error = np.max(np.abs(delta_q))
            iterations = iterations + 1

        '''
        q_old = self.stabilizer(q_old)
            
        self.idx += 1
        return q_old, t+step_size
