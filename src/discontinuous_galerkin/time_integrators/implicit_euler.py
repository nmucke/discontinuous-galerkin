import numpy as np
from discontinuous_galerkin.time_integrators.base_time_integrator import BaseTimeIntegrator
import pdb

import matplotlib.pyplot as plt

class ImplicitEuler(BaseTimeIntegrator):
    """Implicit Euler time integrator."""

    def __init__(
        self, 
        DG_vars, 
        max_newton_iter=200, 
        newton_tol=1e-5, 
        step_size=1e-1
        ):

        self.DG_vars = DG_vars
        self.time = 0
        self.max_newton_iter = max_newton_iter
        self.newton_tol = newton_tol
        self.step_size = step_size

        self.num_DOFs = self.DG_vars.num_states*self.DG_vars.Np*self.DG_vars.K

        self.idx = 0

        self.indices = np.arange(self.num_DOFs)
        self.indices = self.indices % (self.DG_vars.Np*self.DG_vars.K)
        self.state_indices = np.zeros(self.num_DOFs)
        for i in range(2*self.DG_vars.num_states):
            self.state_indices[i*self.num_DOFs:(i+1)*self.num_DOFs] = i
        self.state_indices = self.state_indices.astype(int)

    def compute_jacobian(self, time, q, rhs):
        """Compute the Jacobian matrix."""

        epsilon = np.finfo(float).eps

        J = np.zeros((self.num_DOFs, self.num_DOFs))

        F = rhs(time, q)

        Upert = q.copy()

        pert_jac = np.sqrt(epsilon) * np.maximum(np.abs(q), 1)
        for global_idx, (state_idx, state_DOF_idx) in \
            enumerate(zip(self.state_indices, self.indices)):

            perturbation = pert_jac[state_idx, state_DOF_idx]

            #self.pert[state_idx, state_DOF_idx] = perturbation
            Upert[state_idx, state_DOF_idx] += perturbation

            Fpert = rhs(time, Upert)

            J[:, global_idx] = ((Fpert - F) / perturbation).flatten('F')

            Upert[state_idx, state_DOF_idx] -= perturbation

        return J


    def time_step(self, t, q, step_size, rhs):
        """Take a time step."""
        
        q_init = q.copy()

        time=t

        if self.idx%5 == 0:
            self.J = self.compute_jacobian(time,q_init,rhs)

        LHS = 1/step_size*np.eye(self.num_DOFs) - self.J

        newton_error = 1e8
        iterations = 0

        q_old = q_init.copy()
        while newton_error > self.newton_tol and \
            iterations < self.max_newton_iter:

            RHS = -(1/step_size*(q_old - q_init) - rhs(time, q_old)).flatten('F')

            delta_q = np.linalg.solve(LHS,RHS)
            delta_q = delta_q.reshape(
                (self.DG_vars.num_states, self.DG_vars.Np*self.DG_vars.K),
                 order='F'
                 )

            alpha = 1.
            q_old = q_old + alpha*delta_q

            newton_error = np.max(np.abs(delta_q))
            iterations = iterations + 1
            
        self.idx += 1
        return q_old, t+step_size
