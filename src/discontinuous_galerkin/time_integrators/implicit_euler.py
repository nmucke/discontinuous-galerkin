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

        self.L = 10000 # meters
        self.d = 0.146 # meters
        self.A = np.pi*self.d**2/4 # meters^2
        self.rho_g_norm = 1.26 # kg/m^3
        self._rho_l = 1000. # kg/m^3
        self.p_amb = 101325.
        self.p_norm = 1.0e5 # Pa
        self.p_outlet = 1.0e6 # Pa
        self.e = 1e-8 # meters
        self.mu_g = 1.8e-5 # Pa*s
        self.mu_l = 8.9e-4 # Pa*s
        self.Cd = 5e-4
        self.T_norm = 300 # Kelvin
        self.T = 278 # Kelvin

        self.rho_l = 1000. # kg/m^3

        self.DG_vars = DG_vars
        self.stabilizer = stabilizer

        self.newton_solver = NewtonSolver(**newton_params)

    
    def density_to_pressure(self, rho):
        """Compute the pressure from the density."""

        return self.p_norm * rho * self.T / self.rho_g_norm / self.T_norm


    def pressure_to_density(self, p):
        """Compute the density from the pressure."""

        return self.rho_g_norm * self.T_norm / self.p_norm * p / self.T

    def primitive_to_conservative(self, q):
        """Compute the conservative variables from the primitive variables."""

        A_l = q[0]
        p = q[1]
        u_m = q[2]

        rho_g = self.pressure_to_density(p)

        A_g = self.A - A_l

        rho_m = rho_g * A_g + self.rho_l * A_l

        rho_g_A_g = rho_g * A_g
        rho_l_A_l = self.rho_l * A_l
        rho_m_u_m = rho_m * u_m

        return np.array([rho_g_A_g, rho_l_A_l, rho_m_u_m])
    
    def conservative_to_primitive(self, q):
        """Compute the primitive variables from the conservative variables."""

        A_l = q[1]/self.rho_l
        A_g = self.A - A_l

        rho_g = q[0]/A_g

        p = self.density_to_pressure(rho_g)

        rho_m = rho_g * A_g + self.rho_l * A_l

        u_m = q[2]/rho_m

        return np.array([A_l, p, u_m])

    def _implicit_euler_rhs(self, q, t_new, q_old, rhs, step_size):
        """Compute the RHS of the implicit Euler equation."""
        
        q = np.reshape(
            q,
            (self.DG_vars.num_states, self.DG_vars.Np*self.DG_vars.K),
            order='F'
            )
        q_old = np.reshape(
            q_old,
            (self.DG_vars.num_states, self.DG_vars.Np*self.DG_vars.K),
            order='F'
            )
        q = self.primitive_to_conservative(q)
        q_old = self.primitive_to_conservative(q_old)

        q = q.flatten('F')
        q_old = q_old.flatten('F')

        time_derivative = (q - q_old)/step_size

        q = np.reshape(
            q,
            (self.DG_vars.num_states, self.DG_vars.Np*self.DG_vars.K),
            order='F'
        )
        q = self.conservative_to_primitive(q)

        #q = q.flatten('F')

        pde_rhs = rhs(
            t = t_new, 
            q = q#np.reshape(
                #q, 
                #(self.DG_vars.num_states, self.DG_vars.Np*self.DG_vars.K), 
                #order='F'
                #)
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