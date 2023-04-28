import numpy as np
from discontinuous_galerkin.base.base_time_integrator import BaseTimeIntegrator
import pdb

import matplotlib.pyplot as plt

class SSPRK(BaseTimeIntegrator):
    """Strong Stability Preserving Runge-Kutta time integrator class."""    

    def __init__(self, DG_vars, stabilizer, **kwargs):

        self.time = 0
        self.DG_vars = DG_vars
        self.stabilizer = stabilizer
        

        self.a = np.array([[1, 0, 0],
                               [3/4, 1/4, 0],
                               [1/3, 0, 2/3]])
        self.b = np.array([1, 1/4, 2/3])
        self.c = np.array([0, 1, 1/2])


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
        self.T_norm = 278 # Kelvin

        self.rho_l = 1000. # kg/m^3

    
    def conservative_to_primitive(self, q):
        """Compute the primitive variables from the conservative variables."""

        A_l = q[1]/self.rho_l
        A_g = self.A - A_l

        rho_g = q[0]/A_g

        p = self.density_to_pressure(rho_g)

        rho_m = rho_g * A_g + self.rho_l * A_l

        u_m = q[2]/rho_m

        return A_l, p, u_m

    def density_to_pressure(self, rho):
        """Compute the pressure from the density."""

        return self.p_norm * rho * self.T_norm / self.rho_g_norm / self.T_norm

    def pressure_to_density(self, p):
        """Compute the density from the pressure."""

        return self.rho_g_norm * self.T_norm / self.p_norm * p / self.T_norm

    def time_step(self, t, q, step_size, rhs):
        """Perform a single time step."""

        A_l = q[0]
        p = q[1]
        u_m = q[2]

        A_g = self.A - A_l

        rho_g = self.pressure_to_density(p)

        rho_m = rho_g * A_g + self.rho_l * A_l

        q_conv = np.zeros_like(q)
        q_conv[0] = A_g * rho_g
        q_conv[1] = A_l * self.rho_l
        q_conv[2] = rho_m * u_m


        q_new = self.a[0,0]*q_conv + \
             self.b[0]*step_size*rhs(t + self.c[0]*step_size,q)
        q_new = self.stabilizer(q_new)

        A_l, p, u_m = self.conservative_to_primitive(q_new)
        q = np.array([A_l, p, u_m])

        q_new = self.a[1,0]*q_conv + self.a[1,1]*q_new + \
            self.b[1]*step_size*rhs(t + self.c[1]*step_size,q)
        q_new = self.stabilizer(q_new)

        A_l, p, u_m = self.conservative_to_primitive(q_new)
        q = np.array([A_l, p, u_m])

        q_new = self.a[2,0]*q_conv + self.a[2,2]*q_new + \
            self.b[2]*step_size*rhs(t+ self.c[2]*step_size,q)
        q_new = self.stabilizer(q_new)

        A_l = q[1]/self.rho_l

        A_g = self.A - A_l

        rho = q[0]/A_g
        p = self.density_to_pressure(rho)
        u_m = q[2]/(q[0] + q[1])

        q_new[0] = A_l
        q_new[1] = p
        q_new[2] = u_m

        return q_new, t + step_size