import numpy as np
from discontinuous_galerkin.base.base_time_integrator import BaseTimeIntegrator
import pdb

import matplotlib.pyplot as plt

class SSPRK(BaseTimeIntegrator):
    """Strong Stability Preserving Runge-Kutta time integrator class."""    

    def __init__(self, DG_vars, stabilizer):

        self.time = 0
        self.DG_vars = DG_vars
        self.stabilizer = stabilizer
        

        self.a = np.array([[1, 0, 0],
                               [3/4, 1/4, 0],
                               [1/3, 0, 2/3]])
        self.b = np.array([1, 1/4, 2/3])
        self.c = np.array([0, 1, 1/2])

    def time_step(self, t, q, step_size, rhs):
        """Perform a single time step."""
        
        q_new = self.a[0,0]*q + \
             self.b[0]*step_size*rhs(t + self.c[0]*step_size,q)
        q_new = self.stabilizer(q_new)

        q_new = self.a[1,0]*q + self.a[1,1]*q_new + \
            self.b[1]*step_size*rhs(t + self.c[1]*step_size,q_new)
        q_new = self.stabilizer(q_new)

        q_new = self.a[2,0]*q + self.a[2,2]*q_new + \
            self.b[2]*step_size*rhs(t+ self.c[2]*step_size,q_new)
        q_new = self.stabilizer(q_new)

        return q_new, t + step_size