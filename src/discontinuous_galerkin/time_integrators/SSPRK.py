import numpy as np
from discontinuous_galerkin.base.base_time_integrator import BaseTimeIntegrator
import pdb

import matplotlib.pyplot as plt

class SSPRK(BaseTimeIntegrator):
    """Strong Stability Preserving Runge-Kutta time integrator class."""    

    def __init__(
        self, 
        DG_vars, 
        stabilizer,
        primitive_to_conservative=None,
        conservative_to_primitive=None,
        **kwargs
        ):

        self.time = 0
        self.DG_vars = DG_vars
        self.stabilizer = stabilizer

        self.primitive_to_conservative = primitive_to_conservative
        self.conservative_to_primitive = conservative_to_primitive
        

        self.a = np.array([
            [1, 0, 0],
            [3/4, 1/4, 0],
            [1/3, 0, 2/3]
        ])
        self.b = np.array([1, 1/4, 2/3])
        self.c = np.array([0, 1, 1/2])
    
    def time_step(self, t, q, step_size, rhs):
        """Perform a single time step."""

        q_cons = self.primitive_to_conservative(q)

        q_new_cons = self.a[0,0]*q_cons + self.b[0]*step_size*rhs(t + self.c[0]*step_size, q)
        q_new_cons = self.stabilizer(q_new_cons)
        q_new_prim = self.conservative_to_primitive(q_new_cons)
        q_new_prim = self.stabilizer(q_new_prim)

        q_new_cons = self.a[1,0]*q_cons + self.a[1,1]*q_new_cons + self.b[1]*step_size*rhs(t + self.c[1]*step_size, q_new_prim)
        q_new_cons = self.stabilizer(q_new_cons)
        q_new_prim = self.conservative_to_primitive(q_new_cons)
        q_new_prim = self.stabilizer(q_new_prim)

        q_new_cons = self.a[2,0]*q_cons + self.a[2,2]*q_new_prim + self.b[2]*step_size*rhs(t+ self.c[2]*step_size,q_new_prim)
        q_new_cons = self.stabilizer(q_new_cons)
        q_new_prim = self.conservative_to_primitive(q_new_cons)
        q_new_prim = self.stabilizer(q_new_prim)

        return q_new_prim, t + step_size