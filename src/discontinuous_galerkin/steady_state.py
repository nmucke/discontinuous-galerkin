
from functools import partial
import numpy as np
import matplotlib.pyplot as plt

from discontinuous_galerkin.nonlinear_solvers.newton import NewtonSolver
import pdb

def steady_state_rhs(q, rhs, DG_vars):
        """Compute the RHS of the implicit Euler equation."""
        
        q = np.reshape(q, (DG_vars.num_states, DG_vars.Np*DG_vars.K), order='F')
        pde_rhs = rhs(t=0, q=q).flatten('F')
        
        return pde_rhs

def compute_steady_state(q, rhs, newton_params, DG_vars):
    """Compute the steady state solution."""

    q = q.flatten('F')

    newton_solver = NewtonSolver(**newton_params)

    func = lambda q: steady_state_rhs(q=q, rhs=rhs, DG_vars=DG_vars)

    q = newton_solver.solve(func=func, q=q)

    q = q.reshape(
            (DG_vars.num_states, DG_vars.Np*DG_vars.K),
            order='F'
            )
    return q

        
    