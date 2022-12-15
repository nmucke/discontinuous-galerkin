import numpy as np
from abc import abstractmethod
from discontinuous_galerkin.base.base_boundary_conditions import BaseBoundaryConditions
import pdb

class DirichletBoundaryConditions(BaseBoundaryConditions):
    """
    Dirichlet boundary conditions class.

    This class contains the functionality for the Dirichlet boundary conditions.
    """
    def __init__(self, DG_vars, boundary_conditions):
        """Initialize the class."""

        super().__init__(DG_vars)

        self.DG_vars = DG_vars
        self.boundary_conditions = boundary_conditions

        self.num_BCs = len(self.boundary_conditions)



    def _compute_ghost_states(self, t, q_boundary):
        """Compute the ghost states."""

        ghost_states = {}

        for i, BC in enumerate(self.boundary_conditions):
            for edge, idx in zip(['left', 'right'], [0, -1]):
                if BC[edge] is not None:
                    ghost_states[edge] = \
                        -q_boundary[i, idx] + 2 * BC[edge](t)
                else:
                    ghost_states[edge] = q_boundary[i, idx]

        return ghost_states

    def _compute_ghost_flux(self, flux_boundary):
        """Compute the ghost states."""

        ghost_flux = {}

        for i, BC in enumerate(self.boundary_conditions):
            for edge, idx in zip(['left', 'right'], [0, -1]):
                if BC[edge] is not None:
                    ghost_flux[edge] = -flux_boundary[i, idx]
                else:
                    ghost_flux[edge] = flux_boundary[i, idx]

        return ghost_flux


    def apply_boundary_conditions(self, q):
        """Apply the boundary conditions."""

        # Apply the boundary conditions

        return q

