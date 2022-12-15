import numpy as np
from abc import abstractmethod
from discontinuous_galerkin.base.base_boundary_conditions import BaseBoundaryConditions
import pdb

class DirichletBoundaryConditions(BaseBoundaryConditions):
    """
    Dirichlet boundary conditions class.

    This class contains the functionality for the Dirichlet boundary conditions.
    """
    def __init__(self, DG_vars, boundary_conditions, flux, numerical_flux):
        """Initialize the class."""

        super().__init__()

        self.DG_vars = DG_vars
        self.boundary_conditions = boundary_conditions
        self.num_BCs = len(self.boundary_conditions(0))

        self.numerical_flux = numerical_flux
        self.flux = flux

    def _compute_ghost_states(self, t, q_boundary):
        """Compute the ghost states."""

        '''
        ghost_states = []


        for i, BC in enumerate(self.boundary_conditions(t)):
            for edge, idx in zip(['left', 'right'], [0, -1]):
                if BC[edge] is not None:
                    ghost_states.append(
                        {edge: -q_boundary[i, idx] + 2 * BC[edge]}
                        )
                else:
                    ghost_states.append(
                        {edge: q_boundary[i, idx]}
                        )
        '''

        q_left = self.boundary_conditions(t)[0]['left']

        ghost_states = np.array([[
            -q_boundary[0, 0] + 2 * q_left,
            q_boundary[0, 1]
            ]])
        return ghost_states

    def _compute_ghost_flux(self, ghost_states):
        """Compute the ghost states."""
        
        return self.flux(ghost_states)


    def apply_boundary_conditions(self, t, q_boundary, flux_boundary):
        """Apply the boundary conditions."""
        
        # Compute the ghost states
        ghost_states = self._compute_ghost_states(t, q_boundary)
        
        # Compute the ghost flux
        ghost_flux = self._compute_ghost_flux(ghost_states)

        # Compute numerical boundary flux
        numerical_flux_left = self.numerical_flux(
            q_inside = q_boundary[:, 0],
            q_outside = ghost_states[:, 0],
            flux_inside = flux_boundary[:, 0],
            flux_outside = ghost_flux[:, 0],
        )

        numerical_flux_right = self.numerical_flux(
            q_inside = q_boundary[:, -1],
            q_outside = ghost_states[:, -1],
            flux_inside = flux_boundary[:, -1],
            flux_outside = ghost_flux[:, -1],
        )
        
        return numerical_flux_left[0:1], numerical_flux_right[-1:]

