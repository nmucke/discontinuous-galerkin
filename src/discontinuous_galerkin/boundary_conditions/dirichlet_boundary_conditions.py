import numpy as np
from abc import abstractmethod
from discontinuous_galerkin.base.base_boundary_conditions import BaseBoundaryConditions
import pdb

class DirichletBoundaryConditions(BaseBoundaryConditions):
    """
    Dirichlet boundary conditions class.

    This class contains the functionality for the Dirichlet boundary conditions.
    """
    def __init__(
        self, 
        DG_vars, 
        boundary_conditions, 
        flux, 
        numerical_flux,
        **args
        ):
        """Initialize the class."""

        super().__init__()

        self.DG_vars = DG_vars
        self.boundary_conditions = boundary_conditions
        self.num_BCs = 2#len(self.boundary_conditions(0))

        self.numerical_flux = numerical_flux
        self.flux = flux

    def _compute_ghost_states(self, t, q_boundary):
        """Compute the ghost states."""

        ghost_states = q_boundary.copy()

        for i in range(self.DG_vars.num_states):
            for edge, idx in zip(['left', 'right'], [0, -1]):
                
                bc = self.boundary_conditions(t, q_boundary)[i][edge]
                
                if bc is not None:                              
                    ghost_states[i, idx] = -q_boundary[i, idx] + 2 * bc

        return ghost_states

    def _compute_ghost_flux(self, ghost_states):
        """Compute the ghost states."""
        
        return self.flux(ghost_states)


    def apply_boundary_conditions(self, t, q_boundary, flux_boundary, **args):
        """Apply the boundary conditions."""
        # Compute the ghost states
        ghost_states = self._compute_ghost_states(t, q_boundary)
        
        # Compute the ghost flux
        ghost_flux = self._compute_ghost_flux(ghost_states)

        # Compute numerical boundary flux
        numerical_flux = self.numerical_flux(
            q_inside = q_boundary,
            q_outside = ghost_states,
            flux_inside = flux_boundary,
            flux_outside = ghost_flux,
            on_boundary=True,
        )
        
        return numerical_flux[:, 0], numerical_flux[:, -1]

