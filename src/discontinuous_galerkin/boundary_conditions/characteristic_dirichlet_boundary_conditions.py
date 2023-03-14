import numpy as np
from abc import abstractmethod
from discontinuous_galerkin.base.base_boundary_conditions import BaseBoundaryConditions
import pdb
from scipy.linalg import eig

class CharacteristicDirichletBoundaryConditions(BaseBoundaryConditions):
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
        system_jacobian,
        source=None,
        conservative_to_primitive_transform_matrix=None,
        **args
        ):
        """Initialize the class."""

        super().__init__()


        self.DG_vars = DG_vars
        self.boundary_conditions = boundary_conditions
        self.num_BCs = 2#len(self.boundary_conditions(0))
        self.system_jacobian = system_jacobian
        self.source = source
        self.numerical_flux = numerical_flux
        self.flux = flux
        self.conservative_to_primitive_transform_matrix = conservative_to_primitive_transform_matrix

    def _solve_boundary_ode(self, t, q_boundary, q_boundary_diff, step_size, side):

        P, P_inv, A, C, S, S_inv, Lambda = \
            self.conservative_to_primitive_transform_matrix(t, q_boundary)

        L = Lambda @ S_inv @ P_inv @ q_boundary_diff
        if side == 'left':
            L[1] = 0
        else:
            L[0] = 0.
        
        ode = lambda q: P @ (S @ L)# + C)

        ghost_state = q_boundary + ode(q_boundary) * step_size

        return ghost_state

    def _compute_characteristic_ghost_states(self, t, q_boundary, q_boundary_diff, step_size):
        """Compute the ghost states."""

        ghost_states = np.zeros(q_boundary.shape)

        for edge, idx in zip(['left', 'right'], [0, -1]):

            ghost_states[:, idx] = self._solve_boundary_ode(
                t=t, 
                q_boundary=q_boundary[:, idx], 
                q_boundary_diff=q_boundary_diff[:, idx],
                step_size=step_size,
                side=edge
            )

            for i in range(self.DG_vars.num_states):

                bc = self.boundary_conditions(t, q_boundary)[i][edge]
                
                if bc is not None:
                    ghost_states[i, idx] = -q_boundary[i, idx] + 2 * bc
            
        return ghost_states
    
    def _compute_ghost_flux(self, ghost_states):
        """Compute the ghost states."""
        
        return self.flux(ghost_states)


    def apply_boundary_conditions(
            self, 
            t, 
            q_boundary, 
            flux_boundary, 
            q_boundary_diff,
            step_size
        ):
        """Apply the boundary conditions."""
        
        # Compute the ghost states
        ghost_states = self._compute_characteristic_ghost_states(
            t, 
            q_boundary, 
            q_boundary_diff,
            step_size=step_size
        )
        
        # Compute the ghost flux
        ghost_flux = self._compute_ghost_flux(ghost_states)

        # Compute numerical boundary flux
        numerical_flux = self.numerical_flux(
            q_inside = q_boundary,
            q_outside = ghost_states,
            flux_inside = flux_boundary,
            flux_outside = ghost_flux,
            on_boundary=True
        )
        
        return numerical_flux[:, 0], numerical_flux[:, -1]

