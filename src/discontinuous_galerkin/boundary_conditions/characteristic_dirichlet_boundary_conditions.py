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

        self.L = 2000
        self.d = 0.508
        self.A = np.pi*self.d**2/4
        self.c = 308.
        self.p_amb = 101325.
        self.p_ref = 5016390.
        self.rho_ref = 52.67
        self.e = 1e-8
        self.mu = 1.2e-5
        self.Cd = 5e-4


    def _get_primitive_form(self, q):
        """Get the primitive form of the state variables."""


        
        A = 2

        P = np.array(
            [[],
            []])


        return A, P, 



    def _compute_eigen_from_system_jacobian(self, q):
        """Compute the eigenvalues of the system."""
        A = self.system_jacobian(q)
        d, l, r = self.eigen = eig(A, left=True, right=True)
        
        return d, l, r

    def _compute_characteristic_ghost_states(self, t, q_boundary):
        """Compute the ghost states."""
        
        ghost_states = q_boundary
        

        return ghost_states

    def _compute_ghost_states(self, t, q_boundary):
        """Compute the ghost states."""

        ghost_states = q_boundary

        J_left = self.system_jacobian(q_boundary[:,0])
        J_right = self.system_jacobian(q_boundary[:,-1])

        # Compute the eigenvalues and eigenvectors
        d_left, l_left, r_left = self._compute_eigen_from_system_jacobian(q_boundary[:,0])
        d_right, l_right, r_right = self._compute_eigen_from_system_jacobian(q_boundary[:,-1])

        inverse_l_left = np.linalg.inv(l_left)
        pdb.set_trace()



        for i in range(self.DG_vars.num_states):
            for edge, idx in zip(['left', 'right'], [0, -1]):
                bc = self.boundary_conditions(t, q_boundary)[i][edge]
                
                if bc is not None:
                    ghost_states[i, idx] = -q_boundary[i, idx] + 2 * bc
            

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
        numerical_flux = self.numerical_flux(
            q_inside = q_boundary,
            q_outside = ghost_states,
            flux_inside = flux_boundary,
            flux_outside = ghost_flux,
            on_boundary=True
        )
        
        return numerical_flux[:, 0], numerical_flux[:, -1]

