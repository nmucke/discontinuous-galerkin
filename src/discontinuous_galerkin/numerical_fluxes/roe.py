import numpy as np
from abc import abstractmethod
from discontinuous_galerkin.base.base_numerical_flux import BaseNumericalFlux
import pdb
from scipy.linalg import eig

class RoeFlux(BaseNumericalFlux):
    """
    Roe numerical flux class.

    This class contains the functionality for the Roe numerical flux.
    """
    def __init__(self, DG_vars, eigen=None, system_jacobian=None, primitive_to_conservative=None, conservative_to_primitive=None):
        """Initialize the class."""

        super(RoeFlux).__init__()

        self.DG_vars = DG_vars

        self.eigen = eigen
        self.system_jacobian = system_jacobian

        self.primitive_to_conservative = primitive_to_conservative
        self.conservative_to_primitive = conservative_to_primitive

        self.nx_boundary = np.array(
            [self.DG_vars.nx[self.DG_vars.mapI], self.DG_vars.nx[self.DG_vars.mapO]]
        )
    
    def _average_operator(self, q_inside, q_outside):
        """Compute the average operator."""

        return (q_inside + q_outside) / 2
    
    def _compute_eigen_from_system_jacobian(self, q):
        """Compute the eigenvalues of the system."""
        A = self.system_jacobian(q)
        d, l, r = self.eigen = eig(A, left=True, right=True)
        
        return d, l, r


    def _get_eigen(self, q_avg, q_diff):
        """Compute the eigenvalues of the system."""

        D = np.zeros((q_avg.shape[1], self.DG_vars.num_states, self.DG_vars.num_states))
        L = np.zeros((q_avg.shape[1], self.DG_vars.num_states, self.DG_vars.num_states))
        R = np.zeros((q_avg.shape[1], self.DG_vars.num_states, self.DG_vars.num_states))

        RDL_q_prod = np.zeros((self.DG_vars.num_states, q_avg.shape[1]))
        if self.system_jacobian is not None:
            for i in range(q_avg.shape[1]):
                #D[:, :, i], L[:, :, i], R[:, :, i] = \
                #    self._compute_eigen_from_system_jacobian(q[0, i])
                D, L, R = \
                    self._compute_eigen_from_system_jacobian(q_avg[:, i])
        else:
            for i in range(q_avg.shape[1]):
                D[i, :, :], L[i, :, :], R[i, :, :] = self.eigen(q_avg[:, i])
                #D, L, R = self.eigen(q_avg[:, i])
                #RDL_q_prod[:, i] = np.matmul(R, np.matmul(np.abs(D), L)) @ q_diff[:, i]
                
        RDL = np.matmul(R, np.matmul(np.abs(D), L))
        RDL_q_prod = np.matmul(RDL, q_diff.T[:, :, None]).squeeze(-1).T
        return RDL_q_prod

    def compute_numerical_flux(
        self, 
        q_inside, 
        q_outside, 
        flux_inside, 
        flux_outside,
        on_boundary=False,
        primitive_to_conservative=None,
        conservative_to_primitive=None,
        ):
        """Compute the numerical flux."""

        # Compute the average of the fluxes
        flux_average = self._average_operator(flux_inside, flux_outside)

        # Compute the jump of the states
        q_roe_average = self._average_operator(q_inside, q_outside)


        # Compute the difference of the states
        if on_boundary:
            q_inside_cons = np.stack([self.primitive_to_conservative(q_inside[:, 0]), self.primitive_to_conservative(q_inside[:, 1])], axis=1)
            q_outside_cons = np.stack([self.primitive_to_conservative(q_outside[:, 0]), self.primitive_to_conservative(q_outside[:, 1])], axis=1)
        else:
            q_inside_cons = primitive_to_conservative(q_inside)
            q_outside_cons = primitive_to_conservative(q_outside)
        q_diff_cons = (q_outside_cons - q_inside_cons)/2

        # Compute the eigenvalues of the system
        RDL_q_prod = self._get_eigen(q_roe_average, q_diff_cons)


        if on_boundary:
            # Compute the numerical flux
            numerical_flux = flux_average - self.nx_boundary*RDL_q_prod

        else:
            # Compute the numerical flux
            numerical_flux = flux_average - self.DG_vars.nx*RDL_q_prod

        
        return numerical_flux

