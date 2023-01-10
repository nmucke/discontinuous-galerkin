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
    def __init__(self, DG_vars, eigen=None, system_jacobian=None):
        """Initialize the class."""

        super(RoeFlux).__init__()

        self.DG_vars = DG_vars

        self.eigen = eigen
        self.system_jacobian = system_jacobian

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
        on_boundary=False
        ):
        """Compute the numerical flux."""

        # Compute the average of the fluxes
        flux_average = self._average_operator(flux_inside, flux_outside)


        # Compute the jump of the states
        q_roe_average = self._average_operator(q_inside, q_outside)
        #rho_inner = q_inside[0, :]
        #rho_outer = q_outside[0, :]
        #q_roe_average = 1/(np.sqrt(rho_inner) + np.sqrt(rho_outer))
        #q_roe_average = q_roe_average * (np.sqrt(rho_inner) * q_inside + np.sqrt(rho_outer) * q_outside)

        # Compute the difference of the states
        q_diff = (q_outside - q_inside)/2

        # Compute the eigenvalues of the system
        RDL_q_prod = self._get_eigen(q_roe_average, q_diff)
        pdb.set_trace()


        if on_boundary:
            # Compute the numerical flux
            numerical_flux = flux_average - self.nx_boundary*RDL_q_prod

            u_inside = q_inside[1]/q_inside[0]
            u_outside = q_outside[1]/q_outside[0]

            # Compute the velocity
            A = 0.2026829916389991
            C_inside = np.abs(u_inside) + 308/np.sqrt(A)
            C_outside = np.abs(u_outside) + 308/np.sqrt(A)
            C = np.maximum(np.abs(C_inside), np.abs(C_outside))

            q_jump = self.nx_boundary * (q_inside - q_outside)
            lax_numerical_flux = flux_average + C * 0.5 * (1 - 0) * q_jump

            print(lax_numerical_flux-numerical_flux)
        else:
            # Compute the numerical flux
            numerical_flux = flux_average - self.DG_vars.nx*RDL_q_prod


            u_inside = q_inside[1]/q_inside[0]
            u_outside = q_outside[1]/q_outside[0]

            # Compute the velocity
            A = 0.2026829916389991
            C_inside = np.abs(u_inside) + 308/np.sqrt(A)
            C_outside = np.abs(u_outside) + 308/np.sqrt(A)
            C = np.maximum(np.abs(C_inside), np.abs(C_outside))

            q_jump = self.DG_vars.nx * (q_inside - q_outside)
            lax_numerical_flux = flux_average + C * 0.5 * (1 - 0) * q_jump
        
        return numerical_flux

