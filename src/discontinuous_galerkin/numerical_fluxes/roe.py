import numpy as np
from abc import abstractmethod
from discontinuous_galerkin.base.base_numerical_flux import BaseNumericalFlux
import pdb
from scipy.linalg import eig
from scipy.optimize import fsolve

class RoeFlux(BaseNumericalFlux):
    """
    Roe numerical flux class.

    This class contains the functionality for the Roe numerical flux.
    """
    def __init__(
        self, 
        DG_vars, 
        eigen=None, 
        system_jacobian=None,
        flux=None,
        BC_equations=None,
        ):
        """Initialize the class."""

        super(RoeFlux).__init__()

        self.L = 10000 # meters
        self.d = 0.146 # meters
        self.A = np.pi*self.d**2/4 # meters^2
        self.c = 308. # m/s
        self.rho_g_norm = 1.26 # kg/m^3
        self.rho_l = 1003. # kg/m^3
        self.p_amb = 101325.
        self.p_norm = 1.0e5 # Pa
        self.p_outlet = 1.0e6 # Pa
        self.e = 1e-8 # meters
        self.mu_g = 1.8e-5 # Pa*s
        self.mu_l = 1.516e-3 # Pa*s
        self.Cd = 5e-4
        self.T_norm = 278 # Kelvin
        self.T = 278 # Kelvin

        self.rho_l = 1000. # kg/m^3





        self.DG_vars = DG_vars

        self.eigen = eigen
        self.system_jacobian = system_jacobian
        self.flux = flux
        self.BC_equations = BC_equations

        self.nx_boundary = np.array(
            [self.DG_vars.nx[self.DG_vars.mapI], self.DG_vars.nx[self.DG_vars.mapO]]
        )
    
    def density_to_pressure(self, rho):
        """Compute the pressure from the density."""

        return self.p_norm * rho * self.T / self.rho_g_norm / self.T_norm


    def pressure_to_density(self, p):
        """Compute the density from the pressure."""

        return self.rho_g_norm * self.T_norm / self.p_norm * p / self.T
    

    def conservative_to_primitive(self, q):
        """Compute the primitive variables from the conservative variables."""

        A_l = q[1]/self.rho_l
        A_g = self.A - A_l

        rho_g = q[0]/A_g

        p = self.density_to_pressure(rho_g)

        rho_m = rho_g * A_g + self.rho_l * A_l

        u_m = q[2]/rho_m

        return A_l, p, u_m
    
    def primitive_to_conservative(self, q):
        """Compute the conservative variables from the primitive variables."""

        A_l = q[0]
        p = q[1]
        u_m = q[2]

        rho_g = self.pressure_to_density(p)

        A_g = self.A - A_l

        rho_m = rho_g * A_g + self.rho_l * A_l

        rho_g_A_g = rho_g * A_g
        rho_l_A_l = self.rho_l * A_l
        rho_m_u_m = rho_m * u_m

        return rho_g_A_g, rho_l_A_l, rho_m_u_m
    
    def _average_operator(self, q_inside, q_outside):
        """Compute the average operator."""

        return (q_inside + q_outside) / 2
    
    def _compute_eigen_from_system_jacobian(self, q):
        """Compute the eigenvalues of the system."""
        A = self.system_jacobian(q)
        d, l, r = self.eigen = eig(A, left=True, right=True)
        
        return d, l, r


    def _get_eigen(self, q_avg):
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
        
        return D, L, R

    def _get_RDL_q_prod(self, q_diff, D, L, R):
        
        RDL = R @ np.abs(D) @ L
        RDL_q_prod = np.matmul(RDL, q_diff.T[:, :, None]).squeeze(-1).T

        return RDL_q_prod
    
    def _sum_over_matrix_col_product(self, mat, vec, columns):

        out = np.zeros(self.DG_vars.num_states)
        for col in columns:
            out += mat[:, col]*vec
            
        return out

    def compute_numerical_flux(
        self, 
        q_inside, 
        q_outside, 
        flux_inside, 
        flux_outside,
        on_boundary=False,
        t = None,
        on_interface=None,
        ):
        """Compute the numerical flux."""


        if on_boundary:
            D, L, R = self._get_eigen(q_inside)

            # find indices of negative eigenvalues
            idx_neg = np.where(np.diag(D[0]) < 0)[0]

            # find indices of positive eigenvalues
            idx_pos = np.where(np.diag(D[0]) > 0)[0]


            # left side
            left_BC_equations = lambda q: self.BC_equations(q, side='left')
            q_BC = fsolve(left_BC_equations, q_inside[:, 0])


            q_BC = np.stack(q_BC, axis=0)


            func = lambda q_ghost: self._sum_over_matrix_col_product(
                mat=R[0], vec=q_BC - q_ghost, columns=idx_neg
            )
            q_outside_left = fsolve(func, q_inside[:, 0])

            q_outside_left = self.conservative_to_primitive(q_outside_left)
            q_outside_left = np.stack(q_outside_left, axis=0)

            # right side
            right_BC_equations = lambda q: self.BC_equations(q, side='right')
            q_BC = fsolve(right_BC_equations, q_inside[:, -1])


            q_BC = self.primitive_to_conservative(q_BC)
            q_BC = np.stack(q_BC, axis=0)



            func = lambda q_ghost: self._sum_over_matrix_col_product(
                mat=R[1], vec=q_BC - q_ghost, columns=idx_pos
            )
            q_outside_right = fsolve(func, q_inside[:, -1])


            q_outside_right = self.conservative_to_primitive(q_outside_right)
            q_outside_right = np.stack(q_outside_right, axis=0)


            q_outside = np.hstack((q_outside_left[:, None], q_outside_right[:, None]))

            flux_outside = self.flux(q_outside)

            # Compute the average of the fluxes
            flux_average = self._average_operator(flux_inside, flux_outside)

        
            # Compute the jump of the states
            q_roe_average = self._average_operator(q_inside, q_outside)

            # Compute the difference of the states

            # Compute the eigenvalues of the system
            D, L, R = self._get_eigen(q_roe_average)

            q_inside = np.stack(q_inside, axis=0)

            q_outside = np.stack(q_outside, axis=0)

            q_diff = (q_outside - q_inside)/2


            # Compute the product of RDL and q_diff
            RDL_q_prod = self._get_RDL_q_prod(q_diff, D, L, R)



            # Compute the numerical flux
            numerical_flux = flux_average - self.nx_boundary*RDL_q_prod

        else:
            
            # Compute the average of the fluxes
            flux_average = self._average_operator(flux_inside, flux_outside)
        
            # Compute the jump of the states
            q_roe_average = self._average_operator(q_inside, q_outside)


            # Compute the eigenvalues of the system
            D, L, R = self._get_eigen(q_roe_average)

            q_inside = np.stack(q_inside, axis=0)

            q_outside = np.stack(q_outside, axis=0)

            q_diff = (q_outside - q_inside)/2

            # Compute the product of RDL and q_diff
            RDL_q_prod = self._get_RDL_q_prod(q_diff, D, L, R)

            # Compute the numerical flux
            numerical_flux = flux_average - self.DG_vars.nx*RDL_q_prod

        
        return numerical_flux

