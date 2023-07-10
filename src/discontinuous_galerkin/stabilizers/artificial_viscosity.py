import numpy as np
from discontinuous_galerkin.base.base_stabilizer import BaseStabilizer
import pdb
import matplotlib.pyplot as plt

class ArtificialViscosity(BaseStabilizer):
    """ Artificial viscosity for 1D problems

    This class implements an artificial viscosity for 1D problems. The viscosity
    is used to dampen high frequency modes in the solution. The viscosity is
    applied to the state vector, which is a vector of the solution at all
    polynomial nodes in all elements. The viscosity is applied to each state
    variable separately. 
    """

    def __init__(self, DG_vars, kappa=0.1):
        """Initialize exponential filter"""

        super(ArtificialViscosity, self).__init__()

        self.DG_vars = DG_vars
        self.kappa = kappa

        self.epsilon_0 = self.DG_vars.deltax/self.DG_vars.N
        self.s_0 = 1/(self.DG_vars.N**4)


    def _detect_shock(self, q):
        """ Detect shock in solution """

        q = np.reshape(
            q, 
            (self.DG_vars.num_states, self.DG_vars.Np, self.DG_vars.K), 
            order='F'
            )

        q_modal = self.DG_vars.invV @ q

        q_low_order = self.DG_vars.V[:, 0:self.DG_vars.N] @ q_modal[:, 0:self.DG_vars.N, :]

        q_diff = q - q_low_order

        '''
        q_norm = np.zeros((self.DG_vars.num_states, self.DG_vars.K))
        for i in range(self.DG_vars.num_states):
            for j in range(self.DG_vars.K):
                q_norm[i, j] = q[i, :, j].T  @ self.DG_vars.Mk @ q[i, :, j]

        q_norm_diff = np.zeros((self.DG_vars.num_states, self.DG_vars.K))
        for i in range(self.DG_vars.num_states):
            for j in range(self.DG_vars.K):
                q_norm_diff[i, j] = q_diff[i, :, j].T  @ self.DG_vars.Mk @ q_diff[i, :, j]
        '''

        q_norm = np.zeros((self.DG_vars.num_states, self.DG_vars.K))
        for i in range(self.DG_vars.num_states):
            q_norm[i, :] = ((q[i, :].T  @ self.DG_vars.Mk) * q[i, :].T).sum(axis=1)
        

        q_norm_diff = np.zeros((self.DG_vars.num_states, self.DG_vars.K))
        for i in range(self.DG_vars.num_states):
            q_norm_diff[i, :] = ((q_diff[i, :].T  @ self.DG_vars.Mk) * q_diff[i, :].T).sum(axis=1)


        S_e = q_norm_diff/q_norm
        '''
        S_e = np.linalg.norm(q_diff, axis=1)/ np.linalg.norm(q, axis=1)
        '''
        S_e = np.max(S_e, axis=0)
        s_e = np.log10(S_e)

        epsilon_e = np.zeros(self.DG_vars.K)
        indices = np.where(np.logical_and(self.s_0 - self.kappa <= s_e, s_e <= self.s_0 + self.kappa))

        epsilon_e[indices] = self.epsilon_0/2 * (1 + np.sin(np.pi*(s_e[indices] - self.s_0)/2/self.kappa))
        epsilon_e[s_e > self.s_0 +self.kappa] = self.epsilon_0

        return epsilon_e
    
    def get_viscosity(self, q):
        """ Get artificial viscosity """

        epsilon_e = self._detect_shock(q)

        #indices = np.where(epsilon_e > 0)[0]

        return epsilon_e#[indices], indices

    def _compute_artificial_viscosity(self, q):
        """ Compute artificial viscosity """

        return 2

    def apply_stabilizer(self, q):
        """return q"""

        return q