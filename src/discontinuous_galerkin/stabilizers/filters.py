import numpy as np
from discontinuous_galerkin.base.base_stabilizer import BaseStabilizer
import pdb

class ExponentialFilter(BaseStabilizer):
    """Exponential filter for 1D problems
    
    This class implements an exponential filter for 1D problems. The filter is
    used to filter out high frequency modes in the solution. The filter is
    applied to the state vector, which is a vector of the solution at all
    polynomial nodes in all elements. The filter is applied to each state
    variable separately. The filter is applied to the state vector in the
    following way:

    qf = V*diag(filterdiag)*invV*q
    """

    def __init__(self, DG_vars, num_modes_to_filter=5, filter_order=32,):
        """Initialize exponential filter"""

        super(ExponentialFilter, self).__init__()

        self.DG_vars = DG_vars

        self.Nc = num_modes_to_filter
        self.s = filter_order

        self.filterMat = self._Filter1D()


    def _Filter1D(self):
        """Initialize 1D filter matrix of size N.
            Order of exponential filter is (even) s with cutoff at Nc;"""

        filterdiag = np.ones((self.DG_vars.Np))

        alpha = -np.log(np.finfo(float).eps)

        for i in range(self.Nc, self.DG_vars.N):
            #filterdiag[i+1] = np.ef.Np(-alpha*((i-Nc)/(N-Nc))**s)
            filterdiag[i+1] = np.exp(-alpha*((i-1)/self.DG_vars.N)**self.s)
            

        return np.dot(self.DG_vars.V,np.dot(np.diag(filterdiag),self.DG_vars.invV))

    def apply_stabilizer(self, q):
        """Apply filter to state vector"""

        states = []
        for i in range(self.DG_vars.num_states):
            filtered_state = np.dot(self.filterMat, np.reshape(
                q[i],
                (self.DG_vars.Np, self.DG_vars.K),
                'F')
                ).flatten('F')
            
            states.append(filtered_state)
            
        return np.asarray(states)