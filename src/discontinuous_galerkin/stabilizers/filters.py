import numpy as np
from discontinuous_galerkin.stabilizers.base_stabilizer import BaseStabilizer
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

    def __init__(self, variables, num_modes_to_filter=5, filter_order=32,):
        """Initialize exponential filter"""

        super(ExponentialFilter, self).__init__()

        self.variables = variables

        self.Nc = num_modes_to_filter
        self.s = filter_order

        self.filterMat = self._Filter1D()


    def _Filter1D(self):
        """Initialize 1D filter matrix of size N.
            Order of exponential filter is (even) s with cutoff at Nc;"""

        filterdiag = np.ones((self.variables.Np))

        alpha = -np.log(np.finfo(float).eps)

        for i in range(self.Nc, self.variables.N):
            #filterdiag[i+1] = np.ef.Np(-alpha*((i-Nc)/(N-Nc))**s)
            filterdiag[i+1] = np.exp(-alpha*((i-1)/self.variables.N)**self.s)
            

        return np.dot(self.variables.V,np.dot(np.diag(filterdiag),self.variables.invV))

    def apply_stabilizer(self, q):
        """Apply filter to state vector"""

        states = []
        for i in range(self.variables.num_states):
            states.append(np.dot(self.filterMat,np.reshape(
                    q[(i * (self.variables.Np * self.variables.K)):((i + 1) * (self.variables.Np * self.variables.K))],
                    (self.variables.Np, self.variables.K), 'F')).flatten('F'))

        return np.asarray(states).flatten()