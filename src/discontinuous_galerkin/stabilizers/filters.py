import numpy as np
from discontinuous_galerkin.stabilizers.base_stabilizer import BaseStabilizer
import pdb

class ExponentialFilter1D(BaseStabilizer):
    """Exponential filter for 1D problems
    
    This class implements an exponential filter for 1D problems. The filter is
    used to filter out high frequency modes in the solution. The filter is
    applied to the state vector, which is a vector of the solution at all
    polynomial nodes in all elements. The filter is applied to each state
    variable separately. The filter is applied to the state vector in the
    following way:

    qf = V*diag(filterdiag)*invV*q
    """

    def __init__(
        self,
        x,
        polynomial_order,
        num_elements,
        num_states,
        num_modes_to_filter,
        filter_order,
        vandermonde_matrix,
        inverse_vandermonde_matrix,
        ):
        super(ExponentialFilter1D, self).__init__(
            polynomial_order=polynomial_order,
            num_elements=num_elements,
            num_polynomials=polynomial_order+1,
            vandermonde_matrix=vandermonde_matrix,
            inverse_vandermonde_matrix=inverse_vandermonde_matrix,
            num_states=num_states,
            x=x,
        )

        self.N = polynomial_order
        self.K = num_elements
        self.Np = polynomial_order + 1
        self.Nc = num_modes_to_filter
        self.s = filter_order

        self.num_states = num_states

        self.V = vandermonde_matrix
        self.invV = inverse_vandermonde_matrix

        self.filterMat = self.Filter1D()


    def Filter1D(self):
        """Initialize 1D filter matrix of size N.
            Order of exponential filter is (even) s with cutoff at Nc;"""

        filterdiag = np.ones((self.Np))

        alpha = -np.log(np.finfo(float).eps)

        for i in range(self.Nc, self.N):
            #filterdiag[i+1] = np.ef.Np(-alpha*((i-Nc)/(N-Nc))**s)
            filterdiag[i+1] = np.exp(-alpha*((i-1)/self.N)**self.s)
            

        return np.dot(self.V,np.dot(np.diag(filterdiag),self.invV))

    def apply_stabilizer(self, q):
        """Apply filter to state vector"""

        states = []
        for i in range(self.num_states):
            states.append(np.dot(self.filterMat,np.reshape(
                    q[(i * (self.Np * self.K)):((i + 1) * (self.Np * self.K))],
                    (self.Np, self.K), 'F')).flatten('F'))

        return np.asarray(states).flatten()