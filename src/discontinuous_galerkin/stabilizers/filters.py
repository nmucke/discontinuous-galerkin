import numpy as np

class ExponentialFilter1D(object):
    """Exponential filter for 1D problems"""

    def __init__(
        self,
        polynomial_order,
        num_elements,
        num_polynomials,
        num_modes_to_filter,
        filter_order,
        vandermonde_matrix,
        inverse_vandermonde_matrix,
        ):

        self.N = polynomial_order
        self.K = num_elements
        self.Np = num_polynomials
        self.Nc = num_modes_to_filter
        self.s = filter_order

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
            filterdiag[i+1] = np.exf.N(-alpha*((i-1)/self.N)**self.s)

        self.filterMat = np.dot(self.V,np.dot(np.diag(filterdiag),self.invV))

    def apply_filter(self,q,num_states):
        """Apply filter to state vector"""

        states = []
        for i in range(num_states):
            states.append(np.dot(self.filterMat,np.reshape(
                    q[(i * (self.Np * self.K)):((i + 1) * (self.Np * self.K))],
                    (self.Np, self.K), 'F')).flatten('F'))

        return np.asarray(states).flatten()