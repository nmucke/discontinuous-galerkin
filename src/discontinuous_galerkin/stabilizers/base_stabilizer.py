from discontinuous_galerkin.stabilizers.slope_limiters import GeneralizedSlopeLimiter
from discontinuous_galerkin.stabilizers.filters import ExponentialFilter1D



class BaseStabilizer(object):
    """Base class for stabilizers"""

    def __init__(
        self, 
        polynomial_order,
        num_elements,
        delta_x,
        num_polynomials,
        vandermonde_matrix,
        inverse_vandermonde_matrix,
        ):

        self.N = polynomial_order
        self.K = num_elements
        self.Np = num_polynomials
        self.V = vandermonde_matrix
        self.invV = inverse_vandermonde_matrix
        self.delta_x = delta_x
    
    def apply_stabilizer(self, q, num_states):
        """Apply stabilizer to state vector"""

        raise NotImplementedError