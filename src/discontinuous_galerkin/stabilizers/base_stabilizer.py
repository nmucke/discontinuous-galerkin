from abc import abstractmethod



class BaseStabilizer(object):
    """Base class for stabilizers"""

    def __init__(
        self, 
        x,
        polynomial_order,
        num_elements,
        num_polynomials,
        num_states,
        vandermonde_matrix,
        inverse_vandermonde_matrix,
        ):
        """Initialize base stabilizer class"""

        self.x = x
        self.N = polynomial_order
        self.K = num_elements
        self.Np = num_polynomials
        self.V = vandermonde_matrix
        self.invV = inverse_vandermonde_matrix
        self.num_states = num_states
    
    @abstractmethod
    def apply_stabilizer(self, q):
        """Apply stabilizer to state vector"""

        raise NotImplementedError

