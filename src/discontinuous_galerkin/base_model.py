import numpy as np
from discontinuous_galerkin.start_up_routines.start_up_1D import StartUp1D
from discontinuous_galerkin.stabilizers.slope_limiters import GeneralizedSlopeLimiter
from discontinuous_galerkin.stabilizers.filters import ExponentialFilter1D
import pdb

class BaseModel(StartUp1D):
    """Base class for all models.

    This class contains the basic functionality for all models. It is not
    intended to be used directly, but rather as a base class for other models.
    """

    def __init__(
        self, 
        xmin=0.,
        xmax=1.,
        num_elements=10,
        polynomial_order=5,
        poly_type='legendre',
        stabilizer=None, 
        stabilizer_params=None,
        time_stepper='ImplicitEuler'
        ):
        super(BaseModel, self).__init__(
            xmin=xmin,
            xmax=xmax,
            K=num_elements,
            N=polynomial_order,
            poly=poly_type
        )
        self.xmin = xmin
        self.xmax = xmax
        self.K = num_elements
        self.N = polynomial_order
        self.poly_type = poly_type
        self.Np = N + 1

        if stabilizer_params is not None:
            self.stabilizer_params = stabilizer_params
            self.set_up_stabilizer(stabilizer)

        self.time_stepper = time_stepper

    def set_up_stabilizer(self, stabilizer):

        base_stabilizer_params = {
            'polynomial_order': self.N,
            'num_elements': self.K,
            'delta_x': self.deltax,
            'num_polynomials': self.Np,
            'vandermonde_matrix': self.V,
            'inverse_vandermonde_matrix': self.invV,
        }

        if stabilizer == 'slope_limiter':
            self.stabilizer_type = 'slope_limiter'

            self.stabilizer = GeneralizedSlopeLimiter(
                **base_stabilizer_params,
                **self.stabilizer_params
            )
        elif stabilizer == 'filter':
            self.stabilizer_type = 'filter'

            self.stabilizer = ExponentialFilter1D(
                **base_stabilizer_params,
                **self.stabilizer_params
            )
        
        self.apply_stabilizer = self.stabilizer.apply_stabilizer

    def __str__(self):
        """ Description of the model. """

        output = "BaseModel: \n"
        output += f"xmin: {self.xmin} \n"
        output += f"xmax: {self.xmax} \n"
        output += f"Number of elements: {self.K} \n"
        output += f"Polynomial order: {self.N} \n"
        output += f"Polynomial type: {self.poly_type} \n"
        output += f"Stabilizer: {self.stabilizer_type} \n"
        output += f"Time stepping: {self.time_stepper} \n"
        
        return output

    def __repr__(self):
        return self.__str__()

    def solve(self, ):
        """Solve the model.

        This method solves the model and returns the solution.
        """
        pass
