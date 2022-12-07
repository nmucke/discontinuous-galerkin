import numpy as np
from discontinuous_galerkin.start_up_routines.start_up_1D import StartUp1D
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
        K=10,
        N=5,
        poly_type='legendre',
        stabilizer=None, 
        time_stepper='ImplicitEuler'
        ):
        super(BaseModel, self).__init__(
            xmin=xmin,
            xmax=xmax,
            K=K,
            N=N,
            poly=poly_type
        )
        self.xmin = xmin
        self.xmax = xmax
        self.K = K
        self.N = N
        self.poly_type = poly_type
        self.Np = N + 1

        self.stabilizer = stabilizer
        self.time_stepper = time_stepper

    def __str__(self):
        """ Description of the model. """

        output = "BaseModel: \n"
        output += f"xmin: {self.xmin} \n"
        output += f"xmax: {self.xmax} \n"
        output += f"Number of elements: {self.K} \n"
        output += f"Polynomial order: {self.N} \n"
        output += f"Polynomial type: {self.poly_type} \n"
        output += f"Stabilizer: {self.stabilizer} \n"
        output += f"Time stepping: {self.time_stepper} \n"
        
        return output

    def __repr__(self):
        return self.__str__()

    def solve(self, ):
        """Solve the model.

        This method solves the model and returns the solution.
        """
        pass
