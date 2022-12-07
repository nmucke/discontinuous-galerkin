import numpy as np
from discontinuous_galerkin.start_up_routines.start_up_1D import StartUp1D


class BaseModel(StartUp1D):
    """Base class for all models.

    This class contains the basic functionality for all models. It is not
    intended to be used directly, but rather as a base class for other models.
    """

    def __init__(self, ):
        super(StartUp1D, self).__init__()

    def __str__(self):
        return "BaseModel"

    def __repr__(self):
        return self.__str__()

    def solve(self, ):
        """Solve the model.

        This method solves the model and returns the solution.
        """
        pass
