import numpy as np
from discontinuous_galerkin.start_up_routines.start_up_1D import StartUp1D
from discontinuous_galerkin.stabilizers.slope_limiters import GeneralizedSlopeLimiter
from discontinuous_galerkin.stabilizers.filters import ExponentialFilter1D
from discontinuous_galerkin.numerical_fluxes.lax_friedrichs import LaxFriedrichsFlux
from abc import abstractmethod

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
        num_states=1,
        stabilizer_type=None, 
        stabilizer_params=None,
        time_stepper='ImplicitEuler',
        numerical_flux='lax_friedrichs',
        ):
        super(BaseModel, self).__init__(
            xmin=xmin,
            xmax=xmax,
            K=num_elements,
            N=polynomial_order,
            poly=poly_type,
        )
        self.xmin = xmin
        self.xmax = xmax
        self.K = num_elements
        self.N = polynomial_order
        self.poly_type = poly_type
        self.Np = self.N + 1
        self.num_states=num_states

        self.stabilizer_type = stabilizer_type

        if stabilizer_params is not None:
            self.stabilizer_params = stabilizer_params
            self.set_up_stabilizer()

        self.time_stepper = time_stepper

        self.numerical_flux = LaxFriedrichsFlux

    def set_up_stabilizer(self):
        """Set up the stabilizer."""

        base_stabilizer_params = {
            'polynomial_order': self.N,
            'num_elements': self.K,
            'vandermonde_matrix': self.V,
            'inverse_vandermonde_matrix': self.invV,
            'num_states': self.num_states,
            'x': self.x,
        }

        if self.stabilizer_type == 'slope_limiter':
            self.stabilizer = GeneralizedSlopeLimiter(
                **base_stabilizer_params,
                **self.stabilizer_params,
                differentiation_matrix=self.Dr,
                delta_x=self.deltax,
            )
        elif self.stabilizer_type == 'filter':
            self.stabilizer = ExponentialFilter1D(
                **base_stabilizer_params,
                **self.stabilizer_params
            )
        
        self.apply_stabilizer = self.stabilizer.apply_stabilizer

    def set_up_numerical_flux(self):
        """Set up the numerical flux."""

        pass

    def set_up_time_stepper(self):
        """Set up the time stepper."""

        pass


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

    @abstractmethod
    def flux(self, q):
        """Compute the flux."""

        raise NotImplementedError

    @abstractmethod
    def initial_condition(self, x):
        """Compute the initial condition."""

        raise NotImplementedError
    
    @abstractmethod
    def boundary_conditions(self, q):
        """Compute the boundary condition."""

        raise NotImplementedError
    
    @abstractmethod
    def source(self, q):
        """Compute the source term."""

        raise NotImplementedError

    @abstractmethod
    def update_parameters(self, **kwargs):

        raise NotImplementedError

    def compute_rhs(self, q):
        """Compute the right hand side of the model."""

        # Compute the flux
        flux = self.flux(q)

        # Compute the source term
        source = self.source(q)

        # Compute the right hand side
        rhs = -self.Dr @ flux + source

        return rhs

    def solve(self, ):
        """Solve the model.

        This method solves the model and returns the solution.
        """

        q = self.flux(self.initial_condition(self.x))

        return q
        

