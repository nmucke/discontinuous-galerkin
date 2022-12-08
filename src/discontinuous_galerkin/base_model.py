import numpy as np
from discontinuous_galerkin.start_up_routines.start_up_1D import StartUp1D
from discontinuous_galerkin.stabilizers.stabilizer import get_stabilizer
from discontinuous_galerkin.numerical_fluxes.numerical_flux import get_numerical_flux
from discontinuous_galerkin.stabilizers.slope_limiters import GeneralizedSlopeLimiter

from abc import abstractmethod


import pdb




#class BaseModel(StartUp1D, Stabilizer, NumericalFlux):
class BaseModel():
    """
    Base class for all models.

    This class contains the basic functionality for all models. It is not
    intended to be used directly, but rather as a base class for other models.
    """

    def __init__(
        self, 
        xmin=0.,
        xmax=1.,
        num_elements=10,
        polynomial_order=5,
        polynomial_type='legendre',
        num_states=1,
        stabilizer_type=None, 
        stabilizer_params=None,
        time_stepper='ImplicitEuler',
        numerical_flux_type='lax_friedrichs',
        numerical_flux_params=None,
        ):
        """Initialize base model class."""        

        # Initialize the start-up routine        
        self.variables = StartUp1D(
            xmin=xmin,
            xmax=xmax,
            num_elements=num_elements,
            polynomial_order=polynomial_order,
            polynomial_type=polynomial_type,
            num_states=num_states,
            )

        # Initialize the stabilizer
        self.stabilizer = get_stabilizer(
            variables=self.variables,
            stabilizer_type=stabilizer_type,
            stabilizer_params=stabilizer_params,
        )

        # Initialize the numerical flux
        self.numerical_flux = get_numerical_flux(
            variables=self.variables,
            numerical_flux_type=numerical_flux_type,
            numerical_flux_params=numerical_flux_params,
        )


    def __str__(self):
        """ Description of the model. """

        output = "BaseModel: \n"
        output += f"xmin: {self.xmin} \n"
        output += f"xmax: {self.xmax} \n"
        output += f"Number of elements: {self.K} \n"
        output += f"Polynomial order: {self.N} \n"
        output += f"Polynomial type: {self.poly_type} \n"
        output += f"Stabilizer: {self.stabilizer_type} \n"
        #output += f"Time stepping: {self.time_stepper} \n"
        
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
        """Compute the right hand side of the discretized model."""

        # Compute the flux
        flux = self.flux(q)

        # Compute the numerical flux
        numerical_flux = self.numerical_flux.compute_numerical_flux(
            q_inside=q.flatten('F')[self.variables.vmapM], 
            q_outside=q.flatten('F')[self.variables.vmapM],
            flux_inside=flux.flatten('F')[self.variables.vmapM],
            flux_outside=flux.flatten('F')[self.variables.vmapM],
            )
        numerical_flux = numerical_flux.reshape(
            self.variables.Nfp * self.variables.Nfaces, 
            self.variables.K, 
            order='F'
            )
            
        # Compute the source term
        source = self.source(q)

        # Compute the right hand side
        rhs = self.variables.Dr @ flux \
            - self.variables.LIFT @ (self.variables.Fscale * numerical_flux) \
            + source 

        return rhs

    def solve(self, t, q):
        """Solve the model.

        This method solves the model and returns the solution.
        """

        # Compute the right hand side
        rhs = self.compute_rhs(q)

        return rhs
        

