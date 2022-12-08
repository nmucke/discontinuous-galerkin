import numpy as np
from abc import abstractmethod
from discontinuous_galerkin.numerical_fluxes.base_numerical_flux import BaseNumericalFlux
import pdb

class LaxFriedrichsFlux(BaseNumericalFlux):
    """
    Lax-Friedrichs numerical flux class.
    
    This class contains the functionality for the Lax-Friedrichs numerical flux. 
    It is not intended to be used directly, but rather as a base class for other
    numerical fluxes.
    """
    def __init__(self, variables, alpha=0.5):
        """Initialize the class."""

        super().__init__(variables)

        self.variables = variables
        self.alpha = alpha
    
    def average_operator(self, q_inside, q_outside):
        """Compute the average operator."""

        average = (q_inside + q_outside) / 2

        return average

    def jump_operator(self, q_inside, q_outside):
        """Compute the jump operator."""

        jump = self.variables.nx * (q_outside - q_inside)

        return jump

    def compute_numerical_flux(self, q_inside, q_outside, flux_inside, flux_outside):
        """Compute the numerical flux."""
        
        # Compute the average of the fluxes
        q_average = self.average_operator(flux_inside, flux_outside)

        # Compute the jump of the states
        q_jump = self.jump_operator(q_inside, q_outside)

        # Compute the numerical flux
        numerical_flux = q_average + (1 - self.alpha) / 2 * q_jump

        return numerical_flux

