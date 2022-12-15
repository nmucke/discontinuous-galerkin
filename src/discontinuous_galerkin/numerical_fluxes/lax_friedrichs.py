import numpy as np
from abc import abstractmethod
from discontinuous_galerkin.base.base_numerical_flux import BaseNumericalFlux
import pdb

class LaxFriedrichsFlux(BaseNumericalFlux):
    """
    Lax-Friedrichs numerical flux class.
    
    This class contains the functionality for the Lax-Friedrichs numerical flux. 
    It is not intended to be used directly, but rather as a base class for other
    numerical fluxes.
    """
    def __init__(self, DG_vars, alpha=0.5):
        """Initialize the class."""

        super(LaxFriedrichsFlux).__init__()

        self.DG_vars = DG_vars
        self.alpha = alpha
    
    def average_operator(self, q_inside, q_outside):
        """Compute the average operator."""

        return (q_inside + q_outside) / 2

    def jump_operator(self, q_inside, q_outside):
        """Compute the jump operator."""

        return self.DG_vars.nx * (q_inside - q_outside)

    def compute_numerical_flux(self, q_inside, q_outside, flux_inside, flux_outside):
        """Compute the numerical flux."""

        # Compute the average of the fluxes
        flux_average = self.average_operator(flux_inside, flux_outside)

        # Compute the jump of the states
        q_jump = self.jump_operator(q_inside, q_outside)

        # Compute the velocity
        C = 2*np.pi

        # Compute the numerical flux
        numerical_flux = flux_average + C * 0.5 * (1 - self.alpha) * q_jump
        
        return numerical_flux

