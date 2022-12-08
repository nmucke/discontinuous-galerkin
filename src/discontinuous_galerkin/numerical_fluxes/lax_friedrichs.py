import numpy as np
from abc import abstractmethod
from discontinuous_galerkin.numerical_fluxes.base_numerical_flux import BaseNumericalFlux

class LaxFriedrichsFlux(BaseNumericalFlux):
    """
    Lax-Friedrichs numerical flux class.
    
    This class contains the functionality for the Lax-Friedrichs numerical flux.    
    """

    def __init__(self, alpha=0.5, nx=None):
        """Initialize the class."""
        self.alpha = alpha
        self.nx = nx
    
    @abstractmethod
    def flux(self, q):
        """Compute the flux."""

        raise NotImplementedError

    
    def compute_numerical_flux(self, q_left, q_right):
        """Compute the numerical flux."""

        flux_left = self.flux(q_left)
        flux_right = self.flux(q_right)
        numerical_flux = 0.5 * (flux_left + flux_right) \
            + 0.5 * (1 - self.alpha) * (q_right - q_left)

        return numerical_flux

