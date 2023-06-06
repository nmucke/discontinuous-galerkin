import numpy as np
from abc import abstractmethod
from discontinuous_galerkin.base.base_numerical_flux import BaseNumericalFlux
import pdb

class LaxFriedrichsFlux(BaseNumericalFlux):
    """
    Lax-Friedrichs numerical flux class.
    
    This class contains the functionality for the Lax-Friedrichs numerical flux. 
    """
    def __init__(self, DG_vars, alpha=0.5, C=lambda q: np.abs(q), **kwargs):
        """Initialize the class."""

        super(LaxFriedrichsFlux).__init__()

        self.DG_vars = DG_vars
        self.alpha = alpha
        self.C = C

        self.nx_boundary = np.array(
            [self.DG_vars.nx[self.DG_vars.mapI], self.DG_vars.nx[self.DG_vars.mapO]]
        )
    
    def _average_operator(self, q_inside, q_outside):
        """Compute the average operator."""

        return (q_inside + q_outside) / 2

    def _jump_operator(self, q_inside, q_outside):
        """Compute the jump operator."""

        return self.DG_vars.nx * (q_inside - q_outside)
    
    def _boundary_jump_operator(self, q_inside, q_outside):
        """Compute the jump operator on the boundary."""
        
        return self.nx_boundary * (q_inside - q_outside)
    
    def _interface_jump_operator(self, q_inside, q_outside, side='left'):
        """Compute the jump operator on the boundary."""

        if side == 'left':
            return -1 *(q_inside - q_outside)
        elif side == 'right':
            return 1 * (q_inside - q_outside)


    def compute_numerical_flux(
        self, 
        q_inside, 
        q_outside, 
        flux_inside, 
        flux_outside,
        on_boundary=False,
        on_interface=None,
        ):
        """Compute the numerical flux."""

        # Compute the average of the fluxes
        flux_average = self._average_operator(flux_inside, flux_outside)

        # Compute the jump of the states
        if on_boundary:
            q_jump = self._boundary_jump_operator(q_inside, q_outside)
        elif on_interface is not None:
            q_jump = self._interface_jump_operator(q_inside, q_outside, side=on_interface)
        else:
            q_jump = self._jump_operator(q_inside, q_outside)

        # Compute the velocity
        C_inside = self.C(q_inside)
        C_outside = self.C(q_outside)
        C = np.maximum(np.abs(C_inside), np.abs(C_outside))

        # Compute the numerical flux
        numerical_flux = flux_average + C * 0.5 * (1 - self.alpha) * q_jump
        
        return numerical_flux

