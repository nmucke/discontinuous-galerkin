import numpy as np
from abc import abstractmethod
from discontinuous_galerkin.base.base_numerical_flux import BaseNumericalFlux
import pdb

class LaxFriedrichsFlux(BaseNumericalFlux):
    """
    Lax-Friedrichs numerical flux class.
    
    This class contains the functionality for the Lax-Friedrichs numerical flux. 
    """
    def __init__(self, DG_vars, alpha=0.5, velocity=lambda q: np.abs(q)):
        """Initialize the class."""

        super(LaxFriedrichsFlux).__init__()

        self.DG_vars = DG_vars
        self.alpha = alpha
        self.velocity = velocity

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

    def compute_numerical_flux(
        self, 
        q_inside, 
        q_outside, 
        flux_inside, 
        flux_outside,
        on_boundary=False,
        primitive_to_conservative=None,
        BC_state_or_flux=None
        ):
        """Compute the numerical flux."""

        if primitive_to_conservative is not None:
            q_inside = primitive_to_conservative(q_inside)
            q_outside = primitive_to_conservative(q_outside)

        # Compute the velocity
        C_inside = self.velocity(q_inside)
        C_outside = self.velocity(q_outside)
        C = np.maximum(np.abs(C_inside), np.abs(C_outside))

        # Compute the average of the fluxes
        flux_average = self._average_operator(flux_inside, flux_outside)

        if on_boundary:

            numerical_flux = flux_average
            for i, side in enumerate(['left', 'right']):
                if BC_state_or_flux[side] == 'flux':
                    numerical_flux[:, i] = flux_outside[:, i]
                else:
                    q_jump = self._boundary_jump_operator(q_inside, q_outside)
                    numerical_flux[:, i] += C[i] * 0.5 * (1 - self.alpha) * q_jump[:, i]
        else:
            q_jump = self._jump_operator(q_inside, q_outside)

            # Compute the numerical flux
            numerical_flux = flux_average + C * 0.5 * (1 - self.alpha) * q_jump
        
        return numerical_flux

