import numpy as np
from abc import abstractmethod


class BaseBoundaryConditions():
    """
    Base numerical flux class.
    
    This class contains the functionality for the base numerical flux.    
    """

    def __init__(self,):
        """Initialize the class."""

        pass
    
    @abstractmethod
    def _compute_ghost_states(self, ):
        """Compute the numerical flux."""
        
        raise NotImplementedError

    @abstractmethod
    def apply_boundary_conditions(self, q):
        """Apply the boundary conditions."""
        
        raise NotImplementedError

    def __call__(self, **kwargs):
        return self.apply_boundary_conditions(**kwargs)