import numpy as np
from abc import abstractmethod


class BaseNumericalFlux():
    """
    Base numerical flux class.
    
    This class contains the functionality for the base numerical flux.    
    """

    def __init__(self, ):
        """Initialize the class."""

        pass
    
    @abstractmethod
    def compute_numerical_flux(self, q_minus, q_plus):
        """Compute the numerical flux."""
        
        raise NotImplementedError


    def __call__(self, **kwargs):
        return self.compute_numerical_flux(**kwargs)


