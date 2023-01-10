from abc import abstractmethod

class BaseTimeIntegrator():
    """Base class for stabilizers"""

    def __init__(self, ):
        """Initialize base stabilizer class"""
        
    @abstractmethod
    def time_step(self, **kwargs):
        """Apply stabilizer to state vector"""

        raise NotImplementedError
    
    def __call__(self, **kwargs):
        return self.time_step(**kwargs)


