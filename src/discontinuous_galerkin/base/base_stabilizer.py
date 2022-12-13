from abc import abstractmethod
import pdb

class BaseStabilizer():
    """Base class for stabilizers"""

    def __init__(self, ):
        """Initialize base stabilizer class"""
        
    @abstractmethod
    def apply_stabilizer(self, q):
        """Apply stabilizer to state vector"""

        raise NotImplementedError

    def __call__(self, **kwargs):
        return self.apply_stabilizer(**kwargs)
