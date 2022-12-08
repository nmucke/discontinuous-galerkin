from abc import abstractmethod

class BaseStabilizer():
    """Base class for stabilizers"""

    def __init__(self, ):
        """Initialize base stabilizer class"""
    
    @abstractmethod
    def apply_stabilizer(self, q):
        """Apply stabilizer to state vector"""

        raise NotImplementedError

