import numpy as np

def get_CFL_step_size(velocity, min_dx, CFL=.1, ):
    """Get the step size for the CFL time integrator."""
    
    C = np.max(velocity)

    return CFL/C*min_dx

