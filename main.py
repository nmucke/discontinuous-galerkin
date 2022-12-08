import numpy as np
from discontinuous_galerkin.base_model import BaseModel
from discontinuous_galerkin.start_up_routines.start_up_1D import StartUp1D
import matplotlib.pyplot as plt
import pdb

class AdvectionEquation(BaseModel):
    """Advection equation model class."""

    def __init__(
        self, 
        xmin=0.,
        xmax=1.,
        num_elements=10,
        polynomial_order=5,
        poly_type='legendre',
        num_states=1,
        stabilizer_type=None, 
        stabilizer_params=None,
        time_stepper='ImplicitEuler',
        ):
        super(AdvectionEquation, self).__init__(
            xmin=xmin,
            xmax=xmax,
            num_elements=num_elements,
            polynomial_order=polynomial_order,
            poly_type=poly_type,
            stabilizer_type=stabilizer_type, 
            stabilizer_params=stabilizer_params,
            time_stepper=time_stepper,
            )
    
    def initial_condition(self, x):
        """Compute the initial condition."""

        return np.sin(x)
    
    def boundary_conditions(self, q):
        """Compute the boundary conditions."""

        return q
        
    def flux(self, q):
        """Compute the flux."""

        return q

if __name__ == '__main__':
    


    stabilizer_type = 'slope_limiter'
    stabilizer_params = {
        'second_derivative_upper_bound': 1e-5,
    }


    stabilizer_type = 'filter'
    stabilizer_params = {
        'num_modes_to_filter': 1,
        'filter_order': 32,
    }

    lol = AdvectionEquation(
        stabilizer_type=stabilizer_type, 
        stabilizer_params=stabilizer_params,
        )
    
    init = lol.initial_condition(lol.x)
    flux = lol.flux(init)
    

    sol = lol.solve()

    sol_ = lol.apply_stabilizer(sol)
    print(sol.flatten('F')- sol_)
    pdb.set_trace()

