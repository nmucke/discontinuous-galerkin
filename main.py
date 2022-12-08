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
        polynomial_type='legendre',
        num_states=1,
        stabilizer_type=None, 
        stabilizer_params=None,
        time_stepper='ImplicitEuler',
        numerical_flux_type='lax_friedrichs',
        numerical_flux_params=None,
        ):
        super(AdvectionEquation, self).__init__(
            xmin=xmin,
            xmax=xmax,
            num_elements=num_elements,
            polynomial_order=polynomial_order,
            polynomial_type=polynomial_type,
            stabilizer_type=stabilizer_type, 
            stabilizer_params=stabilizer_params,
            time_stepper=time_stepper,
            numerical_flux_type=numerical_flux_type,
            numerical_flux_params=numerical_flux_params,
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
    
    def source(self, q):
        """Compute the source."""

        return 0

if __name__ == '__main__':
    

    xmin=0.
    xmax=1.
    num_elements=10
    polynomial_order=5
    polynomial_type='legendre'
    num_states=1

    stabilizer_type = 'slope_limiter'
    stabilizer_params = {
        'second_derivative_upper_bound': 1e-5,
    }

    lol = AdvectionEquation(
        xmin=xmin,
        xmax=xmax,
        num_elements=num_elements,
        polynomial_order=polynomial_order,
        polynomial_type=polynomial_type,
        num_states=num_states,
        stabilizer_type=stabilizer_type, 
        stabilizer_params=stabilizer_params,
        )

    init = lol.initial_condition(lol.variables.x)
    flux = lol.compute_rhs(init)
    '''
    stabilizer_type = 'filter'
    stabilizer_params = {
        'num_modes_to_filter': 1,
        'filter_order': 32,
    }
    lol = AdvectionEquation(
        stabilizer_type=stabilizer_type, 
        stabilizer_params=stabilizer_params,
        )
    '''

    init = lol.initial_condition(lol.variables.x)
    flux = lol.flux(init)
    

    sol = lol.solve(init)
    sol_ = lol.stabilizer.apply_stabilizer(sol)
    print(sol.flatten('F')- sol_)
    pdb.set_trace()

