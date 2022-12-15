import numpy as np
from discontinuous_galerkin.base.base_model import BaseModel
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
        time_integrator_type='implicit_euler',
        time_integrator_params=None,
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
            time_integrator_type=time_integrator_type,
            time_integrator_params=time_integrator_params,
            numerical_flux_type=numerical_flux_type,
            numerical_flux_params=numerical_flux_params,
            )
    
    def initial_condition(self, x):
        """Compute the initial condition."""

        init = np.sin(x)
        init = np.expand_dims(init, 0)

        return init
    
    def boundary_conditions(self, t):
        """Compute the boundary conditions."""

        BC_state_1 = {
            'left': -np.sin(2 * np.pi * t),
            'right': None,
        }

        BCs = [BC_state_1]
        
        return BCs
        
    def flux(self, q):
        """Compute the flux."""

        return 2*np.pi*q
    
    def source(self, q):
        """Compute the source."""

        return np.zeros((self.DG_vars.num_states,self.DG_vars.Np*self.DG_vars.K))

if __name__ == '__main__':
    

    xmin=0.
    xmax=1.
    num_elements=10
    polynomial_order=5
    polynomial_type='legendre'
    num_states=1

    numerical_flux_type = 'lax_friedrichs'
    numerical_flux_params = {
        'alpha': 0.0,
    }

    stabilizer_type = 'slope_limiter'
    stabilizer_params = {
        'second_derivative_upper_bound': 1e-5,
    }

    time_integrator_type = 'SSPRK'
    time_integrator_params = {
        'step_size': 1e-3
    }
    #time_integrator_params = {
    #    'max_newton_iter': 200, 
    #    'newton_tol': 1e-5, 
    #    'step_size': 1e-8
    #}

    lol = AdvectionEquation(
        num_states=num_states,
        xmin=xmin,
        xmax=xmax,
        num_elements=num_elements,
        polynomial_order=polynomial_order,
        polynomial_type=polynomial_type,
        stabilizer_type=stabilizer_type, 
        stabilizer_params=stabilizer_params,
        time_integrator_type=time_integrator_type,
        time_integrator_params=time_integrator_params, 
        numerical_flux_type=numerical_flux_type,
        numerical_flux_params=numerical_flux_params,
        )

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
    true_sol = lambda t: np.sin(lol.DG_vars.x.flatten('F') - 2*np.pi*t)

    init = lol.initial_condition(lol.DG_vars.x.flatten('F'))
    #flux = lol.compute_rhs(t=0, q=init)
    
    t = 0
    sol = init.copy()
    for i in range(10):
        sol = lol.solve(t=t, q=sol)
        t = t + lol.time_integrator.step_size
    plt.figure()
    plt.plot(lol.DG_vars.x.flatten('F'), init[0])
    plt.plot(lol.DG_vars.x.flatten('F'), sol[0])
    plt.plot(lol.DG_vars.x.flatten('F'), true_sol(t))
    plt.show()
    pdb.set_trace()
    sol_ = lol.stabilizer(sol)
    print(sol.flatten('F')- sol_)
    pdb.set_trace()

