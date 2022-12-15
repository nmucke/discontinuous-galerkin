import numpy as np
from discontinuous_galerkin.base.base_model import BaseModel
from discontinuous_galerkin.start_up_routines.start_up_1D import StartUp1D
import matplotlib.pyplot as plt
import pdb

class BurgersEquation(BaseModel):
    """Advection equation model class."""

    def __init__(
        self, 
        xmin=0.,
        xmax=1.,
        num_elements=10,
        polynomial_order=5,
        polynomial_type='legendre',
        num_states=1,
        BC_types='dirichlet',
        stabilizer_type=None, 
        stabilizer_params=None,
        time_integrator_type='implicit_euler',
        time_integrator_params=None,
        numerical_flux_type='lax_friedrichs',
        numerical_flux_params=None,
        ):
        super().__init__(
            xmin=xmin,
            xmax=xmax,
            num_elements=num_elements,
            polynomial_order=polynomial_order,
            polynomial_type=polynomial_type,
            num_states=num_states,
            BC_types=BC_types,
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
            'left': None,
            'right': None,
        }

        BCs = [BC_state_1]
        
        return BCs
        
    def flux(self, q):
        """Compute the flux."""

        return q*q
    
    def source(self, q):
        """Compute the source."""

        return np.zeros((self.DG_vars.num_states,self.DG_vars.Np*self.DG_vars.K))

if __name__ == '__main__':
    

    xmin=0.
    xmax=2.*np.pi

    BC_types = 'dirichlet'

    numerical_flux_type = 'lax_friedrichs'
    numerical_flux_params = {
        'alpha': .5,
    }

    '''
    stabilizer_type = 'slope_limiter'
    stabilizer_params = {
        'second_derivative_upper_bound': 1e-1,
    }
    '''
    stabilizer_type = None
    stabilizer_params = {
        'num_modes_to_filter': 1,
        'filter_order': 32,
    }

    time_integrator_type = 'SSPRK'
    time_integrator_params = {
        'step_size': 1e-3
    }

    polynomial_type='legendre'
    num_states=1

    true_sol = lambda t: np.sin(advection_DG.DG_vars.x.flatten('F') - 2*np.pi*t)

    error = []
    conv_list = [1, 2, 4, 8, 16]
    num_DOFs = []
    for polynomial_order in conv_list:

        #polynomial_order=15
        num_elements=5

        num_DOFs.append((polynomial_order+1)*num_elements)

        advection_DG = AdvectionEquation(
            xmin=xmin,
            xmax=xmax,
            num_elements=num_elements,
            polynomial_order=polynomial_order,
            polynomial_type=polynomial_type,
            num_states=num_states,
            BC_types=BC_types,
            stabilizer_type=stabilizer_type, 
            stabilizer_params=stabilizer_params,
            time_integrator_type=time_integrator_type,
            time_integrator_params=time_integrator_params, 
            numerical_flux_type=numerical_flux_type,
            numerical_flux_params=numerical_flux_params,
            )
        

        init = advection_DG.initial_condition(advection_DG.DG_vars.x.flatten('F'))
        #flux = lol.compute_rhs(t=0, q=init)
        
        sol, t_vec = advection_DG.solve(t=0, q_init=init, t_final=0.1)

        true_sol_array = np.zeros(
            (advection_DG.DG_vars.num_states, 
            advection_DG.DG_vars.Np * advection_DG.DG_vars.K, 
            len(t_vec))
            )
        for i in range(len(t_vec)):
            true_sol_array[:, :, i] = true_sol(t_vec[i])
            
        l2_error = np.linalg.norm(sol - true_sol_array) / np.linalg.norm(true_sol_array)
    
        error.append(l2_error)

    plt.figure()
    plt.loglog(num_DOFs, error, '.-', label='error', linewidth=2, markersize=10)
    plt.loglog(num_DOFs, [1e-1, 1e-2, 1e-3, 1e-4, 1e-5], label='slope 1', linewidth=2)
    plt.loglog(num_DOFs, [1e-2, 1e-4, 1e-6, 1e-8, 1e-10], label='slope 2', linewidth=2)
    plt.loglog(num_DOFs, [1e-3, 1e-6, 1e-9, 1e-12, 1e-15], label='slope 3', linewidth=2)
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(advection_DG.DG_vars.x.flatten('F'), init[0], label='initial', linewidth=4)
    plt.plot(advection_DG.DG_vars.x.flatten('F'), true_sol(t_vec[-1]), label='true', linewidth=3)
    plt.plot(advection_DG.DG_vars.x.flatten('F'), sol[0, :, -1], label='final', linestyle='--', linewidth=2)
    plt.grid()
    plt.legend()
    plt.show()

