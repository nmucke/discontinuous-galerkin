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

        init = np.ones(x.shape)
        init[x<-0.5] = 2*init[x<-0.5]

        return np.expand_dims(init, 0)
    
    def boundary_conditions(self, t):
        """Compute the boundary conditions."""

        BC_state_1 = {
            'left': 2.0,
            'right': None,
        }

        BCs = [BC_state_1]
        
        return BCs
    
    def wave_speed(self, q):
        """Compute the wave speed."""

        return 2*q
        
    def flux(self, q):
        """Compute the flux."""

        return q*q
    
    def source(self, q):
        """Compute the source."""

        return np.zeros((self.DG_vars.num_states,self.DG_vars.Np*self.DG_vars.K))

if __name__ == '__main__':
    

    xmin = -1.
    xmax = 1

    BC_types = 'dirichlet'

    numerical_flux_type = 'lax_friedrichs'
    numerical_flux_params = {
        'alpha': 0.5,
    }

    stabilizer_type = 'slope_limiter'
    stabilizer_params = {
        'second_derivative_upper_bound': 1e1,
    }
    '''
    stabilizer_type = 'filter'
    stabilizer_params = {
        'num_modes_to_filter': 0,
        'filter_order': 6,
    }
    '''

    time_integrator_type = 'SSPRK'
    time_integrator_params = {
    }

    polynomial_type='legendre'
    num_states=1

    error = []
    conv_list = [2, 4, 8, 16]
    num_DOFs = []
    for polynomial_order in conv_list:

        #polynomial_order=8
        num_elements=25

        num_DOFs.append((polynomial_order+1)*num_elements)

        burgers_DG = BurgersEquation(
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

        init = burgers_DG.initial_condition(burgers_DG.DG_vars.x.flatten('F'))

        true_sol = lambda t: burgers_DG.initial_condition(
            burgers_DG.DG_vars.x.flatten('F') - 3*t
            )
        
        sol, t_vec = burgers_DG.solve(t=0, q_init=init, t_final=0.4)

        true_sol_array = np.zeros(
            (burgers_DG.DG_vars.num_states, 
            burgers_DG.DG_vars.Np * burgers_DG.DG_vars.K, 
            len(t_vec))
            )
        for i in range(len(t_vec)):
            true_sol_array[:, :, i] = true_sol(t_vec[i])
            
        l2_error = np.linalg.norm(sol - true_sol_array) / np.linalg.norm(true_sol_array)
    
        error.append(l2_error)

    plt.figure()
    plt.loglog(num_DOFs, error, '.-', label='error', linewidth=2, markersize=10)
    plt.loglog(num_DOFs, [10**(-i) for i in range(len(num_DOFs))], label='slope 1', linewidth=2)
    plt.loglog(num_DOFs, [10**(-i*2) for i in range(len(num_DOFs))], label='slope 2', linewidth=2)
    plt.loglog(num_DOFs, [10**(-i*3) for i in range(len(num_DOFs))], label='slope 3', linewidth=2)
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(burgers_DG.DG_vars.x.flatten('F'), init[0], label='initial', linewidth=4)
    plt.plot(burgers_DG.DG_vars.x.flatten('F'), true_sol(t_vec[-1])[0], label='true', linewidth=3)
    plt.plot(burgers_DG.DG_vars.x.flatten('F'), sol[0, :, -1], label='final', linestyle='--', linewidth=2)
    plt.grid()
    plt.legend()
    plt.show()

