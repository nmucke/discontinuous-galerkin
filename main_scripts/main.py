import numpy as np
from discontinuous_galerkin.base.base_model import BaseModel
from discontinuous_galerkin.start_up_routines.start_up_1D import StartUp1D
import matplotlib.pyplot as plt
import pdb

class AdvectionEquation(BaseModel):
    """Advection equation model class."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def initial_condition(self, x):
        """Compute the initial condition."""

        init = np.sin(x)
        init = np.expand_dims(init, 0)

        return init
    
    def boundary_conditions(self, t, q):
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
    
    def velocity(self, q):
        return 2*np.pi

    def source(self, t, q):
        """Compute the source."""

        return np.zeros((self.DG_vars.num_states,self.DG_vars.Np*self.DG_vars.K))

if __name__ == '__main__':
    

    true_sol = lambda t: np.sin(advection_DG.DG_vars.x.flatten('F') - 2*np.pi*t)

    error = []
    conv_list = [1,2,3,4,5]
    num_DOFs = []
    for polynomial_order in conv_list:

        basic_args = {
            'xmin': 0,
            'xmax': 2.*np.pi,
            'num_elements': 50,
            'num_states': 1,
            'polynomial_order': polynomial_order,
            'polynomial_type': 'legendre',
        }

        BC_args = {
            'type': 'dirichlet',
            'treatment': 'naive'
        }

        numerical_flux_params = {
            'type': 'lax_friedrichs',
            'alpha': .0,
        }

        '''
        stabilizer_type = 'slope_limiter'
        stabilizer_params = {
            'second_derivative_upper_bound': 1e-1,
        }
        '''
        stabilizer_params = {
            'type': None,#'filter',
            'num_modes_to_filter': 10,
            'filter_order': 32,
        }

        time_integrator_params = {
            'type': 'SSPRK',
            'step_size': 0.0001,
            'newton_params':{
                'solver': 'krylov',
                'max_newton_iter': 200,
                'newton_tol': 1e-5
                }
        }


        #polynomial_order=15
        num_elements=50

        num_DOFs.append((polynomial_order+1)*num_elements)

        advection_DG = AdvectionEquation(
            basic_args=basic_args,
            BC_args=BC_args,
            stabilizer_args=stabilizer_params,
            time_integrator_args=time_integrator_params,
            numerical_flux_args=numerical_flux_params,
            )
        

        init = advection_DG.initial_condition(advection_DG.DG_vars.x.flatten('F'))
        #flux = lol.compute_rhs(t=0, q=init)
        
        sol, t_vec = advection_DG.solve(t=0, q_init=init, t_final=7.4)

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

