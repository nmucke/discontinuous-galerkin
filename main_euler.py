import numpy as np
from discontinuous_galerkin.base.base_model import BaseModel
from discontinuous_galerkin.start_up_routines.start_up_1D import StartUp1D
import matplotlib.pyplot as plt
import pdb

class EulersEquations(BaseModel):
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

        self.gamma = 1.4
    
    def initial_condition(self, x):
        """Compute the initial condition."""
        init = np.zeros((self.DG_vars.num_states, x.shape[0]))

        init[0, x<0.5] = 1.0
        init[0, x>=0.5] = 0.125

        init[1] = 0.0

        init[2, x<0.5] =  1.
        init[2, x>=0.5] = 0.1
        init[2] *= 1.0/(self.gamma-1)

        return init
    
    def boundary_conditions(self, t):
        """Compute the boundary conditions."""

        gamma = 1.4
        p_in = 1.0
        p_out = 0.1

        BC_state_1 = {
            'left': 1.0,
            'right': 0.125,
        }
        BC_state_2 = {
            'left': 0.0,
            'right': 0.0,
        }
        BC_state_3 = {
            'left': p_in/(gamma-1),
            'right': p_out/(gamma-1),
        }

        BCs = [BC_state_1, BC_state_2, BC_state_3]
        
        return BCs
    
    def wave_speed(self, q):
        """Compute the wave speed."""

        p = (self.gamma-1)*(q[2] - 0.5*q[1]*q[1]/q[0])

        c = np.sqrt(self.gamma*p/q[0])

        c += abs(q[1]/q[0])

        return c
        
    def flux(self, q):
        """Compute the flux."""

        u = q[1]/q[0]

        p = (self.gamma-1)*(q[2] - 0.5*q[0]*u*u)

        flux = np.zeros((self.DG_vars.num_states, q.shape[1]))

        flux[0] = q[1]
        flux[1] = q[0] * u * u + p
        flux[2] = (q[2] + p) * u

        return flux

    
    def source(self, q):
        """Compute the source."""

        return np.zeros((self.DG_vars.num_states,self.DG_vars.Np*self.DG_vars.K))

if __name__ == '__main__':
    

    xmin = 0.
    xmax = 1

    BC_types = 'dirichlet'

    numerical_flux_type = 'lax_friedrichs'
    numerical_flux_params = {
        'alpha': 0.0,
    }

    '''
    stabilizer_type = 'slope_limiter'
    stabilizer_params = {
        'second_derivative_upper_bound': 1e-8,
    }
    '''
    stabilizer_type = 'filter'
    stabilizer_params = {
        'num_modes_to_filter': 0,
        'filter_order': 6,
    }

    time_integrator_type = 'SSPRK'
    time_integrator_params = {
    }

    polynomial_type='legendre'
    num_states=3

    error = []
    conv_list = [5]
    num_DOFs = []
    for polynomial_order in conv_list:

        #polynomial_order=8
        num_elements=300

        num_DOFs.append((polynomial_order+1)*num_elements)

        eulers_DG = EulersEquations(
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

        init = eulers_DG.initial_condition(eulers_DG.DG_vars.x.flatten('F'))

        
        sol, t_vec = eulers_DG.solve(t=0, q_init=init, t_final=0.2)

        '''
        true_sol = lambda t: eulers_DG.initial_condition(
            eulers_DG.DG_vars.x.flatten('F') - 3*t
            )
        true_sol_array = np.zeros(
            (eulers_DG.DG_vars.num_states, 
            eulers_DG.DG_vars.Np * eulers_DG.DG_vars.K, 
            len(t_vec))
            )
        for i in range(len(t_vec)):
            true_sol_array[:, :, i] = true_sol(t_vec[i])
            
        l2_error = np.linalg.norm(sol - true_sol_array) / np.linalg.norm(true_sol_array)
    
        error.append(l2_error)
        '''
    '''
    plt.figure()
    plt.loglog(num_DOFs, error, '.-', label='error', linewidth=2, markersize=10)
    plt.loglog(num_DOFs, [10**(-i) for i in range(len(num_DOFs))], label='slope 1', linewidth=2)
    plt.loglog(num_DOFs, [10**(-i*2) for i in range(len(num_DOFs))], label='slope 2', linewidth=2)
    plt.loglog(num_DOFs, [10**(-i*3) for i in range(len(num_DOFs))], label='slope 3', linewidth=2)
    plt.legend()
    plt.grid()
    plt.show()
    '''

    plt.figure()
    plt.plot(eulers_DG.DG_vars.x.flatten('F'), init[0], label='initial rho', linewidth=1)
    plt.plot(eulers_DG.DG_vars.x.flatten('F'), init[1]/init[0], label='initial u', linewidth=1)
    #plt.plot(eulers_DG.DG_vars.x.flatten('F'), true_sol(t_vec[-1])[0], label='true', linewidth=3)
    plt.plot(eulers_DG.DG_vars.x.flatten('F'), sol[0, :, -1], label='rho', linewidth=2)
    plt.plot(eulers_DG.DG_vars.x.flatten('F'), sol[1, :, -1]/sol[0, :, -1], label='u', linewidth=2)
    plt.grid()
    plt.legend()
    plt.show()

