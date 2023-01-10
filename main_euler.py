import numpy as np
from discontinuous_galerkin.base.base_model import BaseModel
from discontinuous_galerkin.start_up_routines.start_up_1D import StartUp1D
import matplotlib.pyplot as plt
import pdb
from matplotlib.animation import FuncAnimation
import time

'''
xmin=0.,
xmax=1.,
num_elements=10,
polynomial_order=5,
polynomial_type='legendre',
num_states=1,
BCs={'type': 'dirichlet'},
stabilizer_type=None, 
stabilizer_params=None,
time_integrator_type='implicit_euler',
time_integrator_params=None,
numerical_flux_type='lax_friedrichs',
numerical_flux_params=None,


    xmin=xmin,
    xmax=xmax,
    num_elements=num_elements,
    polynomial_order=polynomial_order,
    polynomial_type=polynomial_type,
    num_states=num_states,
    BC_types=BCs,
    stabilizer_type=stabilizer_type, 
    stabilizer_params=stabilizer_params,
    time_integrator_type=time_integrator_type,
    time_integrator_params=time_integrator_params,
    numerical_flux_type=numerical_flux_type,
    numerical_flux_params=numerical_flux_params,
'''
class EulersEquations(BaseModel):
    """Advection equation model class."""

    def __init__(
        self, 
        **kwargs
        ):
        super().__init__(**kwargs)

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
    
    def boundary_conditions(self, t, q=None):
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
    
    def eigen(self, q):
        """Compute the eigenvalues."""

        u = q[1]/q[0]
        p = (self.gamma-1)*(q[2] - 0.5*q[0]*u*u)
        c = np.sqrt(self.gamma*p/q[0])

        H = c*c/(self.gamma-1) + 0.5*u*u

        lambda_1 = u - c
        lambda_2 = u
        lambda_3 = u + c

        D = np.diag(np.array([lambda_1, lambda_2, lambda_3]))

        R = np.zeros((self.DG_vars.num_states, self.DG_vars.num_states))

        R[:, 0] = np.array([1.0, u - c, H - u*c])
        R[:, 1] = np.array([1.0, u, 0.5*u*u])
        R[:, 2] = np.array([1.0, u + c, H + u*c])


        L = np.zeros((self.DG_vars.num_states, self.DG_vars.num_states))

        L[0, :] = 0.5/c/c * np.array([
            (2*c + u*(self.gamma-1)) * u/2,
            - c - u*(self.gamma-1),
            self.gamma-1
            ])
        L[1, :] = 1/c/c * np.array([
            c*c - (self.gamma-1)*u*u/2,
            (self.gamma-1)*u,
            -(self.gamma - 1)
        ])
        L[2, :] = 0.5/c/c * np.array([
            -(2*c - u*(self.gamma-1)) * u/2,
            c - u*(self.gamma-1),
            self.gamma-1
        ])

        return D, L, R
        
    def velocity(self, q):
        """Compute the wave speed."""

        p = (self.gamma-1)*(q[2] - 0.5*q[1]*q[1]/q[0])

        return np.sqrt(self.gamma*p/q[0]) + abs(q[1]/q[0])
        
    def flux(self, q):
        """Compute the flux."""

        u = q[1]/q[0]

        p = (self.gamma-1)*(q[2] - 0.5*q[0]*u*u)

        flux = np.zeros((self.DG_vars.num_states, q.shape[1]))

        flux[0] = q[1]
        flux[1] = q[0] * u * u + p
        flux[2] = (q[2] + p) * u

        return flux
    
    def source(self, t, q):
        """Compute the source."""

        return np.zeros((self.DG_vars.num_states,self.DG_vars.Np*self.DG_vars.K))

if __name__ == '__main__':
    

    xmin = 0.
    xmax = 1

    BC_params = {
        'type':'dirichlet',
        'treatment': 'characteristic',
        'numerical_flux': 'lax_friedrichs',
    }

    numerical_flux_type = 'roe'
    numerical_flux_params = {
    }
    
    '''
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
        'num_modes_to_filter': 10,
        'filter_order': 6,
    }
    '''

    time_integrator_type = 'SSPRK'
    time_integrator_params = {
        'step_size': 0.0001,
        'newton_params':{
            'solver': 'krylov',
            'max_newton_iter': 200,
            'newton_tol': 1e-5
            }
    }

    polynomial_type='legendre'
    num_states=3

    error = []
    num_DOFs = []
    polynomial_order=2
    num_elements=50

    num_DOFs.append((polynomial_order+1)*num_elements)

    eulers_DG = EulersEquations(
        xmin=xmin,
        xmax=xmax,
        num_elements=num_elements,
        polynomial_order=polynomial_order,
        polynomial_type=polynomial_type,
        num_states=num_states,
        BC_params=BC_params,
        stabilizer_type=stabilizer_type, 
        stabilizer_params=stabilizer_params,
        time_integrator_type=time_integrator_type,
        time_integrator_params=time_integrator_params, 
        numerical_flux_type=numerical_flux_type,
        numerical_flux_params=numerical_flux_params,
        )

    init = eulers_DG.initial_condition(eulers_DG.DG_vars.x.flatten('F'))

    
    t1 = time.time()
    roe_sol, t_vec = eulers_DG.solve(t=0, q_init=init, t_final=0.2)
    t2 = time.time()
    print('Time to solve: ', t2-t1)

    del eulers_DG

    time_integrator_type = 'implicit_euler'
    time_integrator_params = {
        'step_size': 0.0001,
        'newton_params':{
            'solver': 'krylov',
            'max_newton_iter': 200,
            'newton_tol': 1e-5
            }
    }


    numerical_flux_type = 'lax_friedrichs'
    numerical_flux_params = {
        'alpha': 0.0,
    }

    BC_params = {
        'type':'dirichlet',
        'treatment': 'naive',
        'numerical_flux': 'lax_friedrichs',
    }
    eulers_DG = EulersEquations(
        xmin=xmin,
        xmax=xmax,
        num_elements=num_elements,
        polynomial_order=polynomial_order,
        polynomial_type=polynomial_type,
        num_states=num_states,
        BC_params=BC_params,
        stabilizer_type=stabilizer_type, 
        stabilizer_params=stabilizer_params,
        time_integrator_type=time_integrator_type,
        time_integrator_params=time_integrator_params, 
        numerical_flux_type=numerical_flux_type,
        numerical_flux_params=numerical_flux_params,
        )


    init = eulers_DG.initial_condition(eulers_DG.DG_vars.x.flatten('F'))

    
    t1 = time.time()
    lax_sol, t_vec = eulers_DG.solve(t=0, q_init=init, t_final=0.2)
    t2 = time.time()
    print('Time to solve: ', t2-t1)

    plt.figure()
    plt.plot(eulers_DG.DG_vars.x.flatten('F'), init[0], label='initial rho', linewidth=1)
    plt.plot(eulers_DG.DG_vars.x.flatten('F'), init[1]/init[0], label='initial u', linewidth=1)
    #plt.plot(eulers_DG.DG_vars.x.flatten('F'), true_sol(t_vec[-1])[0], label='true', linewidth=3)
    plt.plot(eulers_DG.DG_vars.x.flatten('F'), roe_sol[0, :, -1], label='Roe rho', linewidth=2)
    plt.plot(eulers_DG.DG_vars.x.flatten('F'), roe_sol[1, :, -1]/roe_sol[0, :, -1], label=' Roe u', linewidth=2)
    plt.plot(eulers_DG.DG_vars.x.flatten('F'), lax_sol[0, :, -1], label='Lax rho', linewidth=2)
    plt.plot(eulers_DG.DG_vars.x.flatten('F'), lax_sol[1, :, -1]/roe_sol[0, :, -1], label='Lax u', linewidth=2)
    plt.grid()
    plt.legend()
    plt.show()
    
    x = eulers_DG.DG_vars.x.flatten('F')
    fig = plt.figure()
    ax = plt.axes(xlim=(x.min(), x.max()), ylim=(0, 1.2))
    line, = ax.plot([], [], lw=3)

    def init():
        line.set_data([], [])
        return line,
    def animate(i):
        y = sol[0, :, i]
        line.set_data(x, y)
        return line,

    anim = FuncAnimation(
        fig, 
        animate, 
        init_func=init,
        frames=200, 
        interval=20, 
        blit=True
        )

    anim.save('lol.gif', writer='imagemagick')

