import time
import numpy as np
from discontinuous_galerkin.base.base_model import BaseModel
from discontinuous_galerkin.start_up_routines.start_up_1D import StartUp1D
import matplotlib.pyplot as plt
import pdb

from matplotlib.animation import FuncAnimation


class PipeflowEquations(BaseModel):
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
        steady_state=None,
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
            steady_state=steady_state,
            stabilizer_type=stabilizer_type, 
            stabilizer_params=stabilizer_params,
            time_integrator_type=time_integrator_type,
            time_integrator_params=time_integrator_params,
            numerical_flux_type=numerical_flux_type,
            numerical_flux_params=numerical_flux_params,
            )

        self.L = 2000
        self.d = 0.508
        self.A = np.pi*self.d**2/4
        self.c = 308.
        self.p_amb = 101325.
        self.p_ref = 5016390.
        self.rho_ref = 52.67
        self.e = 1e-8
        self.mu = 1.2e-5
        self.Cd = 5e-4

        

    def density_to_pressure(self, rho):
        """Compute the pressure from the density."""

        return self.c**2*(rho - self.rho_ref) + self.p_ref

    def pressure_to_density(self, p):
        """Compute the density from the pressure."""

        return (p - self.p_ref)/self.c**2 + self.rho_ref

    def friction_factor(self, q):
        """Compute the friction factor."""

        rho = q[0]/self.A
        u = q[1]/q[0]

        Re = rho * u * self.d / self.mu
        
        f = (self.e/self.d/3.7)**(1.11) + 6.9/Re
        f *= -1/4*1.8*np.log10(f)**(-2)
        f *= -1/2/self.d * rho * u*u 

        return f

    def initial_condition(self, x):
        """Compute the initial condition."""

        init = np.ones((self.DG_vars.num_states, x.shape[0]))

        init[0] = self.pressure_to_density(self.p_ref) * self.A
        init[1] = init[0] * 4.0

        return init
    
    def boundary_conditions(self, t, q=None):
        """Compute the boundary conditions."""

        rho_out = self.pressure_to_density(self.p_ref)

        BC_state_1 = {
            'left': None,
            'right': rho_out * self.A,
        }
        BC_state_2 = {
            'left': q[0, 0] * 4.0,
            'right': None
        }

        BCs = [BC_state_1, BC_state_2]
        
        return BCs
    
    def velocity(self, q):
        """Compute the wave speed."""
        
        u = q[1]/q[0]

        c = np.abs(u) + self.c/np.sqrt(self.A)

        return c
        
    def flux(self, q):
        """Compute the flux."""


        p = self.density_to_pressure(q[0]/self.A)

        flux = np.zeros((self.DG_vars.num_states, q.shape[1]))

        flux[0] = q[1]
        flux[1] = q[1]*q[1]/q[0] + p * self.A

        return flux
    
    def source(self, t, q):
        """Compute the source."""

        s = np.zeros((self.DG_vars.num_states,self.DG_vars.Np*self.DG_vars.K))

        point_source = np.zeros((self.DG_vars.Np*self.DG_vars.K))
        if t>0:
            x = pipe_DG.DG_vars.x.flatten('F')
            width = 50
            point_source = (np.heaviside(x-1000 + width/2, 1) - np.heaviside(x-1000-width/2, 1))
            point_source *= 1/width

        rho = q[0]/self.A
        p = self.density_to_pressure(rho)

        s[0] = - self.Cd * np.sqrt(rho * (p - self.p_amb)) * point_source

        s[1] = -self.friction_factor(q)

        return s

if __name__ == '__main__':
    

    xmin = 0.
    xmax = 2000

    BC_types = 'dirichlet'

    steady_state = {
        'newton_params':{
            'solver': 'direct',
            'max_newton_iter': 200,
            'newton_tol': 1e-5
        }
    }

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

    time_integrator_type = 'implicit_euler'
    time_integrator_params = {
        'step_size': 0.1,
        'newton_params':{
            'solver': 'krylov',
            'max_newton_iter': 200,
            'newton_tol': 1e-5
            }
        }

    polynomial_type='legendre'
    num_states=2

    error = []
    conv_list = [3]
    num_DOFs = []
    for polynomial_order in conv_list:

        #polynomial_order=8
        num_elements=100

        num_DOFs.append((polynomial_order+1)*num_elements)

        pipe_DG = PipeflowEquations(
            xmin=xmin,
            xmax=xmax,
            num_elements=num_elements,
            polynomial_order=polynomial_order,
            polynomial_type=polynomial_type,
            num_states=num_states,
            BC_types=BC_types,
            steady_state=steady_state,
            stabilizer_type=stabilizer_type, 
            stabilizer_params=stabilizer_params,
            time_integrator_type=time_integrator_type,
            time_integrator_params=time_integrator_params, 
            numerical_flux_type=numerical_flux_type,
            numerical_flux_params=numerical_flux_params,
            )


        init = pipe_DG.initial_condition(pipe_DG.DG_vars.x.flatten('F'))

        t1 = time.time()
        sol, t_vec = pipe_DG.solve(t=0, q_init=init, t_final=25.0)
        t2 = time.time()
        print('time to solve: ', t2-t1)

    plt.figure()
    #plt.plot(pipe_DG.DG_vars.x.flatten('F'), init[0], label='initial rho', linewidth=1)
    plt.plot(pipe_DG.DG_vars.x.flatten('F'), init[1]/init[0], label='initial u', linewidth=1)
    #plt.plot(eulers_DG.DG_vars.x.flatten('F'), true_sol(t_vec[-1])[0], label='true', linewidth=3)
    #plt.plot(pipe_DG.DG_vars.x.flatten('F'), sol[0, :, -1], label='rho', linewidth=2)
    plt.plot(pipe_DG.DG_vars.x.flatten('F'), sol[1, :, -1]/sol[0, :, -1], label='u', linewidth=2)
    plt.grid()
    plt.legend()
    plt.show()

    u = sol[1]/sol[0]#sol[0]/pipe_DG.A## #
    u = u[:, 1:-1]

    fig = plt.figure()
    ax = plt.axes(xlim=(0, xmax), ylim=(u.min(), u.max()))
    line, = ax.plot([], [], lw=3)

    def init():
        line.set_data([], [])
        return line,
    def animate(i):
        x = pipe_DG.DG_vars.x.flatten('F')
        y = u[:, i]
        line.set_data(x, y)
        return line,

    anim = FuncAnimation(
        fig, 
        animate, 
        init_func=init,
        frames=u.shape[1], 
        interval=20, 
        blit=True
        )


    anim.save('lol.gif', writer='imagemagick')