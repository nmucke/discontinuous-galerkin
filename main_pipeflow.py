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
        self, **kwargs):
        super().__init__(**kwargs)

        self.L = 2000
        self.d = 0.508
        self.A = np.pi*self.d**2/4
        self.c = 308.
        self.rho_ref = 52.67
        self.p_amb = 101325.
        self.p_ref = self.rho_ref*self.c**2#5016390.
        self.e = 1e-8
        self.mu = 1.2e-5
        self.Cd = 5e-4

    def density_to_pressure(self, rho):
        """Compute the pressure from the density."""
        return self.c**2*(rho - self.rho_ref) + self.p_ref


    def pressure_to_density(self, p):
        """Compute the density from the pressure."""

        return (p - self.p_ref)/self.c**2 + self.rho_ref
    
    def conservative_to_primitive_transform_matrix(self, t, q):
        """Compute the conservative to primitive transform matrices."""

        rho = q[0]/self.A
        u = q[1]/q[0]

        P = np.zeros((self.DG_vars.num_states, self.DG_vars.num_states))
        P_inv = np.zeros((self.DG_vars.num_states, self.DG_vars.num_states))
        A = np.zeros((self.DG_vars.num_states, self.DG_vars.num_states))
        S = np.zeros((self.DG_vars.num_states, self.DG_vars.num_states))
        S_inv = np.zeros((self.DG_vars.num_states, self.DG_vars.num_states))
        Lambda = np.zeros((self.DG_vars.num_states, self.DG_vars.num_states))

        P[0, 0] = 1/self.c/self.c * self.A
        P[1, 0] = 1/self.c/self.c * u * self.A
        P[0, 1] = 0
        P[1, 1] = rho*self.A

        P_inv[0, 0] = self.c*self.c/self.A
        P_inv[1, 0] = -u/rho/self.A
        P_inv[0, 1] = 0
        P_inv[1, 1] = 1/rho/self.A

        A[0, 0] = u
        A[1, 0] = 1/rho
        A[0, 1] = self.c*self.c*rho
        A[1, 1] = u

        C = P_inv @ self.source(t, q)

        S[0, 0] = 1
        S[1, 0] = -1/rho/self.c
        S[0, 1] = 1
        S[1, 1] = 1/rho/self.c
        S *= 1/2

        S_inv[0, 0] = 1
        S_inv[1, 0] = 1
        S_inv[0, 1] = -rho*self.c
        S_inv[1, 1] = rho*self.c

        Lambda[0, 0] = u - self.c
        Lambda[1, 0] = 0
        Lambda[0, 1] = 0
        Lambda[1, 1] = u + self.c

        return P, P_inv, A, C, S, S_inv, Lambda

    def friction_factor(self, q):
        """Compute the friction factor."""

        rho = q[0]/self.A
        u = q[1]/q[0]

        Re = rho * u * self.d / self.mu
        
        f = (self.e/self.d/3.7)**(1.11) + 6.9/Re
        f *= -1/4*1.8*np.log10(f)**(-2)
        f *= -1/2/self.d * rho * u*u 

        return f


    '''
    def eigen(self, q):
        """Compute the eigenvalues."""

        u = q[1]/q[0]
        p = (self.gamma-1)*(q[2] - 0.5*q[0]*u*u)
        c = np.sqrt(self.gamma*p/q[0])

        H = c*c/(self.gamma-1) + 0.5*u*u

        lambda_1 = u - c
        lambda_2 = u

        D = np.diag(np.array([lambda_1, lambda_2]))

        R = np.zeros((self.DG_vars.num_states, self.DG_vars.num_states))

        R[:, 0] = np.array([x, x])
        R[:, 1] = np.array([x, x])


        L = np.zeros((self.DG_vars.num_states, self.DG_vars.num_states))

        L[0, :] = np.array([
            x,
            x
            ])
        L[1, :] = 1/c/c * np.array([
            x,
            x
        ])

        return D, L, R
    '''

    def system_jacobian(self, q):

        u = q[1]/q[0]

        J =np.array(
            [[0, 1],
            [-u*u + self.c*self.c/np.sqrt(self.A), 2*u]]
            )

        return J

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
            'left': q[0, 0] * (4.0 + 0.5),#*np.sin(0.2*t)),
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
            point_source = \
                (np.heaviside(x-500 + width/2, 1) - np.heaviside(x-500-width/2, 1))
            point_source *= 1/width

        rho = q[0]/self.A
        p = self.density_to_pressure(rho)

        s[0] = - self.Cd * np.sqrt(rho * (p - self.p_amb)) * point_source
        s[0] *= 0.
        s[1] = -self.friction_factor(q)

        #s = 0*s

        return s

if __name__ == '__main__':
    


    basic_args = {
        'xmin': 0,
        'xmax': 2000,
        'num_elements': 75,
        'num_states': 2,
        'polynomial_order': 4,
        'polynomial_type': 'legendre',
    }

    steady_state = {
        'newton_params':{
            'solver': 'direct',
            'max_newton_iter': 200,
            'newton_tol': 1e-5
        }
    }

    numerical_flux_args = {
        'type': 'lax_friedrichs',
        'alpha': 0.5,
    }
    
    '''
    stabilizer_args = {
        'type': 'slope_limiter',
        'second_derivative_upper_bound': 1e-8,
    }
    '''
    stabilizer_args = {
        'type': 'filter',
        'num_modes_to_filter': 10,
        'filter_order': 6,
    }

    time_integrator_args = {
        'type': 'implicit_euler',
        'step_size': 0.05,
        'newton_params':{
            'solver': 'krylov',
            'max_newton_iter': 200,
            'newton_tol': 1e-5
            }
        }

    BC_args = {
        'type': 'dirichlet',
        'treatment': 'naive',
    }

    error = []
    conv_list = [2]
    num_DOFs = []
    for polynomial_order in conv_list:

        #polynomial_order=8
        num_elements = 150

        num_DOFs.append((polynomial_order+1)*num_elements)

        pipe_DG = PipeflowEquations(
            basic_args=basic_args,
            BC_args=BC_args,
            stabilizer_args=stabilizer_args,
            time_integrator_args=time_integrator_args,
            numerical_flux_args=numerical_flux_args,
        )


        init = pipe_DG.initial_condition(pipe_DG.DG_vars.x.flatten('F'))

        t_final = 20.0
        sol, t_vec = pipe_DG.solve(
            t=0, 
            q_init=init, 
            t_final=t_final, 
            steady_state_args=None
        )

        x = np.linspace(0, 2000, 2000)

        u = np.zeros((len(x), len(t_vec)))
        rho = np.zeros((len(x), len(t_vec)))
        for t in range(sol.shape[-1]):
            rho[:, t] = pipe_DG.evaluate_solution(x, sol_nodal=sol[0, :, t])
            u[:, t] = pipe_DG.evaluate_solution(x, sol_nodal=sol[1, :, t])
        rho = rho / pipe_DG.A
        u = u / rho / pipe_DG.A
    
    t_vec = np.arange(0, u.shape[1])

    plt.figure()
    plt.imshow(u, extent=[0, t_final, 2000, 0], aspect='auto')
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.plot(x, u[:, 0], label='u', linewidth=2)
    plt.plot(x, u[:, -1], label='u', linewidth=2)
    plt.grid()
    plt.legend()
    plt.savefig('pipeflow.png')
    plt.show()

    fig, ax = plt.subplots()
    xdata, ydata = [], []
    ln, = ax.plot([], [], lw=3, animated=True)

    def init():
        ax.set_xlim(0, 2000)
        ax.set_ylim(u.min(), u.max())
        return ln,

    def update(frame):
        xdata.append(x)
        ydata.append(u[:, frame])
        ln.set_data(x, u[:, frame])
        return ln,

    ani = FuncAnimation(
        fig,
        update,
        frames=t_vec,
        init_func=init, 
        blit=True,
        interval=10,
        )
    ani.save('pipeflow.gif', fps=30)
    plt.show()