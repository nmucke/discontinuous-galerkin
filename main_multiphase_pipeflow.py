import time
import numpy as np
from discontinuous_galerkin.base.base_model import BaseModel
from discontinuous_galerkin.start_up_routines.start_up_1D import StartUp1D
import matplotlib.pyplot as plt
import pdb

from matplotlib.animation import FuncAnimation


class PipeflowEquations(BaseModel):
    """Multiphase equation model class."""

    def __init__(
        self, **kwargs):
        super().__init__(**kwargs)
        
        self.L = 10000 # meters
        self.d = 0.146 # meters
        self.A = np.pi*self.d**2/4 # meters^2
        self.c = 308. # m/s
        self.rho_g_norm = 1.26 # kg/m^3
        self._rho_l = 1000. # kg/m^3
        self.p_amb = 101325.
        self.p_norm = 1.0e5 # Pa
        self.p_outlet = 1.0e6 # Pa
        self.e = 1e-8 # meters
        self.mu_g = 1.8e-5 # Pa*s
        self.mu_l = 8.9e-4 # Pa*s
        self.Cd = 5e-4
        self.T_norm = 278 # Kelvin

        self.rho_l = 1000. # kg/m^3
        

    def density_to_pressure(self, rho):
        """Compute the pressure from the density."""

        return self.p_norm * rho * self.T_norm / self.rho_g_norm / self.T_norm


    def pressure_to_density(self, p):
        """Compute the density from the pressure."""

        return self.rho_g_norm * self.T_norm / self.p_norm * p / self.T_norm
    
    def transform_matrices(self, t, q):
        """Compute the conservative to primitive transform matrices."""

        rho_g_A_g = q[0]
        rho_l_A_l = q[1]
        rho_m_u_m_A = q[2]

        rho_m_A = rho_g_A_g + rho_l_A_l
        rho_m = rho_m_A/self.A

        u_m = rho_m_u_m_A/rho_m_A

        A_l = rho_l_A_l/self.rho_l
        A_g = self.A - A_l

        rho_g = rho_g_A_g/A_g

        c_g_squared = rho_g * self.T_norm / self.rho_g_norm / self.T_norm

        p = self.density_to_pressure(rho_g)

        P = np.zeros((self.DG_vars.num_states, self.DG_vars.num_states))
        P_inv = np.zeros((self.DG_vars.num_states, self.DG_vars.num_states))
        A = np.zeros((self.DG_vars.num_states, self.DG_vars.num_states))
        S = np.zeros((self.DG_vars.num_states, self.DG_vars.num_states))
        S_inv = np.zeros((self.DG_vars.num_states, self.DG_vars.num_states))
        Lambda = np.zeros((self.DG_vars.num_states, self.DG_vars.num_states))

        drho_gdp = self.rho_g_norm * self.T_norm / self.p_norm / self.T_norm
        drho_ldp = 0
        drho_mdA_l = self.rho_l/self.A
        drho_mdp = drho_gdp*A_g/self.A
        P[0, :] = np.array([-rho_g, A_g * drho_gdp, 0])
        P[1, :] = np.array([self.rho_l, A_l * drho_ldp, 0])
        P[2, :] = np.array([u_m * drho_mdA_l * self.A, u_m * drho_mdp * self.A, rho_m * self.A])

        P_inv = np.linalg.inv(P)

        A[0, :] = np.array([u_m, 0, A_l])
        A[1, :] = np.array([0, u_m, p*self.A/A_g])
        A[2, :] = np.array(
            [0, c_g_squared * self.A/(A_l * self.rho_l * c_g_squared + A_g * p), u_m]
            )

        term_1 = np.sqrt(A_l * self.rho_l * c_g_squared + A_g * p)
        term_2 = self.A * np.sqrt(c_g_squared) * np.sqrt(p)

        S[0, :] = np.array(
            [- np.sqrt(A_g) * A_l * term_1 / term_2, np.sqrt(A_g) * A_l * term_1 / term_2, 1.]
            )
        S[1, :] = np.array(
            [- np.sqrt(p) * term_1 / term_2, np.sqrt(p) * term_1 / term_2, 0.]
            )
        S[2, :] = np.array(
            [1., 1., 0.]
            ) 
        
        S_inv[0, :] = np.array(
            [0, -np.sqrt(A_g * c_g_squared) / (2 * np.sqrt(p) * term_1), 0.5]
            )
        S_inv[1, :] = np.array(
            [0, np.sqrt(A_g * c_g_squared) / (2 * np.sqrt(p) * term_1), 0.5]
            )
        S_inv[2, :] = np.array(
            [1, - A_g * A_l / (self.A * p), 0]
            ) 

        term_1 = A_g**(3/2) * p * u_m
        term_2 = self.A * np.sqrt(c_g_squared) * np.sqrt(p) * np.sqrt(A_l * self.rho_l * c_g_squared + A_g * p)
        term_3 = np.sqrt(A_g) * A_l * c_g_squared * self.rho_l * u_m
        term_4 = A_g**(3/2) * p + np.sqrt(A_g) * A_l * c_g_squared * self.rho_l

        Lambda[0, 0] = (term_1 - term_2 + term_3) / term_4
        Lambda[1, 1] = (term_1 + term_2 + term_3) / term_4
        Lambda[2, 2] = u_m

        return P, P_inv, A, S, S_inv, Lambda

    def friction_factor(self, q):
        """Compute the friction factor."""

        rho_g_A_g = q[0]
        rho_l_A_l = q[1]
        rho_m_u_m_A = q[2]

        rho_m_A = rho_g_A_g + rho_l_A_l
        rho_m = rho_m_A/self.A

        u_m = rho_m_u_m_A/rho_m_A

        A_l = rho_l_A_l/self.rho_l
        A_g = self.A - A_l

        mu_m = (self.mu_g * A_g + self.mu_l * A_l) / self.A

        Re = rho_m * u_m * self.d / mu_m

        a = (-2.457 * np.log((7/Re)**0.9 + 0.27*self.e/self.d))**16
        b = (37530/Re)**16
        
        f_w = 8 * ((8/Re)**12 + (a + b)**(-1.5))**(1/12)

        T_w = self.A/self.d/2 * f_w * rho_m * u_m * np.abs(u_m)

        return T_w

    def initial_condition(self, x):
        """Compute the initial condition.
        
        initial condition:
        rho_g * A_g * u_m = a
        rho_l * A_l * u_m = b
        rho_m * u_m * A = a + b
        
        Conservative:
        q[0] = rho_g * A_g = 
        q[1] = rho_l * A_l
        q[2] = rho_m * u_m * A
        
        Primitive:
        q[0] = A_l
        q[1] = p
        q[2] = u_m

        rho_g = self.p_outlet * (rho_g_norm * self.T_norm) / (self.p_norm * self.T_norm)
        alpha_l  = b/self.rho_l / (a/rho_g + b/self.rho_l)
        u_m = b /(self.rho_l * self.A * self.alpha_l)
        """

        
        a = 0.2
        b = 20

        rho_g = self.p_outlet * (self.rho_g_norm * self.T_norm) / (self.p_norm * self.T_norm)
        alpha_l  = b/self.rho_l / (a/rho_g + b/self.rho_l)
        u_m = b /(self.rho_l * self.A * alpha_l)

        alpha_g = 1 - alpha_l

        rho_g_A_l = rho_g * alpha_g * self.A
        rho_l_A_l = self.rho_l * alpha_l * self.A
        rho_m_u_m = (rho_g * alpha_g + self.rho_l * alpha_l) * u_m * self.A 

        init = np.ones((self.DG_vars.num_states, x.shape[0]))

        init[0, :] = rho_g_A_l#0.2 / 4
        init[1, :] = rho_l_A_l#20 / 4
        init[2, :] = rho_m_u_m#init[0, :] + init[1, :]

        return init
    
    def boundary_conditions(self, t, q=None):
        """Compute the boundary conditions.
        
        Conservative: 
        q[0] = rho_g * A_g
        q[1] = rho_l * A_l
        q[2] = rho_m * u_m * A

        Primitive:
        q[0] = A_l
        q[1] = p
        q[2] = u_m        
        """

        rho_g_A_g = q[0]
        rho_l_A_l = q[1]
        rho_m_u_m_A = q[2]

        rho_m_A = rho_g_A_g + rho_l_A_l
        rho_m = rho_m_A/self.A

        u_m = rho_m_u_m_A/rho_m_A

        A_l = rho_l_A_l/self.rho_l
        A_g = self.A - A_l

        rho_g = rho_g_A_g/A_g

        p = self.density_to_pressure(rho_g)

        a = 0.2
        b = 20

        rho_g = self.p_outlet * (self.rho_g_norm * self.T_norm) / (self.p_norm * self.T_norm)
        alpha_l  = b/self.rho_l / (a/rho_g + b/self.rho_l)
        u_m_bc = b /(self.rho_l * self.A * alpha_l)

        A_l_BC = alpha_l * self.A

        #rho_out = self.pressure_to_density(self.p_ref)
        BC_state_1 = {
            'left': (A_l - A_l_BC)/self.step_size,
            'right': None,
        }
        BC_state_2 = {
            'left': None,
            'right': (p-self.p_outlet)/self.step_size
        }
        BC_state_3 = {
            'left': (u_m - u_m_bc)/self.step_size,
            'right': None
        }

        BCs = [BC_state_1, BC_state_2, BC_state_3]
        
        return BCs
    
    def velocity(self, q):
        """Compute the wave speed."""

        rho_g_A_g = q[0]
        rho_l_A_l = q[1]
        rho_m_u_m_A = q[2]

        rho_m_A = rho_g_A_g + rho_l_A_l
        rho_m = rho_m_A/self.A

        u_m = rho_m_u_m_A/rho_m_A

        A_l = rho_l_A_l/self.rho_l
        A_g = self.A - A_l

        rho_g = rho_g_A_g/A_g

        c_g_squared = rho_g * self.T_norm / self.rho_g_norm / self.T_norm
        term_1 = A_g**(3/2) * p * u_m
        term_2 = self.A * np.sqrt(c_g_squared) * np.sqrt(p) * np.sqrt(A_l * self.rho_l * c_g_squared + A_g * p)
        term_3 = np.sqrt(A_g) * A_l * c_g_squared * self.rho_l * u_m
        term_4 = A_g**(3/2) * p + np.sqrt(A_g) * A_l * c_g_squared * self.rho_l

        lambda_1 = (term_1 - term_2 + term_3) / term_4
        lambda_2 = (term_1 + term_2 + term_3) / term_4
        lambda_3 = u_m

        return np.maximum(np.abs(lambda_1), np.abs(lambda_2), np.abs(lambda_3))
        
    def flux(self, q):
        """Compute the flux."""

        flux = np.zeros((self.DG_vars.num_states, q.shape[1]))

        rho_g_A_g = q[0]
        rho_l_A_l = q[1]
        rho_m_u_m_A = q[2]

        rho_m_A = rho_g_A_g + rho_l_A_l
        rho_m = rho_m_A/self.A

        u_m = rho_m_u_m_A/rho_m_A

        A_l = rho_l_A_l/self.rho_l
        A_g = self.A - A_l

        flux[0] = rho_g_A_g * u_m
        flux[1] = rho_l_A_l * u_m
        
        p = self.density_to_pressure(rho_g_A_g/A_g)
        flux[2] = rho_m_u_m_A * u_m + p * self.A
        
        return flux
    
    def source(self, t, q):
        """Compute the source."""

        s = np.zeros((self.DG_vars.num_states,self.DG_vars.Np*self.DG_vars.K))

        '''
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
        '''


        return s

if __name__ == '__main__':


    basic_args = {
        'xmin': 0,
        'xmax': 10000,
        'num_elements': 100,
        'num_states': 3,
        'polynomial_order': 2,
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
    
    stabilizer_args = {
        'type': 'slope_limiter',
        'second_derivative_upper_bound': 1e-8,
    }
    '''
    stabilizer_args = {
        'type': 'filter',
        'num_modes_to_filter': 20,
        'filter_order': 6,
    }
    '''

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
        'treatment': 'characteristic',
        'form': {'left': 'primitive', 'right': 'primitive'},
    }

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
        steady_state_args=steady_state
    )

    x = np.linspace(0, 10000, 2000)

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