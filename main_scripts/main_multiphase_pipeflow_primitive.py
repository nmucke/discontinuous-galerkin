import time
import numpy as np
from discontinuous_galerkin.base.base_model import BaseModel
from discontinuous_galerkin.start_up_routines.start_up_1D import StartUp1D
import matplotlib.pyplot as plt
import pdb
from scipy.optimize import fsolve, least_squares

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
        self.rho_l = 1003. # kg/m^3
        self.p_amb = 101325.
        self.p_norm = 1.0e5 # Pa
        self.p_outlet = 1.0e6 # Pa
        self.e = 1e-8 # meters
        self.mu_g = 1.8e-5 # Pa*s
        self.mu_l = 1.516e-3 # Pa*s
        self.Cd = 5e-4
        self.T_norm = 278 # Kelvin
        self.T = 278 # Kelvin

        self.rho_l = 1000. # kg/m^3

        self.conservative_or_primitive = 'primitive'
        

    def density_to_pressure(self, rho):
        """Compute the pressure from the density."""

        return self.p_norm * rho * self.T / self.rho_g_norm / self.T_norm


    def pressure_to_density(self, p):
        """Compute the density from the pressure."""

        return self.rho_g_norm * self.T_norm / self.p_norm * p / self.T
    
    def conservative_to_primitive(self, q):
        """Compute the primitive variables from the conservative variables."""

        A_l = q[1]/self.rho_l
        A_g = self.A - A_l

        rho_g = q[0]/A_g

        p = self.density_to_pressure(rho_g)

        alpha_l = A_l/self.A
        alpha_g = A_g/self.A

        rho_m = rho_g * alpha_g + self.rho_l * alpha_l

        u_m = q[2]/rho_m

        return np.array([A_l, p, u_m])
    
    def primitive_to_conservative(self, q):
        """Compute the conservative variables from the primitive variables."""

        A_l = q[0]
        p = q[1]*self.p_outlet
        u_m = q[2]

        rho_g = self.pressure_to_density(p)

        A_g = self.A - A_l

        alpha_l = A_l/self.A
        alpha_g = A_g/self.A

        rho_m = rho_g * alpha_g + self.rho_l * alpha_l

        rho_g_A_g = rho_g * A_g
        rho_l_A_l = self.rho_l * A_l
        rho_m_u_m_A = rho_m * u_m * self.A

        return np.array([rho_g_A_g, rho_l_A_l, rho_m_u_m_A])
    

    def eigen(self, q):
        """Compute the eigenvalues and eigenvectors of the flux Jacobian."""
        
        if self.conservative_or_primitive == 'conservative':
            rho_g_A_g = q[0]
            rho_l_A_l = q[1]
            rho_m_u_m = q[2]

            A_l, p, u_m = self.conservative_to_primitive(q=q)
        elif self.conservative_or_primitive == 'primitive':
            A_l = q[0]
            p = q[1]
            u_m = q[2]

            rho_g_A_g, rho_l_A_l, rho_m_u_m = self.primitive_to_conservative(q=q)

        rho_g = self.pressure_to_density(p)

        A_g = self.A - A_l

        rho_m = rho_g * A_g + self.rho_l * A_l

        c_g_squared = rho_g * self.T_norm / self.rho_g_norm / self.T_norm

        L = np.zeros((self.DG_vars.num_states, self.DG_vars.num_states))
        R = np.zeros((self.DG_vars.num_states, self.DG_vars.num_states))
        D = np.zeros((self.DG_vars.num_states, self.DG_vars.num_states))

        term_1 = np.sqrt(A_l * self.rho_l * c_g_squared + A_g * p)
        term_2 = self.A * np.sqrt(c_g_squared) * np.sqrt(p)

        R[0, :] = np.array(
            [-np.sqrt(A_g) * A_l * term_1 / term_2, np.sqrt(A_g) * A_l * term_1 / term_2, 1.]
            )
        R[1, :] = np.array(
            [-np.sqrt(p) * term_1 / term_2, np.sqrt(p) * term_1 / term_2, 0.]
            )
        R[2, :] = np.array(
            [1., 1., 0.]
            ) 

        L[0, :] = np.array(
            [0, -np.sqrt(A_g * c_g_squared) / (2 * np.sqrt(p) * term_1), 0.5]
            )
        L[1, :] = np.array(
            [0, np.sqrt(A_g * c_g_squared) / (2 * np.sqrt(p) * term_1), 0.5]
            )
        L[2, :] = np.array(
            [1, - A_g * A_l / (self.A * p), 0]
            ) 

        term_1 = A_g**(3/2) * p * u_m
        term_2 = self.A * np.sqrt(c_g_squared) * np.sqrt(p) * np.sqrt(A_l * self.rho_l * c_g_squared + A_g * p)
        term_3 = np.sqrt(A_g) * A_l * c_g_squared * self.rho_l * u_m
        term_4 = A_g**(3/2) * p + np.sqrt(A_g) * A_l * c_g_squared * self.rho_l

        D[0, 0] = (term_1 - term_2 + term_3) / term_4
        D[1, 1] = (term_1 + term_2 + term_3) / term_4
        D[2, 2] = u_m

        return D, L, R

    def transform_matrices(
        self, 
        t: float, 
        q: np.ndarray = None,
        ):
        """Compute the conservative to primitive transform matrices."""

        if self.conservative_or_primitive == 'conservative':
            rho_g_A_g = q[0]
            rho_l_A_l = q[1]
            rho_m_u_m = q[2]

            A_l, p, u_m = self.conservative_to_primitive(q=q)
        elif self.conservative_or_primitive == 'primitive':
            A_l = q[0]
            p = q[1]*self.p_outlet
            u_m = q[2]

            rho_g_A_g, rho_l_A_l, rho_m_u_m = self.primitive_to_conservative(q=q)

        rho_g = self.pressure_to_density(p)

        A_g = self.A - A_l

        rho_m = rho_g * A_g + self.rho_l * A_l

        c_g_squared = rho_g * self.T_norm / self.rho_g_norm / self.T_norm

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
            [-np.sqrt(A_g) * A_l * term_1 / term_2, np.sqrt(A_g) * A_l * term_1 / term_2, 1.]
            )
        S[1, :] = np.array(
            [-np.sqrt(p) * term_1 / term_2, np.sqrt(p) * term_1 / term_2, 0.]
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

        A_l = q[0]
        p = q[1]*self.p_outlet
        u_m = q[2]

        A_g = self.A - A_l

        alpha_l = A_l/self.A
        alpha_g = A_g/self.A

        rho_g = self.pressure_to_density(p)

        #rho_m = rho_g * A_g + self.rho_l * A_l
        rho_m = rho_g * alpha_g + self.rho_l * alpha_l
        mu_m = self.mu_g * alpha_g + self.mu_l * alpha_l

        Re = rho_m * np.abs(u_m) * self.d / mu_m

        a = (-2.457 * np.log((7/Re)**0.9 + 0.27*self.e/self.d))**16
        b = (37530/Re)**16
        
        f_w = 8 * ((8/Re)**12 + (a + b)**(-1.5))**(1/12)

        T_w = f_w * rho_m * u_m * np.abs(u_m) / (2 * self.d) * self.A

        return T_w

    def initial_condition(self, x):
        """Compute the initial condition.
        
        initial condition:
        rho_g * A_g * u_m = a
        rho_l * A_l * u_m = b
        rho_m * u_m * A = a + b
        
        Conservative:
        q[0] = rho_g * A_g
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

        rho_g = self.pressure_to_density(self.p_outlet)#self.p_outlet * (self.rho_g_norm * self.T_norm) / (self.p_norm * self.T)
        alpha_l  = b/self.rho_l / (a/rho_g + b/self.rho_l)
        u_m = b /(self.rho_l * self.A * alpha_l)

        init = np.ones((self.DG_vars.num_states, x.shape[0]))

        init[0, :] = alpha_l*self.A
        init[1, :] = 1.#self.p_outlet/self.p_outlet
        init[2, :] = u_m

        return init
    

    def boundary_conditions(self, t=0, q=None):
        """Compute the boundary conditions."""

        #A_l_left, p_left, u_m_left = self._get_primitive_BCs_left(q[:,0], t=t)


        BC_state_1 = {
            'left': None,
            'right': None,
        }
        BC_state_2 = {
            'left': None,
            'right': 1.#self.p_outlet,
        }
        BC_state_3 = {
            'left': None,
            'right': None
        }

        BCs_state = [BC_state_1, BC_state_2, BC_state_3]

        t_start = 100
        t_end = 110

        # a increases linearly from 0.2 to 0.4 over 10 seconds
        if t < t_end and t > t_start:
            t_ = t - t_start
            gas_mass_inflow = 0.2 + 0.02 * t_ / 10
        elif t > t_end:
            gas_mass_inflow = 0.4 #+ 0.1*np.sin(t/200)
        else:
            gas_mass_inflow = 0.2

        liquid_mass_inflow = 20.

        BC_flux_1 = {
            'left': gas_mass_inflow,
            'right': None,
        }
        BC_flux_2 = {
            'left': liquid_mass_inflow,
            'right': None,
        }
        BC_flux_3 = {
            'left': None,
            'right': None
        }

        BCs_flux = [BC_flux_1, BC_flux_2, BC_flux_3]

        BCs = {
            'state': BCs_state,
            'flux': BCs_flux,
        }

        return BCs

    
    def velocity(self, q):
        """Compute the wave speed."""

        A_l = q[0]
        p = q[1]*self.p_outlet
        u_m = q[2]

        A_g = self.A - A_l

        rho_g = self.pressure_to_density(p)

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

        A_l = q[0]
        p = q[1]*self.p_outlet
        u_m = q[2]

        A_g = self.A - A_l

        rho_g = self.pressure_to_density(p)

        alpha_l = A_l/self.A
        alpha_g = A_g/self.A

        rho_m = rho_g * alpha_g + self.rho_l * alpha_l

        flux[0] = rho_g * A_g * u_m
        flux[1] = self.rho_l * A_l * u_m
        
        flux[2] = rho_m * u_m**2 * self.A + p * self.A
        
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
        '''
        s[-1] = -self.friction_factor(q)

        return s

if __name__ == '__main__':


    basic_args = {
        'xmin': 0,
        'xmax': 10000,
        'num_elements': 200,
        'num_states': 3,
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
        'alpha': .5,
    }    
    
    '''
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
    time_integrator_args = {
        'type': 'implicit_euler',
        'step_size': 2,
        'newton_params': {
            'solver': 'direct',
            'max_newton_iter': 200,
            'newton_tol': 1e-5
        }
    }
    '''
    time_integrator_args = {
        'type': 'SSPRK',
    }
    '''

    BC_args = {
        'type': 'dirichlet',
        'treatment': 'naive',
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

    t_final = 4000.
    sol, t_vec = pipe_DG.solve(
        t=0, 
        q_init=init, 
        t_final=t_final, 
        steady_state_args=steady_state
    )

    x = np.linspace(0, 10000, 2000)

    A_l = np.zeros((len(x), len(t_vec)))
    p = np.zeros((len(x), len(t_vec)))
    u_m = np.zeros((len(x), len(t_vec)))
    for t in range(sol.shape[-1]):
        A_l[:, t] = pipe_DG.evaluate_solution(x, sol_nodal=sol[0, :, t])
        p[:, t] = pipe_DG.evaluate_solution(x, sol_nodal=sol[1, :, t])
        u_m[:, t] = pipe_DG.evaluate_solution(x, sol_nodal=sol[2, :, t])
    
    
    u = u_m
    alpha_l = A_l/pipe_DG.A#rho_l_A_l/pipe_DG.A/pipe_DG.rho_l
    #alpha_l = u_m

    
    rho_g = pipe_DG.pressure_to_density(p*pipe_DG.p_outlet)
    rho_g_A_g_u_m = u_m * rho_g * (pipe_DG.A - A_l)
    rho_l_A_l_u_m = u_m * pipe_DG.rho_l * A_l
    print(rho_g_A_g_u_m[0, :])
    #alpha_l = p
    #pdb.set_trace()
    #print(alpha_l[0, -1])

    #alpha_l = p
    
    t_vec = np.arange(0, u.shape[1]-1)

    plt.figure()
    plt.imshow(alpha_l, extent=[0, t_final, 10000, 0], aspect='auto')
    plt.colorbar()
    plt.show()

    plt.figure()
    #plt.plot(x, alpha_l[:, 0], label='alpha_l', linewidth=2)
    plt.plot(x, alpha_l[:, -1], label='alpha_l', linewidth=2)
    #plt.ylim(0.1, 0.75)
    #plt.ylim(18, 35)
    plt.grid()
    plt.legend()
    plt.savefig('pipeflow.png')
    plt.show()

    if alpha_l.shape[-1] > 200:
        t_idx = np.linspace(0, alpha_l.shape[-1]-1, 200, dtype=int)
        alpha_l = alpha_l[:, t_idx]

    t_vec = np.arange(0, alpha_l.shape[-1])

    fig, ax = plt.subplots()
    xdata, ydata = [], []
    ln, = ax.plot([], [], lw=3, animated=True)

    def init():
        ax.set_xlim(0, 10000)
        ax.set_ylim(alpha_l.min(), alpha_l.max())
        #ax.set_ylim(18, 35)
        return ln,

    def update(frame):
        xdata.append(x)
        ydata.append(alpha_l[:, frame])
        ln.set_data(x, alpha_l[:, frame])
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