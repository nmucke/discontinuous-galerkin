import time
import numpy as np
from discontinuous_galerkin.base.base_model import BaseModel
from discontinuous_galerkin.polynomials.jacobi_polynomials import JacobiP
from discontinuous_galerkin.start_up_routines.start_up_1D import StartUp1D
import matplotlib.pyplot as plt
import pdb
from scipy.optimize import fsolve, least_squares

from matplotlib.animation import FuncAnimation

from scipy.linalg import eig
class Brownian():
    """
    A Brownian motion class constructor
    """
    def __init__(self,x0=0):
        """
        Init class
        """
        assert (type(x0)==float or type(x0)==int or x0 is None), "Expect a float or None for the initial value"
        
        self.x0 = float(x0)
    
    def gen_random_walk(self,n_step=100):
        """
        Generate motion by random walk
        
        Arguments:
            n_step: Number of steps
            
        Returns:
            A NumPy array with `n_steps` points
        """
        # Warning about the small number of steps
        if n_step < 30:
            print("WARNING! The number of steps is small. It may not generate a good stochastic process sequence!")
        
        w = np.ones(n_step)*self.x0
        
        for i in range(1,n_step):
            # Sampling from the Normal distribution with probability 1/2
            yi = np.random.choice([1,-1])
            # Weiner process
            w[i] = w[i-1]+(yi/np.sqrt(n_step))
        
        return w
    
    def gen_normal(self,n_step=100):
        """
        Generate motion by drawing from the Normal distribution
        
        Arguments:
            n_step: Number of steps
            
        Returns:
            A NumPy array with `n_steps` points
        """
        if n_step < 30:
            print("WARNING! The number of steps is small. It may not generate a good stochastic process sequence!")
        
        w = np.ones(n_step)*self.x0
        
        for i in range(1,n_step):
            # Sampling from the Normal distribution
            yi = np.random.normal()
            # Weiner process
            w[i] = w[i-1]+(yi/np.sqrt(n_step))
        
        return w
    
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


class PipeflowEquations(BaseModel):
    """Multiphase equation model class."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.L = basic_args['xmax'] # meters
        self.d = 0.2#0.146 # meters
        self.A = np.pi*self.d**2/4 # meters^2
        self.c = 308. # m/s
        self.rho_g_norm = 1.26 # kg/m^3
        self.rho_l = 1003. # kg/m^3
        self.p_amb = 1.01325e5 # Pa
        self.p_norm = 1.0e5 # Pa
        self.p_outlet = 1.0e6 # Pa
        self.e = 1e-8 # meters
        self.mu_g = 1.8e-5 # Pa*s
        self.mu_l = 1.516e-3 # Pa*s
        #self.Cd = 5e-4
        self.T_norm = 278 # Kelvin
        self.T = 278 # Kelvin

        self.leak = True

        self.D_orifice = 0.03 # meters
        self.A_orifice = np.pi*(self.D_orifice/2)**2
        self.leak_location = 3221.31
        self.Cd = 2.
        self.Cv = self.A/np.sqrt(self.rho_g_norm/2 * ((self.A/(self.A_orifice*self.Cd))**2-1)) #
        print(f'Cv: {self.Cv:.2E}')

        self.conservative_or_primitive = 'primitive'

        self.added_boundary_noise = 0.0
        self._t = 0.0

        self.xElementL = np.int32(self.leak_location / self.basic_args['xmax'] * self.DG_vars.K)

        self.lagrange = []
        l = np.zeros(self.DG_vars.N + 1)
        rl = 2 * (self.leak_location - self.DG_vars.VX[self.xElementL]) / self.DG_vars.deltax - 1
        for i in range(0, self.DG_vars.N + 1):
            l[i] = JacobiP(np.array([rl]), 0, 0, i)
        self.lagrange = np.linalg.solve(np.transpose(self.DG_vars.V), l)

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

        u_m = q[2]/rho_m/self.A

        return np.array([A_l, p/self.p_outlet, u_m])
    
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
    

    def jacobian(self, q):
        """Compute the Jacobian of the flux function."""

        A_l = q[0]
        p = q[1]*self.p_outlet
        u_m = q[2]

        return 2
    
    '''
    def eigen(self, q, outward_unit_normal=None):

        A_l = q[0]
        p = q[1]*self.p_outlet
        u_m = q[2]

        rho_g = self.pressure_to_density(p)
        A_g = self.A-A_l


        c_g_squared = rho_g * self.T_norm / self.rho_g_norm / self.T_norm

        A = np.zeros((self.DG_vars.num_states, self.DG_vars.num_states))
        A[0, :] = np.array([u_m, 0, A_l])
        A[1, :] = np.array([0, u_m, p*self.A/A_g])
        A[2, :] = np.array([0, c_g_squared*self.A/(A_l*self.rho_l*c_g_squared + A_g*p), u_m])


        if outward_unit_normal is not None:

            B = outward_unit_normal * A
            Lambda, R = eig(B, right=True)
            Lambda = Lambda.real

            Lambda_minus = np.diag(np.minimum(Lambda,0))
            Lambda_plus = np.diag(np.maximum(Lambda,0))

            R_inv = np.linalg.inv(R)
            B_minus = np.matmul(R, np.matmul(Lambda_minus, R_inv))
            B_plus = np.matmul(R, np.matmul(Lambda_plus, R_inv))

            return B_minus, B_plus, R
        else:
            Lambda, R = eig(A, right=True)
            Lambda = Lambda.real

            return Lambda, R
    '''
    '''
    def eigen(self, q):
        """Compute the eigenvalues and eigenvectors of the flux Jacobian."""
        
        A_l = q[0]
        p = q[1]*self.p_outlet
        u_m = q[2]

        rho_g = self.pressure_to_density(p)

        A_g = self.A - A_l

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
    '''

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
    
    def transform_matrices(
        self, 
        t: float, 
        q: np.ndarray = None,
        ):
        """Compute the conservative to primitive transform matrices."""

        A_l = q[0]
        p = q[1]*self.p_outlet
        u_m = q[2]

        rho_g = self.pressure_to_density(p)

        A_g = self.A - A_l

        alpha_l = A_l/self.A
        alpha_g = A_g/self.A

        #rho_m = rho_g * A_g + self.rho_l * A_l
        rho_m = rho_g * alpha_g + self.rho_l * alpha_l

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
        '''

        Lambda, S = eig(A, right=True, left=False)

        Lambda = np.diag(Lambda.real)

        S_inv = np.linalg.inv(S)
        '''


        return P, P_inv, A, S, S_inv, Lambda
    
    def start_solver(self):

        brownian = Brownian()

        window_size = 600
        self.inflow_boundary_noise = brownian.gen_normal(n_step=np.int64(self.t_final/self.step_size + window_size + 1))
        self.inflow_boundary_noise = moving_average(self.inflow_boundary_noise, n=window_size)
        self.inflow_boundary_noise = np.abs(self.inflow_boundary_noise)


        

    def BC_eqs(self, q, gas_mass_inflow, liquid_mass_inflow):

        A_l = q[0]
        p = q[1]*self.p_outlet
        u_m = q[2]

        rho_g = self.pressure_to_density(p)

        A_g = self.A - A_l

        gas_mass_flow = rho_g * A_g * u_m
        liquid_mass_flow = self.rho_l * A_l * u_m

        return np.array([gas_mass_flow - gas_mass_inflow, liquid_mass_flow - liquid_mass_inflow, 0.0])

    def boundary_conditions(self, t=0, q=None):
        """Compute the boundary conditions."""


        inflow_noise = self.inflow_boundary_noise[len(self.t_vec)]*15

        
        t_start = 10000000.
        t_end = 200000000.

        # a increases linearly from 0.2 to 0.4 over 10 seconds
        if t < t_end and t > t_start:
            t_ = t - t_start
            gas_mass_inflow = 0.2 + 0.02 * t_ / 10
        elif t > t_end:
            gas_mass_inflow = 0.4
        else:
            gas_mass_inflow = 0.2 + inflow_noise#+ self.added_boundary_noise# * np.sin(2*np.pi*t/50)

        gas_mass_inflow = 0.2# + inflow_noise
        liquid_mass_inflow = 20.0# + inflow_noise

        if len(q.shape) == 1:
            rho_g = self.pressure_to_density(q[1]*self.p_outlet)
        else:
            rho_g = self.pressure_to_density(q[1]*self.p_outlet)[0]
        _alpha_l  = liquid_mass_inflow/self.rho_l / (gas_mass_inflow/rho_g + liquid_mass_inflow/self.rho_l)
        _u_m = liquid_mass_inflow /(self.rho_l * self.A * _alpha_l)

        func = lambda q: self.BC_eqs(q, gas_mass_inflow, liquid_mass_inflow)
        q_guess = np.array([_alpha_l*self.A, self.density_to_pressure(rho_g)/1e6, _u_m])
        q_sol = fsolve(func, q_guess)

        A_l = q_sol[0]
        p = q_sol[1]*self.p_outlet
        u_m = q_sol[2]

        #print('A_l: ', A_l-_alpha_l*self.A, 'u_m: ', u_m-_u_m)

        if self.steady_state_solve or self.BC_args['treatment'] == 'naive':
            BC_state_1 = {
                'left': A_l,
                'right': None,
            }
            BC_state_2 = {
                'left': None,
                'right': 1. #+ outflow_noise#self.p_outlet,
            }
            BC_state_3 = {
                'left': u_m,#,
                'right': None
            }
        else:
            BC_state_1 = {
                'left': (q[0] - A_l) / self.step_size,
                'right': None,
            }
            BC_state_2 = {
                'left': None,
                'right': 0.,#(q[1] - 1.0) / self.step_size,
            }
            BC_state_3 = {
                'left': (q[2] - u_m) / self.step_size,
                'right': None
            }


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

        BCs_state = [BC_state_1, BC_state_2, BC_state_3]
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

        return np.maximum(np.abs(u_m + self.c), np.abs(u_m - self.c))

        '''

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
        '''
        
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

    def leakage(self, pressure=0, rho_m=0):
        """Compute leakage"""


        f_l = np.zeros((self.DG_vars.x.shape))

        pressureL = self.evaluate_solution(np.array([self.leak_location]), pressure)[0]
        rhoL = self.evaluate_solution(np.array([self.leak_location]), rho_m)[0]


        self.Cv = self.A/np.sqrt(rhoL/2 * ((self.A/(self.A_orifice*self.Cd))**2-1)) #

        discharge_sqrt_coef = (pressureL - self.p_amb) * rhoL
        f_l[:, self.xElementL] = self.Cv * np.sqrt(discharge_sqrt_coef) * self.lagrange
        f_l[:, self.xElementL] = self.DG_vars.invMk @ f_l[:, self.xElementL]

        return f_l
    
    def source(self, t, q):
        """Compute the source."""

        A_l = q[0]
        p = q[1]*self.p_outlet
        u_m = q[2]

        alpha_l = A_l/self.A
        alpha_g = 1 - alpha_l

        rho_g = self.pressure_to_density(p)
        rho_m = rho_g * alpha_g + self.rho_l * alpha_l
        
        s = np.zeros((self.DG_vars.num_states,self.DG_vars.Np*self.DG_vars.K))

        
        point_source = np.zeros((self.DG_vars.Np*self.DG_vars.K))
        if t>0.:
            '''
            x = self.DG_vars.x.flatten('F')
            width = 50
            point_source = \
                (np.heaviside(x-self.leak_location + width/2, 1) - np.heaviside(x-self.leak_location-width/2, 1))
            point_source *= 1/width

            leak_mass = self.Cv * np.sqrt(rho_m * (p - self.p_amb)) * point_source
            s[0] = -alpha_g * leak_mass
            s[1] = -alpha_l * leak_mass
            '''
            leak = self.leakage(pressure=p, rho_m=rho_m).flatten('F')
            s[0] = -alpha_g * leak
            s[1] = -alpha_l * leak

        s[-1] = -self.friction_factor(q)

        return s

if __name__ == '__main__':

    basic_args = {
        'xmin': 0,
        'xmax': 5000,
        'num_elements': 500,
        'num_states': 3,
        'polynomial_order': 2,
        'polynomial_type': 'legendre',
    }

    steady_state_args = {
        'newton_params': {
            'solver': 'direct',
            'max_newton_iter': 200,
            'newton_tol': 1e-5,
            'num_jacobian_reuses': 1000,
        }
    }

    numerical_flux_args = {
        'type': 'lax_friedrichs',
        'alpha': 0.5,
    }
    '''
    stabilizer_args = {
        'type': 'artificial_viscosity',
        'kappa': 5.,
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
        'filter_order': 36,
    }
    '''
    time_integrator_args = {
        'type': 'BDF2',
        'step_size': .01,
        'newton_params': {
            'solver': 'direct',
            'max_newton_iter': 100,
            'newton_tol': 1e-5,
            'num_jacobian_reuses': 2500,
        }
    }
    BC_args = {
        'type': 'dirichlet',
        'treatment': 'naive',
        'state_or_flux': {'left': 'flux', 'right': 'state'},
    }

    pipe_DG = PipeflowEquations(
        basic_args=basic_args,
        BC_args=BC_args,
        stabilizer_args=stabilizer_args,
        time_integrator_args=time_integrator_args,
        numerical_flux_args=numerical_flux_args,
    )
    
            
    init = pipe_DG.initial_condition(pipe_DG.DG_vars.x.flatten('F'))

    t_final = 180.0
    sol, t_vec = pipe_DG.solve(
        t=0, 
        q_init=init, 
        t_final=t_final, 
        steady_state_args=steady_state_args
    )


    num_steps_to_plot = 500
    if sol.shape[-1] > num_steps_to_plot:

        t_idx = np.linspace(0, sol.shape[-1]-1, num_steps_to_plot, dtype=int)

        sol = sol[:, :, t_idx]

    t_vec = np.arange(0, sol.shape[-1])

    x = np.linspace(0, basic_args['xmax'], 512)

    A_l = np.zeros((len(x), len(t_vec)))
    p = np.zeros((len(x), len(t_vec)))
    u_m = np.zeros((len(x), len(t_vec)))
    for t in range(sol.shape[-1]):
        A_l[:, t] = pipe_DG.evaluate_solution(x, sol_nodal=sol[0, :, t])
        p[:, t] = pipe_DG.evaluate_solution(x, sol_nodal=sol[1, :, t])
        u_m[:, t] = pipe_DG.evaluate_solution(x, sol_nodal=sol[2, :, t])
    
    
    u = u_m
    alpha_l = A_l/pipe_DG.A
    alpha_g = 1 - alpha_l

    
    rho_g = pipe_DG.pressure_to_density(p*pipe_DG.p_outlet)
    rho_g_A_g_u_m = u_m * rho_g * (pipe_DG.A - A_l)
    rho_l_A_l_u_m = u_m * pipe_DG.rho_l * A_l
    #print(rho_g_A_g_u_m[0, :])

    plt.figure()
    plt.plot(t_vec, rho_l_A_l_u_m[0, :])
    plt.show()

    
    t_vec = np.arange(0, u.shape[1]-1)

    plt.figure()
    plt.imshow(u, extent=[0, t_final, basic_args['xmax'], 0], aspect='auto')
    plt.colorbar()
    plt.show()

    plt.figure()
    #plt.plot(x, alpha_l[:, 0], label='alpha_l', linewidth=2)
    plt.plot(x, alpha_l[:, -1], label='alpha_l', linewidth=2)
    plt.plot(x, alpha_g[:, -1], label='alpha_g', linewidth=2)
    plt.plot(x, u[:, -1], label='u', linewidth=2)
    #plt.ylim(0.1, 0.75)
    #plt.ylim(18, 35)
    plt.grid()
    plt.legend()
    plt.savefig('pipeflow.png')
    plt.show()



 

    fig, ax = plt.subplots()
    xdata, ydata = [], []
    ln, = ax.plot([], [], lw=3, animated=True)

    def init():
        ax.set_xlim(0, basic_args['xmax'])
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
    ani.save('pipeflow_velocity.gif', fps=30)
    plt.show()


    fig, ax = plt.subplots()
    xdata, ydata, ydata_1 = [], [], []
    ln, = ax.plot([], [], lw=3, animated=True)
    ln_1, = ax.plot([], [], lw=3, animated=True)

    def init():
        ax.set_xlim(0, basic_args['xmax'])
        ax.set_ylim(p.min(), p.max())
        #ax.set_ylim(0.1, 0.9)
        #ax.set_ylim(18, 35)
        return ln,

    def update(frame):
        xdata.append(x)
        ydata.append(p[:, frame])
        #ydata_1.append(alpha_g[:, frame])
        ln.set_data(x, p[:, frame])
        #ln_1.set_data(x, alpha_g[:, frame])
        return ln, #ln_1,

    ani = FuncAnimation(
        fig,
        update,
        frames=t_vec,
        init_func=init, 
        blit=True,
        interval=10,
        )
    ani.save('pipeflow_pressure.gif', fps=30)
    plt.show()
    


    fig, ax = plt.subplots()
    xdata, ydata, ydata_1 = [], [], []
    ln, = ax.plot([], [], lw=3, animated=True)
    ln_1, = ax.plot([], [], lw=3, animated=True)

    def init():
        ax.set_xlim(0, basic_args['xmax'])
        ax.set_ylim(0.0, 1.0)
        #ax.set_ylim(0.1, 0.9)
        #ax.set_ylim(18, 35)
        return ln,

    def update(frame):
        xdata.append(x)
        ydata.append(alpha_l[:, frame])
        ydata_1.append(alpha_g[:, frame])
        ln.set_data(x, alpha_l[:, frame])
        ln_1.set_data(x, alpha_g[:, frame])
        return ln, #ln_1,

    ani = FuncAnimation(
        fig,
        update,
        frames=t_vec,
        init_func=init, 
        blit=True,
        interval=10,
        )
    ani.save('pipeflow_holdup.gif', fps=30)
    plt.show()