import time
import numpy as np
from discontinuous_galerkin.base.base_model import BaseModel
from discontinuous_galerkin.start_up_routines.start_up_1D import StartUp1D
import matplotlib.pyplot as plt
import pdb

from scipy.optimize import fsolve

from matplotlib.animation import FuncAnimation

class SinglePipe(BaseModel):
    """Advection equation model class."""

    def __init__(
        self, PDE_params, **kwargs):
        super().__init__(**kwargs)

        self.L = PDE_params['L']
        self.d = PDE_params['d']
        self.A = np.pi*self.d**2/4
        self.c = PDE_params['c']
        self.rho_ref = PDE_params['rho_ref']
        self.p_amb = PDE_params['p_amb']
        self.p_ref = self.rho_ref*self.c**2
        self.e = PDE_params['e']
        self.mu = PDE_params['mu']


        if PDE_params.get('leak') is not None:
            self.leak = True

            self.D_orifice = PDE_params['leak']['D_orifice']
            self.A_orifice = np.pi*(self.D_orifice/2)**2
            self.leak_location = PDE_params['leak']['location']
            self.Cd = PDE_params['leak']['Cd']
            self.Cv = self.A/np.sqrt(self.rho_ref/2 * ((self.A/(self.A_orifice*self.Cd))**2-1))
        else: 
            self.leak = False
            

        if kwargs.get('init_BCs') is not None:
            q_left = kwargs.get('init_BCs')['left']
            q_right = kwargs.get('init_BCs')['right']

            self.BC_state_1 = {
                'left': q_left[0],
                'right': q_right[0]
            }

            self.BC_state_2 = {
                'left': q_left[1],
                'right': q_right[1]
            }

    def update_leak(self, Cd, leak_location):

        self.Cv = self.A/np.sqrt(self.rho_ref/2 * ((self.A/(self.A_orifice*Cd))**2-1))
        self.leak_location = leak_location

    def density_to_pressure(self, rho):
        """Compute the pressure from the density."""
        return self.c**2*(rho - self.rho_ref) + self.p_ref


    def pressure_to_density(self, p):
        """Compute the density from the pressure."""

        return (p - self.p_ref)/self.c**2 + self.rho_ref
    

    def transform_matrices(self, t, q):
        """Compute the conservative to primitive transform matrices."""

        rho = q[0]/self.A
        u = q[1]/q[0]

        P = np.zeros((self.DG_vars.num_states, self.DG_vars.num_states))
        P_inv = np.zeros((self.DG_vars.num_states, self.DG_vars.num_states))
        A = np.zeros((self.DG_vars.num_states, self.DG_vars.num_states))
        S = np.zeros((self.DG_vars.num_states, self.DG_vars.num_states))
        S_inv = np.zeros((self.DG_vars.num_states, self.DG_vars.num_states))
        Lambda = np.zeros((self.DG_vars.num_states, self.DG_vars.num_states))

        P[0, 0] = 1/self.c/self.c# * self.A
        P[1, 0] = 1/self.c/self.c * u# * self.A
        P[0, 1] = 0
        P[1, 1] = rho#*self.A

        P_inv[0, 0] = self.c*self.c#/self.A
        P_inv[1, 0] = -u/rho#/self.A
        P_inv[0, 1] = 0
        P_inv[1, 1] = 1/rho#/self.A

        A[0, 0] = u
        A[1, 0] = 1/rho
        A[0, 1] = self.c*self.c*rho
        A[1, 1] = u

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

        return P, P_inv, A, S, S_inv, Lambda
    
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
    
    def update_BCs(self, t, q_left, q_right):

        self.BC_state_1 = {
            'left': q_left[0],
            'right': q_right[0]
        }

        self.BC_state_2 = {
            'left': q_left[1],
            'right': q_right[1]
        }

    def set_external_state(self, q_left=None, q_right=None):
        '''
        if q_left is None:
            self.q_left = np.zeros((self.DG_vars.num_states, 2))
            self.flux_left = np.zeros((self.DG_vars.num_states, 2))
        if q_right is None:
            self.q_right = np.zeros((self.DG_vars.num_states, 2))
            self.flux_right = np.zeros((self.DG_vars.num_states, 2))
        '''

        if q_left is not None:
            self.q_left = q_left#np.stack((q_left, q_left), axis=1)
            self.flux_left = self.flux(np.expand_dims(q_left, axis=1))[:, 0]#np.concatenate((self.flux(np.expand_dims(q_left, axis=1)), np.expand_dims(q_left, axis=1)), axis=1)
        if q_right is not None:
            self.q_right = q_right#np.stack((q_right, q_right), axis=1)
            self.flux_right = self.flux(np.expand_dims(q_right, axis=1))[:, 0]#np.concatenate((self.flux(np.expand_dims(q_right, axis=1)), self.flux(np.expand_dims(q_right, axis=1))), axis=1)

            
    
    def boundary_conditions(self, t, q=None):
        """Compute the boundary conditions."""
        
        BC_state_1 = {
            'left': self.BC_state_1['left'],
            'right': self.BC_state_1['right']
        }

        BC_state_2 = {
            'left': self.BC_state_2['left'],
            'right': self.BC_state_2['right']
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

        if self.leak:
            point_source = np.zeros((self.DG_vars.Np*self.DG_vars.K))
            if t>0:
                x = self.DG_vars.x.flatten('F')
                width = 50
                point_source = \
                    (np.heaviside(x-500 + width/2, 1) - np.heaviside(x-500-width/2, 1))
                point_source *= 1/width

            rho = q[0]/self.A
            p = self.density_to_pressure(rho)
            
            s[0] = - self.Cv * np.sqrt(rho * (p - self.p_amb)) * point_source
        
        s[1] = -self.friction_factor(q)

        return s
    