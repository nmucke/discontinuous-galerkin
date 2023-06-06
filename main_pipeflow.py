import time
import numpy as np
from discontinuous_galerkin.base.base_model import BaseModel
from discontinuous_galerkin.start_up_routines.start_up_1D import StartUp1D
import matplotlib.pyplot as plt
import pdb

from scipy.optimize import fsolve

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

        point_source = np.zeros((self.DG_vars.Np*self.DG_vars.K))
        if t>0:
            x = self.DG_vars.x.flatten('F')
            width = 50
            point_source = \
                (np.heaviside(x-500 + width/2, 1) - np.heaviside(x-500-width/2, 1))
            point_source *= 1/width

        rho = q[0]/self.A
        p = self.density_to_pressure(rho)

        s[0] = - self.Cd * np.sqrt(rho * (p - self.p_amb)) * point_source
        s[0] *= 0.
        s[1] = -self.friction_factor(q)

        return s
    

class PipeNetwork():
    def __init__(
        self,
        basic_args,
        numerical_flux_args,
        stabilizer_args,
        time_integrator_args,
        BC_args
    ):
        super().__init__()

        self.step_size = time_integrator_args['step_size']
        self.pipe_DG_1 = PipeflowEquations(
            basic_args=basic_args,
            BC_args=BC_args,
            stabilizer_args=stabilizer_args,
            time_integrator_args=time_integrator_args,
            numerical_flux_args=numerical_flux_args,
        )
        time_integrator_args['step_size'] = self.step_size
        self.pipe_DG_2 = PipeflowEquations(
            basic_args=basic_args,
            BC_args=BC_args,
            stabilizer_args=stabilizer_args,
            time_integrator_args=time_integrator_args,
            numerical_flux_args=numerical_flux_args,
        )


    def set_BCs(self, q, BCs):
        '''
        self.pipe_DG_1.update_BCs(
            t=0, 
            q_left=np.array([None, q[0][-1][0, 0]*4.5]),
            q_right=np.array([q[1][-1][0, 0], None])
        )
        self.pipe_DG_2.update_BCs(
            t=0, 
            q_left=np.array([None, q[0][-1][1, -1]]),
            q_right=np.array([self.pipe_DG_2.rho_ref*self.pipe_DG_2.A, None])
        )
        self.pipe_DG_1.update_BCs(
            t=0, 
            q_left=np.array([None, q[0][-1][0, 0]*4.5]),
            q_right=np.array([q[1][-1][0, 0], q[1][-1][1, 0]])
        )
        self.pipe_DG_2.update_BCs(
            t=0, 
            q_left=np.array([q[0][-1][0, -1], q[0][-1][1, -1]]),
            q_right=np.array([self.pipe_DG_2.rho_ref*self.pipe_DG_2.A, None])
        )
        '''
        self.pipe_DG_1.update_BCs(
            t=0, 
            q_left=np.array([None, q[0][-1][0, 0]*4.5]),
            q_right=np.array([None, None])
        )
        self.pipe_DG_2.update_BCs(
            t=0, 
            q_left=np.array([None, None]),
            q_right=np.array([self.pipe_DG_2.rho_ref*self.pipe_DG_2.A, None])
        )

    def set_external_state(self, q):

        self.pipe_DG_1.set_external_state(
            q_left=None,
            q_right=q[1],
        )
        self.pipe_DG_2.set_external_state(
            q_left=q[0],
            q_right=None,
        )

    def solve(self, t, t_final):


        q_init = self.pipe_DG_1.initial_condition(self.pipe_DG_1.DG_vars.x.flatten('F'))
        
        t_final = 5.0
        t = 0

        sol = {}

        sol[0] = [q_init]
        sol[1] = [q_init]

        sol1 = np.expand_dims(q_init, axis=2)
        sol2 = np.expand_dims(q_init, axis=2)

        xxx = []
        sss = []

        for i in range(0, 500):

            t_next = t + self.step_size

            for j in range(1):
                P, P_inv, A, S, S_inv, Lambda = self.pipe_DG_1.transform_matrices(t, sol1[:, -1, -1])

                prim_state = np.array(
                    [
                    self.pipe_DG_1.density_to_pressure(sol1[0, -self.pipe_DG_1.DG_vars.Np:, -1]/self.pipe_DG_1.A), 
                    sol1[1, -self.pipe_DG_1.DG_vars.Np:, -1]/sol1[0, -self.pipe_DG_1.DG_vars.Np:, -1]
                    ]
                )
                prim_state *= self.pipe_DG_1.A
                dpdx = (self.pipe_DG_1.DG_vars.Dr @ prim_state[0])[-1]
                dudx = (self.pipe_DG_1.DG_vars.Dr @ prim_state[1])[-1]

                prim_state_dx = np.array([dpdx, dudx])

                #Lambda = np.diag(Lambda)
                for iter, ll in enumerate(np.diag(Lambda)):
                    if ll > 0:
                        lambda_plus = ll
                        i_plus = iter
                    elif ll < 0:
                        lambda_minus = ll
                        i_minus = iter
                Lambda_minus = Lambda.copy()
                Lambda_plus = Lambda.copy()
                Lambda_minus[i_minus, i_minus] = 0
                Lambda_plus[i_plus, i_plus] = 0

                A_minus = S @ Lambda_minus @ S_inv
                A_plus = S @ Lambda_plus @ S_inv

                dprim_dt = A_plus @ prim_state_dx
                dqdt = - P @ dprim_dt - self.pipe_DG_1.source(t, sol1[:, -1, -1])[:, -1]


                func = lambda q: q - sol1[:, -1, -1] - self.step_size * dqdt 
                q1_new = fsolve(func, sol1[:, -1, -1])


                P, P_inv, A, S, S_inv, Lambda = self.pipe_DG_2.transform_matrices(t, sol2[:, 0, -1])

                prim_state = np.array(
                    [
                    self.pipe_DG_2.density_to_pressure(sol2[0, 0:self.pipe_DG_2.DG_vars.Np, -1]/self.pipe_DG_2.A), 
                    sol2[1, 0:self.pipe_DG_2.DG_vars.Np, -1]/sol2[0, 0:self.pipe_DG_2.DG_vars.Np, -1]
                    ]
                )
                prim_state *= self.pipe_DG_2.A
                dpdx = (self.pipe_DG_1.DG_vars.Dr @ prim_state[0])[-1]
                dudx = (self.pipe_DG_1.DG_vars.Dr @ prim_state[1])[-1]

                prim_state_dx = np.array([dpdx, dudx])

                #Lambda = np.diag(Lambda)
                for iter, ll in enumerate(np.diag(Lambda)):
                    if ll > 0:
                        lambda_plus = ll
                        i_plus = iter
                    elif ll < 0:
                        lambda_minus = ll
                        i_minus = iter
                Lambda_minus = Lambda.copy()
                Lambda_plus = Lambda.copy()
                Lambda_minus[i_minus, i_minus] = 0
                Lambda_plus[i_plus, i_plus] = 0

                A_minus = S @ Lambda_minus @ S_inv
                A_plus = S @ Lambda_plus @ S_inv

                dprim_dt = A_minus @ prim_state_dx
                dqdt = - P @ dprim_dt - self.pipe_DG_1.source(t, sol2[:, 0, -1])[:, 0]


                func = lambda q: q - sol2[:, 0, -1] - self.step_size * dqdt
                q2_new = fsolve(func, sol2[:, 0, -1])
                
                '''

                #L = S_inv @ A @ np.array([dpdx, dudx])
                L_plus = Lambda @ S_inv[i_plus, :] @ np.array([dpdx, dudx])
                
                dqdt = - P @ S[:, i_plus] * L_plus - self.pipe_DG_1.source(t, sol1[:, -1, -1])[:, -1]
                
                func = lambda q: q - sol1[:, -1, -1] - self.step_size * dqdt
                q1_new = fsolve(func, sol1[:, -1, -1])
                #q1_new = sol1[:, -1, -1] + self.step_size * dqdt

                #if Lambda[0, 0] < 0:
                #    q1_new[0] = sol1[0, -1, -1]
                #if Lambda[1, 1] < 0:
                #    q1_new[1] = sol1[1, -1, -1]

                P, P_inv, A, S, S_inv, Lambda = self.pipe_DG_2.transform_matrices(t, sol2[:, 0, -1])

                prim_state = np.array(
                    [
                    self.pipe_DG_2.density_to_pressure(sol2[0, 0:self.pipe_DG_2.DG_vars.Np, -1]/self.pipe_DG_2.A), 
                    sol2[1, 0:self.pipe_DG_2.DG_vars.Np, -1]/sol2[0, 0:self.pipe_DG_2.DG_vars.Np, -1]
                    ]
                )
                prim_state *= self.pipe_DG_2.A
                dpdx = (self.pipe_DG_1.DG_vars.Dr @ prim_state[0])[-1]
                dudx = (self.pipe_DG_1.DG_vars.Dr @ prim_state[1])[-1]

                #Lambda = np.diag(Lambda)
                for iter, ll in enumerate(np.diag(Lambda)):
                    if ll > 0:
                        lambda_plus = ll
                        i_plus = iter
                    elif ll < 0:
                        lambda_minus = ll
                        i_minus = iter
                Lambda[i_plus, i_plus]
                L_minus = Lambda @ S_inv[i_minus, :] @ np.array([dpdx, dudx])
                
                dqdt = - P @ S[:, i_minus] * L_minus - self.pipe_DG_1.source(t, sol1[:, -1, -1])[:, -1]

                #L1 = self.pipe_DG_2.DG_vars.Dr @ prim_state[0]
                #L2 = self.pipe_DG_2.DG_vars.Dr @ prim_state[1]
                #L = np.array([L1[0], L2[0]])
                #L = Lambda @ S_inv @ L

                #dqdt = - P @ S @ L - self.pipe_DG_2.source(t, sol2[:, 0, -1])[:, 0]
                
                func = lambda q: q - sol2[:, 0, -1] - self.step_size * dqdt
                q2_new = fsolve(func, sol2[:, 0, -1])

                #q2_new = sol2[:, 0, -1] + self.step_size * dqdt

                #if Lambda[0, 0] > 0:
                #    q2_new[0] = sol2[0, 0, -1]
                #if Lambda[1, 1] > 0:
                #    q2_new[1] = sol2[1, 0, -1]
                '''

                '''
                pipe_1_numerical_flux = self.pipe_DG_1.numerical_flux(
                    q_inside=sol1[:, -1, -1],
                    q_outside=sol2[:, 0, -1],
                    flux_inside=self.pipe_DG_1.flux(sol1[:, -1, -1]),
                    flux_outside=self.pipe_DG_1.flux(sol2[:, 0, -1]),
                    on_boundary=True
                )
                '''

                #q1_new = sol1[:, -1, -1]
                #q2_new = sol2[:, 0, -1]

                #self.set_BCs(q=sol, BCs=None)
                self.set_BCs(q=sol, BCs=None)

                #self.set_BCs(q=sol, BCs=None)
                #self.set_BCs(q=[sol1, sol2], BCs=None)
                self.set_external_state(q=[q1_new, q2_new])

                sol1, t_vec = self.pipe_DG_1.solve(
                    t=t, 
                    q_init=sol[0][-1], 
                    t_final=t_next, 
                    print_progress=False,
                    steady_state_args=None,
                    external_state={'left': False, 'right': True}
                )
                '''
                pipe_1_numerical_flux = self.pipe_DG_1.numerical_flux(
                    q_inside=sol1[:, -1:, -1],
                    q_outside=sol2[:, 0:1, -1],
                    flux_inside=self.pipe_DG_1.flux(sol1[:, -1:, -1]),
                    flux_outside=self.pipe_DG_1.flux(sol2[:, 0:1, -1]),
                    on_interface='right'
                )
                '''

                #self.set_BCs(q=sol, BCs=None)

                #self.set_external_state(q=[sol1[:, -1, -1], sol2[:, 0, -1]])

                sol2, t_vec = self.pipe_DG_2.solve(
                    t=t, 
                    q_init=sol[1][-1], 
                    t_final=t_next, 
                    print_progress=False,
                    steady_state_args=None,
                    external_state={'left': True, 'right': False}
                )

                '''
                pipe_2_numerical_flux = self.pipe_DG_2.numerical_flux(
                    q_inside=sol2[:, 0:1, -1],
                    q_outside=sol1[:, -1:, -1],
                    flux_inside=self.pipe_DG_2.flux(sol2[:, 0:1, -1]),
                    flux_outside=self.pipe_DG_2.flux(sol1[:, -1:, -1]),
                    on_interface='left'
                )
                '''
                #print(pipe_1_numerical_flux-pipe_2_numerical_flux)

                #BC_err = np.abs(np.sum(sol1[:, -1, -1] - sol2[:, 0, -1]))
                #BC_err = np.linalg.norm(pipe_1_numerical_flux-pipe_2_numerical_flux)
                #if BC_err < 1e-10:
                #    break
                #else:
                #    if j % 10 == 0:
                #        print(f'BC error: {BC_err}, iteration: {j}')
            print(i)

            sol[0].append(sol1[:, :, -1])
            sol[1].append(sol2[:, :, -1])

            t = t_next

            xx = np.concatenate((self.pipe_DG_1.DG_vars.x.flatten('F'), self.pipe_DG_1.DG_vars.x.flatten('F')[-1]+self.pipe_DG_2.DG_vars.x.flatten('F')))
            ss = np.concatenate((sol1[1, :, -1]/sol1[0, :, -1], sol2[1, :, -1]/sol2[0, :, -1]))

            xxx.append(xx)
            sss.append(ss)

        xxx = np.array(xxx)[0]
        sss = np.array(sss)

        fig, ax = plt.subplots()
        xdata, ydata = [], []
        ln, = ax.plot([], [], lw=3, animated=True)

        t_vec = np.arange(0, sss.shape[0])

        def init():
            ax.set_xlim(0, 4000)
            ax.set_ylim(sss.min() - .3, sss.max() + .3)
            return ln,

        def update(frame):
            xdata.append(xxx)
            ydata.append(sss[frame, :])
            ln.set_data(xxx, sss[frame, :])
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


        plt.plot(xxx, sss[-1])
        plt.show()



if __name__ == '__main__':
    


    basic_args = {
        'xmin': 0,
        'xmax': 2000,
        'num_elements': 100,
        'num_states': 2,
        'polynomial_order': 3,
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
        'type': 'filter',
        'num_modes_to_filter': 20,
        'filter_order': 6,
    }
    '''
    stabilizer_args = {
        'type': 'slope_limiter',
        'second_derivative_upper_bound': 1e-8,
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
        'form': {'left': 'primitive', 'right': 'primitive'},
    }


    #polynomial_order=8
    num_elements = 150
    polynomial_order = 2

    pipe_DG = PipeNetwork(
        basic_args=basic_args,
        BC_args=BC_args,
        stabilizer_args=stabilizer_args,
        time_integrator_args=time_integrator_args,
        numerical_flux_args=numerical_flux_args,
    )


    t_final = 5.0
    sol, t_vec = pipe_DG.solve(
        t=0, 
        t_final=t_final, 
    )

    x = np.linspace(0, 2000, 2000)

    u = np.zeros((len(x), len(t_vec)))
    rho = np.zeros((len(x), len(t_vec)))
    for t in range(sol.shape[-1]):
        rho[:, t] = pipe_DG.evaluate_solution(x, sol_nodal=sol[0, :, t])
        u[:, t] = pipe_DG.evaluate_solution(x, sol_nodal=sol[1, :, t])
    rho = rho / pipe_DG.A
    u = u / rho / pipe_DG.A

    print(rho[0, :])
    
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