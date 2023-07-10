import numpy as np
import matplotlib.pyplot as plt
import pdb

from abc import abstractmethod
from tqdm import tqdm
from discontinuous_galerkin.polynomials.jacobi_polynomials import JacobiP

from discontinuous_galerkin.start_up_routines.start_up_1D import StartUp1D
import discontinuous_galerkin.factories as factories
from discontinuous_galerkin.time_integrators.CFL import get_CFL_step_size
from discontinuous_galerkin.steady_state import compute_steady_state 
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

#class BaseModel(StartUp1D, Stabilizer, NumericalFlux):

class BaseModel():
    """
    Base class for all models.

    This class contains the basic functionality for all models. It is not
    intended to be used directly, but rather as a base class for other models.
    """

    def __init__(
        self, 
        basic_args,
        BC_args={'type': 'dirichlet'},
        stabilizer_args=None,
        time_integrator_args=None,
        numerical_flux_args=None,
        ):
        """Initialize base model class.""" 

        self.basic_args = basic_args
        self.BC_args = BC_args
        self.time_integrator_args = time_integrator_args
        self.numerical_flux_params = numerical_flux_args
        self.stabilizer_args = stabilizer_args

        # Initialize the start-up routine        
        self.DG_vars = StartUp1D(
            **basic_args,
            )

        # Initialize the stabilizer
        self.stabilizer = factories.get_stabilizer(
            DG_vars=self.DG_vars,
            stabilizer_args=stabilizer_args,
        )

        # Check if the eigenvalues and eigenvectors are implemented
        if getattr(self.eigen, '__isabstractmethod__', False):
            self.eigen = None
            
        # Check if the system Jacobian is implemented
        if getattr(self.system_jacobian, '__isabstractmethod__', False):
            self.system_jacobian = None
        
        # Check if primitive_to_conservative is implemented
        if getattr(self.primitive_to_conservative, '__isabstractmethod__', False):
            self.primitive_to_conservative = None
        
        # Check if conservative_to_primitive is implemented
        if getattr(self.conservative_to_primitive, '__isabstractmethod__', False):
            self.conservative_to_primitive = None
        
        # Initialize the numerical flux
        self.numerical_flux = factories.get_numerical_flux(
            DG_vars=self.DG_vars,
            numerical_flux_args=numerical_flux_args,#, 'alpha': 0.5},#
            system_jacobian=self.system_jacobian,
            eigen=self.eigen,
            velocity=self.velocity,
            primitive_to_conservative=self.primitive_to_conservative,
            conservative_to_primitive=self.conservative_to_primitive,
        )
        
        # Initialize the boundary conditions
        self.BCs = factories.get_boundary_conditions(
            DG_vars=self.DG_vars,
            BC_args=BC_args,
            numerical_flux=self.numerical_flux,
            boundary_conditions=self.boundary_conditions,
            flux=self.flux,
            system_jacobian=self.system_jacobian,
            source=self.source,
            transform_matrices=self.transform_matrices,
            primitive_to_conservative=self.primitive_to_conservative,
            conservative_to_primitive=self.conservative_to_primitive,
            eigen=self.eigen,
        )

        # initialize steady state BCs
        self.steady_state_BCs = factories.get_boundary_conditions(
            DG_vars=self.DG_vars,
            BC_args={
                'type': 'dirichlet',
                'treatment': 'naive',
                'state_or_flux': {'left': 'state', 'right': 'state'},
            },
            numerical_flux=self.numerical_flux,
            boundary_conditions=self.boundary_conditions,
            flux=self.flux,
            system_jacobian=self.system_jacobian,
            source=self.source,
            transform_matrices=self.transform_matrices,
        )
        
        # Initialize the time integrator
        if time_integrator_args.get('step_size') is not None:
            self.step_size = time_integrator_args.get('step_size')
            time_integrator_args.pop('step_size')
        else:
            self.step_size = 0.00001

        self.time_integrator = factories.get_time_integrator(
            DG_vars=self.DG_vars,
            time_integrator_args=time_integrator_args,
            stabilizer=self.stabilizer,
            primitive_to_conservative=self.primitive_to_conservative,
            conservative_to_primitive=self.conservative_to_primitive,
        )
        if self.time_integrator_args['type'] == 'BDF2':
            self.init_time_integrator_args = time_integrator_args.copy()
            self.init_time_integrator_args['type'] = 'implicit_euler'
            self.init_time_integrator = factories.get_time_integrator(
                DG_vars=self.DG_vars,
                time_integrator_args=self.init_time_integrator_args,
                stabilizer=self.stabilizer,
                primitive_to_conservative=self.primitive_to_conservative,
                conservative_to_primitive=self.conservative_to_primitive,
            )


    def __str__(self):
        """ Description of the model. """

        output = "BaseModel: \n"
        output += f"xmin: {self.DG_vars.xmin} \n"
        output += f"xmax: {self.DG_vars.xmax} \n"
        output += f"Number of elements: {self.DG_vars.K} \n"
        output += f"Polynomial order: {self.DG_vars.N} \n"
        #output += f"Polynomial type: {self.poly_type} \n"
        #output += f"Stabilizer: {self.stabilizer_type} \n"
        #output += f"Time stepping: {self.time_stepper} \n"
        
        return output

    def __repr__(self):
        return self.__str__()
    
    @abstractmethod
    def transform_matrices(self, q):
        """Compute the transform matrices."""

        raise NotImplementedError

    @abstractmethod
    def primitive_to_conservative(self, q):

        raise NotImplementedError

    
    @abstractmethod
    def conservative_to_primitive(self, q):

        raise NotImplementedError
    
    @abstractmethod
    def eigen(self, q):
        """Compute the eigenvalues and eigenvectors."""

        raise NotImplementedError
    
    @abstractmethod
    def system_jacobian(self, q):
        """Compute the eigenvalues and eigenvectors."""

        raise NotImplementedError

    @abstractmethod
    def flux(self, q):
        """Compute the flux."""

        raise NotImplementedError

    @abstractmethod
    def initial_condition(self, x):
        """Compute the initial condition."""

        raise NotImplementedError
    
    @abstractmethod
    def boundary_conditions(self, q):
        """Compute the boundary condition."""

        raise NotImplementedError
    
    @abstractmethod
    def source(self, q):
        """Compute the source term."""

        raise NotImplementedError

    @abstractmethod
    def update_parameters(self, **kwargs):

        raise NotImplementedError
    
    @abstractmethod
    def velocity(self, q):
        """Compute the wave speed."""

        raise NotImplementedError

    def compute_rhs(self, t, q):
        """Compute the right hand side of the discretized model."""
        
        # Compute the flux
        flux = self.flux(q)

        if self.stabilizer_args['type'] == 'artificial_viscosity':
            
            if self.primitive_to_conservative is not None:
                q_cons = self.primitive_to_conservative(q)
            else:
                q_cons = q
            epsilon_e = self.stabilizer.get_viscosity(q_cons)

            q_cons = q_cons.reshape(
                (self.DG_vars.num_states, self.DG_vars.Np, self.DG_vars.K), 
                order='F'
            )
            epsilon_e = np.expand_dims(epsilon_e, axis=0)
            epsilon_e = np.expand_dims(epsilon_e, axis=0)
            epsilon_e = np.tile(epsilon_e, (self.DG_vars.num_states, self.DG_vars.Np, 1))

            #epsilon_e = 1e2*np.ones((self.DG_vars.num_states, self.DG_vars.Np, self.DG_vars.K))
            
            viscosity_flux = q_cons

            viscosity_flux = viscosity_flux.reshape(
                (self.DG_vars.num_states, self.DG_vars.Np * self.DG_vars.K), 
                order='F'
            )
            q_cons = q_cons.reshape(
                (self.DG_vars.num_states, self.DG_vars.Np * self.DG_vars.K), 
                order='F'
            )

            viscosity_numerical_flux = self.numerical_flux._average_operator(
                viscosity_flux[:, self.DG_vars.vmapM],
                viscosity_flux[:, self.DG_vars.vmapP],
                )
                        
            d_viscosity_flux = self.DG_vars.nx * (viscosity_flux[:, self.DG_vars.vmapM] - viscosity_numerical_flux)

            # Reshape the flux and source terms
            d_viscosity_flux = d_viscosity_flux.reshape(
                (self.DG_vars.num_states, self.DG_vars.Nfp * self.DG_vars.Nfaces, self.DG_vars.K), 
                order='F'
                )
            viscosity_flux = viscosity_flux.reshape(
                (self.DG_vars.num_states, self.DG_vars.Np, self.DG_vars.K), 
                order='F'
                )
            
            # Compute the right hand side
            viscosity_term = \
                + np.sqrt(epsilon_e) * np.multiply(self.DG_vars.rx, self.DG_vars.Dr @ viscosity_flux) \
                - self.DG_vars.LIFT @ (np.multiply(self.DG_vars.Fscale, d_viscosity_flux))
            
            viscosity_term = viscosity_term.reshape(
                (self.DG_vars.num_states, self.DG_vars.Np * self.DG_vars.K), 
                order='F'
                )
            epsilon_e = epsilon_e.reshape(
                (self.DG_vars.num_states, self.DG_vars.Np * self.DG_vars.K), 
                order='F'
                )
                        
            flux -= np.sqrt(epsilon_e) * viscosity_term
        
        # Compute the numerical flux
        numerical_flux = self.numerical_flux(
            q_inside=q[:, self.DG_vars.vmapM], 
            q_outside=q[:, self.DG_vars.vmapP],
            flux_inside=flux[:, self.DG_vars.vmapM],
            flux_outside=flux[:, self.DG_vars.vmapP],
            primitive_to_conservative=self.primitive_to_conservative,
            )
            
        # Compute the source term
        source = self.source(t, q)
    
        # Compute boundary conditions
        if self.steady_state_solve:
            q_boundary=q[:, [self.DG_vars.vmapI, self.DG_vars.vmapO]]
            flux_boundary=flux[:, [self.DG_vars.vmapI, self.DG_vars.vmapO]]
            numerical_flux[:, self.DG_vars.mapI], numerical_flux[:, self.DG_vars.mapO] = \
                self.steady_state_BCs.apply_boundary_conditions(
                    t=t, 
                    q_boundary=q_boundary,
                    flux_boundary=flux_boundary,
                    step_size=self.step_size,
                    )
        if self.BC_args['treatment'] == 'naive' and not self.steady_state_solve:
            q_boundary=q[:, [self.DG_vars.vmapI, self.DG_vars.vmapO]]
            flux_boundary=flux[:, [self.DG_vars.vmapI, self.DG_vars.vmapO]]
            numerical_flux[:, self.DG_vars.mapI], numerical_flux[:, self.DG_vars.mapO] = \
                self.BCs.apply_boundary_conditions(
                    t=t, 
                    q_boundary=q_boundary,
                    flux_boundary=flux_boundary,
                    step_size=self.step_size,
                    )
        
        d_flux = self.DG_vars.nx * (flux[:, self.DG_vars.vmapM] - numerical_flux)

        # Reshape the flux and source terms
        d_flux = d_flux.reshape(
            (self.DG_vars.num_states, self.DG_vars.Nfp * self.DG_vars.Nfaces, self.DG_vars.K), 
            order='F'
            )
        flux = flux.reshape(
            (self.DG_vars.num_states, self.DG_vars.Np, self.DG_vars.K), 
            order='F'
            )
        source = source.reshape(
            (self.DG_vars.num_states, self.DG_vars.Np, self.DG_vars.K), 
            order='F'
            )
        
        
        # Compute the right hand side
        rhs = \
            - np.multiply(self.DG_vars.rx, self.DG_vars.Dr @ flux) \
            + self.DG_vars.LIFT @ (np.multiply(self.DG_vars.Fscale, d_flux)) \
            + source
        
        
        if self.BC_args['treatment'] == 'characteristic' and not self.steady_state_solve:
        
            BCs_left, BCs_right  = self.BCs.get_BC_rhs(
                t=t, 
                q=q, 
                source=source,
                #primitive_to_conservative=self.primitive_to_conservative
                )
            
            rhs[:, 0, 0] = BCs_left
            rhs[:, -1, -1] = BCs_right
            
        rhs = rhs.reshape(
            (self.DG_vars.num_states, self.DG_vars.Np * self.DG_vars.K), 
            order='F'
            )
        
        
        return rhs

    def solve(
        self,
        t, 
        q_init, 
        t_final, 
        print_progress=True,
        steady_state_args=None,
        ):
        """Solve the model.

        This method solves the model and returns the solution.
        """

        self.t_final = t_final


        brownian = Brownian()

        window_size = 600
        self.inflow_boundary_noise = brownian.gen_normal(n_step=np.int64(self.t_final/self.step_size + window_size + 1))
        self.inflow_boundary_noise = moving_average(self.inflow_boundary_noise, n=window_size)

        self.outflow_boundary_noise = brownian.gen_normal(n_step=np.int64(self.t_final/self.step_size + window_size + 1))
        self.outflow_boundary_noise = moving_average(self.outflow_boundary_noise, n=window_size)

        sol = []
        self.t_vec = []

        self.steady_state_solve = False
        # Compute the steady state solution
        if steady_state_args is not None:
            self.steady_state_solve = True
            q_init = compute_steady_state(
                q=q_init,
                rhs=self.compute_rhs,
                newton_params = steady_state_args['newton_params'],
                DG_vars=self.DG_vars,
            )

            q_init = self.stabilizer(q_init)

            self.steady_state_solve = False

        # Set initial condition
        sol.append(q_init)

        self.t_vec.append(t)

        if print_progress:
            pbar = tqdm(
                total=t_final,
                bar_format = "{desc}: {percentage:.2f}%|{bar:20}| {n:.2f}/{total_fmt} [{elapsed}<{remaining}]"#
                )
        while t < t_final:
            if self.time_integrator_args['type'] == 'SSPRK':
                self.step_size = get_CFL_step_size(
                    velocity=self.velocity(sol[-1]), 
                    min_dx=self.DG_vars.dx, 
                    CFL=.9
                    )

            if t + self.step_size - 1e-1 > t_final:
                self.step_size = t_final - t
            
            if self.time_integrator_args['type'] == 'BDF2':
                if len(sol) < 2:
                    sol_, t = self.init_time_integrator(
                    t=self.t_vec[-1], 
                    q=sol[-1],
                    step_size=self.step_size,
                    rhs=self.compute_rhs
                    )
                else:
                    sol_, t = self.time_integrator(
                        t=self.t_vec[-1], 
                        q=sol[-2:],
                        step_size=self.step_size,
                        rhs=self.compute_rhs
                        )
            else:
                sol_, t = self.time_integrator(
                    t=self.t_vec[-1], 
                    q=sol[-1],
                    step_size=self.step_size,
                    rhs=self.compute_rhs
                    )
            
            self.t_vec.append(t)

            sol.append(sol_)

            if print_progress:
                pbar.set_postfix({'':f'{t:.2f}/{t_final:.2f}'})

                pbar.update(self.step_size)
        if print_progress:        
            pbar.close()
        
        return np.stack(sol, axis=-1) , self.t_vec
        

    def evaluate_solution(self, x, sol_nodal):
        """Evaluate the solution at the given points."""

        sol_nodal = sol_nodal.reshape(
            (self.DG_vars.Np, self.DG_vars.K), 
            order='F'
            )
        sol_modal = np.dot(self.DG_vars.invV, sol_nodal)

        interval_indices = np.searchsorted(self.DG_vars.VX, x, side='left')
        if x[0] == self.DG_vars.xmin:
            interval_indices[1:] = interval_indices[1:] - 1
        else:
            interval_indices = interval_indices - 1

        VX_repeat = self.DG_vars.VX[interval_indices]
        x_ref = 2*(x-VX_repeat)/self.DG_vars.deltax - 1
        sol_modal_repeat = sol_modal[:, interval_indices]

        P = np.zeros((self.DG_vars.Np, x.shape[0]))
        for i in range(self.DG_vars.Np):
            P[i, :] = JacobiP(x_ref, 0, 0, i)
        
        sol_xVec = np.sum(P*sol_modal_repeat, axis=0)
       
        if x[0] == self.DG_vars.xmin:
            sol_xVec[0] = sol_nodal[0,0]
        if x[-1] == self.DG_vars.xmax:
            sol_xVec[-1] = sol_nodal[-1, -1]
            
        #i_interface = np.where(x == self.DG_vars.VX)[0]
        #sol_xVec[i_interface] = 0.5*(sol_nodal[-1, i_interface[0]-1]+sol_nodal[0, i_interface[0]])

        return sol_xVec