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

        # Initialize the numerical flux
        self.numerical_flux = factories.get_numerical_flux(
            DG_vars=self.DG_vars,
            numerical_flux_args=numerical_flux_args,
            system_jacobian=self.system_jacobian,
            eigen=self.eigen,
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
            conservative_to_primitive_transform_matrix=self.conservative_to_primitive_transform_matrix,
        )
        
        # Initialize the time integrator
        self.step_size = time_integrator_args.get('step_size')
        time_integrator_args.pop('step_size')

        self.time_integrator = factories.get_time_integrator(
            DG_vars=self.DG_vars,
            time_integrator_args=time_integrator_args,
            stabilizer=self.stabilizer,
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
        
        # Compute the numerical flux
        numerical_flux = self.numerical_flux(
            q_inside=q[:, self.DG_vars.vmapM], 
            q_outside=q[:, self.DG_vars.vmapP],
            flux_inside=flux[:, self.DG_vars.vmapM],
            flux_outside=flux[:, self.DG_vars.vmapP],
            )
            
        # Compute the source term
        source = self.source(t, q)

        # Compute boundary conditions
        q_boundary=q[:, [self.DG_vars.vmapI, self.DG_vars.vmapO]]
        flux_boundary=flux[:, [self.DG_vars.vmapI, self.DG_vars.vmapO]]
        if self.BC_args['treatment'] == 'characteristic':
            q_boundary_diff = np.zeros((self.DG_vars.num_states, 2))
            q_boundary_diff[:, 0] = q[:, 1] - q[:, 0]
            q_boundary_diff[:, 1] = q[:, -2] - q[:, -1]
            q_boundary_diff /= self.DG_vars.dx
        else:
            q_boundary_diff = None


        numerical_flux[:, self.DG_vars.mapI], numerical_flux[:, self.DG_vars.mapO] = \
             self.BCs.apply_boundary_conditions(
                t=t, 
                q_boundary=q_boundary,
                flux_boundary=flux_boundary,
                q_boundary_diff=q_boundary_diff,
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
        sol = []

        # Compute the steady state solution
        if steady_state_args is not None:
            q_init = compute_steady_state(
                q=q_init,
                rhs=self.compute_rhs,
                newton_params = steady_state_args['newton_params'],
                DG_vars=self.DG_vars,
            )

            q_init = self.stabilizer(q_init)
        
        # Set initial condition
        sol.append(q_init)

        t_vec = [t]

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
                    CFL=.1
                    )

            if t + self.step_size - 1e-1 > t_final:
                self.step_size = t_final - t
            
            sol_, t = self.time_integrator(
                t=t_vec[-1], 
                q=sol[-1],
                step_size=self.step_size,
                rhs=self.compute_rhs
                )
            
            t_vec.append(t)

            sol.append(sol_)

            if print_progress:
                pbar.set_postfix({'':f'{t:.2f}/{t_final:.2f}'})

                pbar.update(self.step_size)
        if print_progress:        
            pbar.close()
        
        return np.stack(sol, axis=-1) , t_vec
        

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
            
        i_interface = np.where(x == self.DG_vars.VX)
        sol_xVec[i_interface] = 0.5*(sol_nodal[i_interface[0]-1,-1]+sol_nodal[i_interface[0],0])

        return sol_xVec