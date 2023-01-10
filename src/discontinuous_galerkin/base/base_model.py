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
        xmin=0.,
        xmax=1.,
        num_elements=10,
        polynomial_order=5,
        polynomial_type='legendre',
        num_states=1,
        BC_params={'type': 'dirichlet'},
        steady_state=None,
        stabilizer_type=None, 
        stabilizer_params=None,
        time_integrator_type='implicit_euler',
        time_integrator_params=None,
        numerical_flux_type='lax_friedrichs',
        numerical_flux_params=None,
        ):
        """Initialize base model class.""" 

        self.polynomial_type = polynomial_type
        self.polynomial_order = polynomial_order
        self.num_states = num_states
        self.num_elements = num_elements
        self.time_integrator_params = time_integrator_params       
        self.steady_state = steady_state

        # Initialize the start-up routine        
        self.DG_vars = StartUp1D(
            xmin=xmin,
            xmax=xmax,
            num_elements=num_elements,
            polynomial_order=polynomial_order,
            polynomial_type=polynomial_type,
            num_states=num_states,
            )


        # Initialize the stabilizer
        self.stabilizer = factories.get_stabilizer(
            DG_vars=self.DG_vars,
            stabilizer_type=stabilizer_type,
            stabilizer_params=stabilizer_params,
        )

        # Initialize the numerical flux
        if numerical_flux_params is None:
            numerical_flux_params = {}

        # Set the velocity for the Lax-Friedrichs flux
        if numerical_flux_type == 'lax_friedrichs':
            numerical_flux_params['C'] = self.velocity
        if numerical_flux_type == 'roe':
            if not getattr(self.eigen, '__isabstractmethod__', False):
                numerical_flux_params['eigen'] = self.eigen
            elif not getattr(self.system_jacobian, '__isabstractmethod__', False):
                numerical_flux_params['system_jacobian'] = self.system_jacobian
            else:
                error_string = "The eigenvalues and eigenvectors must be implemented for the Roe flux. "
                error_string += "Please implement the eigen() or system_jacobian() methods."
                raise NotImplementedError(
                    error_string
                )

        self.numerical_flux = factories.get_numerical_flux(
            DG_vars=self.DG_vars,
            numerical_flux_type=numerical_flux_type,
            numerical_flux_params=numerical_flux_params,
        )
        
        # Initialize the boundary conditions
        if BC_params.get('numerical_flux') is None:
            BC_params['numerical_flux'] = self.numerical_flux
        else:
            numerical_BC_flux_params = {}
            if BC_params['numerical_flux'] == 'lax_friedrichs':
                numerical_BC_flux_params['C'] = self.velocity
            if BC_params['numerical_flux'] == 'roe':
                if not getattr(self.eigen, '__isabstractmethod__', False):
                    numerical_BC_flux_params['eigen'] = self.eigen
                elif not getattr(self.system_jacobian, '__isabstractmethod__', False):
                    numerical_BC_flux_params['system_jacobian'] = self.system_jacobian
                else:
                    error_string = "The eigenvalues and eigenvectors must be implemented for the Roe flux. "
                    error_string += "Please implement the eigen() or system_jacobian() methods."
                    raise NotImplementedError(
                        error_string
                    )
        if BC_params['treatment'] == 'characteristic':
            BC_params['eigen'] = self.eigen
            BC_params['source'] = self.source
        else:
            BC_params['eigen'] = None
            BC_params['source'] = None

        self.numerical_BC_flux = factories.get_numerical_flux(
            DG_vars=self.DG_vars,
            numerical_flux_type=BC_params['numerical_flux'],
            numerical_flux_params=numerical_BC_flux_params,
        )
        
        self.BCs = factories.get_boundary_conditions(
            DG_vars=self.DG_vars,
            BC_params=BC_params,
            numerical_BC_flux=self.numerical_BC_flux,
            boundary_conditions=self.boundary_conditions,
            flux=self.flux,
            eigen=BC_params['eigen'],
            source=BC_params['source'],
            )

        # Initialize the time integrator
        if time_integrator_params is None:
            time_integrator_params = {}
        if time_integrator_type == 'SSPRK':
            self.step_size = get_CFL_step_size
        else:
            self.step_size = time_integrator_params['step_size']
            time_integrator_params.pop('step_size', None)
            
        time_integrator_params['stabilizer'] = self.stabilizer
        self.time_integrator_type = time_integrator_type

        self.time_integrator = factories.get_time_integrator(
            DG_vars=self.DG_vars,
            time_integrator_type=time_integrator_type,
            time_integrator_params=time_integrator_params,
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
        numerical_flux[:, self.DG_vars.mapI], numerical_flux[:, self.DG_vars.mapO] = \
             self.BCs.apply_boundary_conditions(
                t=t, 
                q_boundary=q[:, [self.DG_vars.vmapI, self.DG_vars.vmapO]],
                flux_boundary=flux[:, [self.DG_vars.vmapI, self.DG_vars.vmapO]],
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
        rhs = -np.multiply(self.DG_vars.rx, self.DG_vars.Dr @ flux) \
            + self.DG_vars.LIFT @ (np.multiply(self.DG_vars.Fscale, d_flux)) \
            + source

        rhs = rhs.reshape(
            (self.DG_vars.num_states, self.DG_vars.Np * self.DG_vars.K), 
            order='F'
            )

        return rhs

    def solve(self, t, q_init, t_final):
        """Solve the model.

        This method solves the model and returns the solution.
        """
        sol = []

        # Compute the steady state solution
        if self.steady_state is not None:
            q_init = compute_steady_state(
                q=q_init,
                rhs=self.compute_rhs,
                newton_params = self.steady_state['newton_params'],
                DG_vars=self.DG_vars,
            )

            q_init = self.stabilizer(q_init)
        
        # Set initial condition
        sol.append(q_init)

        t_vec = [t]

        pbar = tqdm(
            total=t_final,
            bar_format = "{desc}: {percentage:.2f}%|{bar:20}| {n:.2f}/{total_fmt} [{elapsed}<{remaining}]"#
            )
        while t < t_final:
            
            if self.time_integrator_type == 'SSPRK':
                step_size = self.step_size(
                    velocity=self.velocity(sol[-1]), 
                    min_dx=self.DG_vars.dx, 
                    CFL=.1
                    )
            else:
                step_size = self.step_size

            if t + step_size - 1e-1 > t_final:
                step_size = t_final - t
            
            sol_, t = self.time_integrator(
                t=t_vec[-1], 
                q=sol[-1],
                step_size=step_size,
                rhs=self.compute_rhs
                )
            
            t_vec.append(t)

            sol.append(sol_)
            pbar.set_postfix({'':f'{t:.2f}/{t_final:.2f}'})

            pbar.update(step_size)
            #print(t)
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