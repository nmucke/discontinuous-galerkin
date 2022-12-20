import numpy as np
from discontinuous_galerkin.start_up_routines.start_up_1D import StartUp1D
from discontinuous_galerkin.stabilizers.stabilizer import get_stabilizer
from discontinuous_galerkin.numerical_fluxes.numerical_flux import get_numerical_flux
from discontinuous_galerkin.time_integrators.time_integrator import get_time_integrator
from discontinuous_galerkin.boundary_conditions.boundary_conditions import get_boundary_conditions

import matplotlib.pyplot as plt

from abc import abstractmethod

import pdb


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
        BC_types='dirichlet',
        stabilizer_type=None, 
        stabilizer_params=None,
        time_integrator_type='implicit_euler',
        time_integrator_params=None,
        numerical_flux_type='lax_friedrichs',
        numerical_flux_params=None,
        ):
        """Initialize base model class.""" 

        self.time_integrator_params = time_integrator_params       

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
        self.stabilizer = get_stabilizer(
            DG_vars=self.DG_vars,
            stabilizer_type=stabilizer_type,
            stabilizer_params=stabilizer_params,
        )

        numerical_flux_params['C'] = self.velocity
        # Initialize the numerical flux
        self.numerical_flux = get_numerical_flux(
            DG_vars=self.DG_vars,
            numerical_flux_type=numerical_flux_type,
            numerical_flux_params=numerical_flux_params,
        )

        # Initialize the boundary conditions
        self.BCs = get_boundary_conditions(
            DG_vars=self.DG_vars,
            BC_types=BC_types,
            boundary_conditions=self.boundary_conditions,
            flux=self.flux,
            numerical_flux=self.numerical_flux,
            )

        # Initialize the time integrastor
        if time_integrator_type == 'SSPRK':
            time_integrator_params['stabilizer'] = self.stabilizer

        self.time_integrator = get_time_integrator(
            DG_vars=self.DG_vars,
            time_integrator_type=time_integrator_type,
            time_integrator_params=time_integrator_params,
        )


    def __str__(self):
        """ Description of the model. """

        output = "BaseModel: \n"
        output += f"xmin: {self.xmin} \n"
        output += f"xmax: {self.xmax} \n"
        output += f"Number of elements: {self.K} \n"
        output += f"Polynomial order: {self.N} \n"
        output += f"Polynomial type: {self.poly_type} \n"
        output += f"Stabilizer: {self.stabilizer_type} \n"
        #output += f"Time stepping: {self.time_stepper} \n"
        
        return output

    def __repr__(self):
        return self.__str__()

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
    
    def wave_speed(self, q):
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
        source = self.source(q)

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
        
        # Set initial condition
        sol.append(q_init)

        t_vec = [t]

        # Solve the model
        #for i in range(num_steps - 1):
        while t < t_final:
            
            C = np.max(self.velocity(sol[-1]))
            CFL=1.
            dt= CFL/C*self.DG_vars.dx
            step_size = .1*dt

            sol_, t = self.time_integrator(
                t=t_vec[-1], 
                q=sol[-1],
                step_size=step_size,
                rhs=self.compute_rhs
                )
            
            t_vec.append(t)

            sol.append(sol_)
        
        return np.stack(sol, axis=-1) , t_vec
        

