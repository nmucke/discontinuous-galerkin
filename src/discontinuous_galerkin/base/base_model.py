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
            numerical_flux_args=numerical_flux_args,#, 'alpha': 0.5},#
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
            transform_matrices=self.transform_matrices,
        )

        # initialize steady state BCs
        self.steady_state_BCs = factories.get_boundary_conditions(
            DG_vars=self.DG_vars,
            BC_args={
                'type': 'dirichlet',
                'treatment': 'naive',
            },
            numerical_flux=self.numerical_flux,
            boundary_conditions=self.steady_state_boundary_conditions,
            flux=self.flux,
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

        #conv = np.array(self.primitive_to_conservative(q))
        
        # Compute the numerical flux
        numerical_flux = self.numerical_flux(
            q_inside=q[:, self.DG_vars.vmapM], 
            q_outside=q[:, self.DG_vars.vmapP],
            flux_inside=flux[:, self.DG_vars.vmapM],
            flux_outside=flux[:, self.DG_vars.vmapP],
            )
            
        # Compute the source term
        source = self.source(t, q)
    
        if True:# self.steady_state_solve:
            
            # Compute boundary conditions
            q_boundary=q[:, [self.DG_vars.vmapI, self.DG_vars.vmapO]]
            flux_boundary=flux[:, [self.DG_vars.vmapI, self.DG_vars.vmapO]]

            numerical_flux[:, self.DG_vars.mapI], numerical_flux[:, self.DG_vars.mapO] = \
                self.steady_state_BCs.apply_boundary_conditions(
                    t=t, 
                    q_boundary=q_boundary,
                    flux_boundary=flux_boundary,
                    step_size=self.step_size,
                    )
        
        elif False: # Compute boundary conditions
            q_boundary=q[:, [self.DG_vars.vmapI, self.DG_vars.vmapO]]
            flux_boundary=flux[:, [self.DG_vars.vmapI, self.DG_vars.vmapO]]

            numerical_flux[:, self.DG_vars.mapI], numerical_flux[:, self.DG_vars.mapO] = \
                self.BCs.apply_boundary_conditions(
                    t=t, 
                    q_boundary=q_boundary,
                    flux_boundary=flux_boundary,
                    step_size=self.step_size,
                    )
            
            '''
            # Compute boundary conditions
            q_boundary=q[:, [self.DG_vars.vmapI, self.DG_vars.vmapO]]
            flux_boundary=flux[:, [self.DG_vars.vmapI, self.DG_vars.vmapO]]

            numerical_flux[:, self.DG_vars.mapI], numerical_flux[:, self.DG_vars.mapO] = \
                self.BCs.apply_boundary_conditions(
                    t=t, 
                    q_boundary=q_boundary,
                    flux_boundary=flux_boundary,
                    step_size=self.step_size,
                    )
            '''
            
        '''
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
        '''
        
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
        
        if self.steady_state_solve:
            #rhs[:, 0] = 0
            #rhs[:, -1] = 0

            bc_left = np.zeros((3))
            bc_right = np.zeros((3))

            bc_mask_left = np.zeros((self.DG_vars.num_states, ))
            bc_mask_right = np.zeros((self.DG_vars.num_states, ))
            for i in range(self.DG_vars.num_states):
                for edge, idx in zip(['left', 'right'], [0, -1]):
                    bc = self.boundary_conditions(t, q_boundary)[i][edge]
                    if edge == 'left' and bc is not None:
                        bc_left[i] = bc[0]
                        bc_mask_left[i] = 1

                        rhs[i, 0] = q[i, 0] - bc_left[i]

                    elif edge == 'right' and bc is not None:
                        bc_right[i] = bc
                        bc_mask_right[i] = 1

                        rhs[i, -1] = q[i, -1] - bc_right[i]
            
            '''
            bc_mask_right = bc_mask_right.astype(bool)
            bc_mask_left = bc_mask_left.astype(bool)
            D, L, R = self.eigen(q[:,0])
            R_inv = np.linalg.inv(R)
            D_vec = np.diag(D)
            for i in range(self.DG_vars.num_states):
                if self.DG_vars.nx[0] * D_vec[i] > 0:
                    lol = R_inv[:, i] * (bc_left - q[:, 0])
                    rhs[i, 0] =(bc_left - q[:, 0])# lol[i]
                    
            print(rhs[i, 0])

            D, L, R = self.eigen(q[:,-1])
            R_inv = np.linalg.inv(R)
            D_vec = np.diag(D)
            for i in range(self.DG_vars.num_states):
                if self.DG_vars.nx[-1] * D_vec[i] > 0:
                    lol = R_inv[:, i] * (bc_right - q[:, -1])
                    rhs[bc_mask_right, 0] = (bc_right - q[:, -1])#lol[bc_mask_right]
            '''
            

        # if self.steady_state_solve:
        '''
        q_boundary=q[:, [self.DG_vars.vmapI, self.DG_vars.vmapO]]

        BCs = self.steady_state_boundary_conditions(q=q_boundary)
        
        rhs[:, [self.DG_vars.vmapI, self.DG_vars.vmapO]] = BCs
        #print(rhs[:, [self.DG_vars.vmapI, self.DG_vars.vmapO]])
        '''
        if False:#not self.steady_state_solve:
            # Compute boundary conditions
            BC_rhs_left, BC_rhs_right = self.BCs.get_BC_rhs(
                t=t,
                q=q,
                source=source,
            )

            # left boundary
            rhs[:, self.DG_vars.vmapI] = BC_rhs_left

            # right boundary
            rhs[:, self.DG_vars.vmapO] = BC_rhs_right

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
        
        #rho_g_A_g, rho_l_A_l, rho_m_u_m = self.primitive_to_conservative(q_init)
    

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