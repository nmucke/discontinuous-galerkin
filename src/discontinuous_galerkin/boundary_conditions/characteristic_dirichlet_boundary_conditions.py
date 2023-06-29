import numpy as np
from abc import abstractmethod
from discontinuous_galerkin.base.base_boundary_conditions import BaseBoundaryConditions
import pdb
from scipy.linalg import eig

class CharacteristicDirichletBoundaryConditions(BaseBoundaryConditions):
    """
    Dirichlet boundary conditions class.

    This class contains the functionality for the Dirichlet boundary conditions.
    """
    def __init__(
        self, 
        DG_vars, 
        boundary_conditions, 
        flux, 
        numerical_flux,
        system_jacobian,
        source=None,
        transform_matrices=None,
        form='conservative',
        primitive_to_conservative=None,
        conservative_to_primitive=None,
        **args
        ):
        """Initialize the class."""

        super().__init__()

        self.DG_vars = DG_vars
        self.boundary_conditions = boundary_conditions
        self.num_BCs = self.DG_vars.num_states
        self.system_jacobian = system_jacobian
        self.source = source
        self.numerical_flux = numerical_flux
        self.flux = flux
        self.transform_matrices = transform_matrices
        self.form = form
        self.primitive_to_conservative = primitive_to_conservative
        self.conservative_to_primitive = conservative_to_primitive

    
    def _get_primitive_state(self, t, conv_state):

        prim_state = np.zeros(conv_state.shape)

        for i in range(prim_state.shape[1]):
            prim_state[:, i, 0] = self.conservative_to_primitive(conv_state[:, i, 0])
        
        return prim_state
    
    def _get_conservative_state(self, t, prim_state):

        conv_state = np.zeros(prim_state.shape)

        for i in range(prim_state.shape[1]):
            conv_state[:, i, 0] = self.primitive_to_conservative(prim_state[:, i, 0])

        return conv_state
    
    def _get_primitive_and_conservative_state(
        self, 
        t: float = None, 
        prim_state: np.ndarray = None,
        conv_state: np.ndarray = None, 
        side: str = 'left'
        ):
        
        if prim_state is None:
            if side == 'left':
                conv_state = conv_state[:, :, 0:1]
            elif side == 'right':
                conv_state = conv_state[:, :, -1:]
            prim_state = self._get_primitive_state(t=t, conv_state=conv_state)

        elif conv_state is None: 
            if side == 'left':
                prim_state = prim_state[:, :, 0:1]
            elif side == 'right':
                prim_state = prim_state[:, :, -1:]
            conv_state = self._get_conservative_state(t=t, prim_state=prim_state)

        dx = self.DG_vars.dx
        #prim_state_diff = self.DG_vars.Dr @ prim_state[:, :, 0:1]

        if side == 'left':

            prim_state_diff = (prim_state[:, 1, 0] - prim_state[:, 0, 0])/dx

            conv_state = conv_state[:, 0, 0]
            prim_state = prim_state[:, 0, 0]

            #prim_state_diff = prim_state_diff[:, 0, 0]


        
        elif side == 'right':
            prim_state_diff = (prim_state[:, -2, 0] - prim_state[:, -1, 0])/dx

            conv_state = conv_state[:, -1, 0]
            prim_state = prim_state[:, -1, 0]




        return conv_state, prim_state, prim_state_diff

    def _get_matrices(
            self, 
            t: float, 
            q: np.ndarray = None, 
            C: np.ndarray = None, 
            side: str = 'left'
        ):

        P, P_inv, A, S, S_inv, Lambda = \
            self.transform_matrices(t=t, q=q)
        
        # Check if the form is primitive or conservative
        if True:#self.form[side] == 'primitive': 
            C = P_inv @ C
        elif self.form[side] == 'conservative':
            S = P @ S
        
        return P, P_inv, A, S, S_inv, Lambda, C
    
    def _get_derived_and_specified_indices(self, t, q, side):

        ind_der = np.zeros(self.DG_vars.num_states, dtype=bool)
        ind_spec = np.zeros(self.DG_vars.num_states, dtype=bool)

        for i in range(self.DG_vars.num_states):
            bc = self.boundary_conditions(t, q)
            bc = bc['state']
            bc = bc[i][side]
            
            if bc is None:
                # Indices where BCs are to be derived
                ind_der[i] = 1
            else:
                # Indices where BCs are specified
                ind_spec[i] = 1
        
        return ind_der, ind_spec
    
    def _get_specified_BCs(self, t, q, side, ind_spec):

        dWgivenLdt = np.zeros(np.sum(ind_spec))

        indices = np.where(ind_spec)[0]

        for i, i_spec in enumerate(indices):
            bc = self.boundary_conditions(t, q)
            bc = bc['state']
            bc = bc[i_spec][side]
            dWgivenLdt[i] = bc#self.boundary_conditions(t, q)[i_spec][side]
        
        return dWgivenLdt

    def _compute_RHS(self, s, L, dWgivenLdt, k, ind_eigenvalues, ind_der, ind_spec, S, C, P, side='left'):

        L_minus = np.linalg.solve(s[:, ind_eigenvalues], -dWgivenLdt -k)
        #if side == 'right':
        #    L[ind_eigenvalues] = L_minus
        #else:
        #    L[ind_eigenvalues] = L_minus
        
        L[ind_eigenvalues] = L_minus

        # RHS for BCs
        dw_dt = S @ L + C

        RHS = np.zeros((self.DG_vars.num_states))

        RHS[ind_der] = -dw_dt[ind_der]

        RHS[ind_spec] = -dWgivenLdt

        #if self.form[side] == 'primitive':
        RHS = P @ RHS

        return RHS

    def get_BC_rhs(
        self, 
        t: float, 
        q: np.ndarray = None, 
        source: np.ndarray = None
        ):
        """Compute the boundary condition right hand side."""
        
        q = q.reshape(
            (self.DG_vars.num_states, self.DG_vars.Np, self.DG_vars.K), 
            order='F'
        )

        ###### Left boundary ######

        # Get source term
        C = source[:, 0, 0]

        # Get the primitive and conservative states and derivative
        q_left, w_left, w_left_diff = self._get_primitive_and_conservative_state(
            t=t, 
            prim_state=q, 
            side='left'
        )
        
        # Get the transformation matrices
        P, P_inv, A, S, S_inv, Lambda, C = \
            self._get_matrices(t=t, q=w_left, C=C, side='left')

        # Indices of positive and negative eigenvalues
        ind_neg_eigenvalues = np.where(Lambda.diagonal() < 0)[0]
        ind_pos_eigenvalues = np.where(Lambda.diagonal() > 0)[0]

        # Indices where BCs are to be derived and specified
        ind_der, ind_spec = \
            self._get_derived_and_specified_indices(t=t, q=w_left, side='left')

        w_left_diff[1] *= 1e6
        L = Lambda @ S_inv @ w_left_diff

        # Incoming wave for specified BCs
        s = S[ind_spec, :]
        c = C[ind_spec]

        # negative eigenvalue contriubtions
        k = s[:, ind_neg_eigenvalues] @ L[ind_neg_eigenvalues] + c

        # positive eigenvalue contriubtions
        dWgivenLdt = self._get_specified_BCs(
            t=t, 
            q=w_left, 
            side='left', 
            ind_spec=ind_spec
        )
        RHS_left = self._compute_RHS(
            s=s,
            L=L, 
            dWgivenLdt=dWgivenLdt, 
            k=k, 
            ind_eigenvalues=ind_pos_eigenvalues, 
            ind_der=ind_der, 
            ind_spec=ind_spec, 
            S=S, 
            C=C, 
            P=P
        )

            
        ###### Right boundary ######

        # Get source term
        C = source[:, -1, -1]

        # Get the primitive and conservative states and derivative
        q_right, w_right, w_right_diff = \
            self._get_primitive_and_conservative_state(t=t, prim_state=q, side='right')    

        # Get the transformation matrices
        P, P_inv, A, S, S_inv, Lambda, C = \
            self._get_matrices(t=t, q=w_right, C=C, side='right')

        # Indices of positive and negative eigenvalues
        ind_neg_eigenvalues = np.where(Lambda.diagonal() < 0)[0]
        ind_pos_eigenvalues = np.where(Lambda.diagonal() > 0)[0]

        # Indices where BCs are to be derived and specified
        ind_der, ind_spec = \
            self._get_derived_and_specified_indices(t=t, q=w_right, side='right')

        w_right_diff[1] *= 1e6
        L = Lambda @ S_inv @ w_right_diff

        # Incoming wave for specified BCs
        s = S[ind_spec, :]
        c = C[ind_spec]
        
        # positive eigenvalue contriubtions
        k = s[:, ind_pos_eigenvalues] @ L[ind_pos_eigenvalues] + c

        # positive eigenvalue contriubtions
        dWgivenLdt = self._get_specified_BCs(
            t=t, 
            q=w_right, 
            side='right', 
            ind_spec=ind_spec
        )

        RHS_right = self._compute_RHS(
            s=s,
            L=L, 
            dWgivenLdt=dWgivenLdt, 
            k=k, 
            ind_eigenvalues=ind_neg_eigenvalues, 
            ind_der=ind_der, 
            ind_spec=ind_spec, 
            S=S, 
            C=C, 
            P=P,
            side='right'
        )

        
        '''
        q_right = q[:, :, -1:]
        w_right = self._get_primitive_state(t=t, q=q_right)
        w_right_diff = self.DG_vars.Dr @ w_right[:, :, 0:1]

        q_right = q_right[:, 0, 0]
        w_right = w_right[:, 0, 0]
        w_right_diff = w_right_diff[:, 0, 0]

        # Get the transformation matrices
        P, P_inv, A, S, S_inv, Lambda = \
            self.transform_matrices(t=t, q=q_right)

        ind_der = np.zeros(self.DG_vars.num_states, dtype=bool)
        ind_spec = np.zeros(self.DG_vars.num_states, dtype=bool)

        # Indices of positive and negative eigenvalues
        ind_neg_eigenvalues = np.where(Lambda.diagonal() < 0)[0]
        ind_pos_eigenvalues = np.where(Lambda.diagonal() > 0)[0]

        # Check if the form is primitive or conservative
        if self.form['left'] == 'primitive': 
            C = P_inv @ C
        elif self.form['left'] == 'conservative':
            S = P @ S

        for i in range(self.DG_vars.num_states):
            bc = self.boundary_conditions(t, q_right)[i]['right']

            if bc is None:
                # Indices where BCs are to be derived
                ind_der[i] = 1
            else:
                # Indices where BCs are specified
                ind_spec[i] = 1

        L = Lambda @ S_inv @ w_right_diff

        # Incoming wave for specified BCs
        s = S[ind_spec, :]
        c = C[ind_spec]

        # negative eigenvalue contriubtions
        k = s[:, ind_pos_eigenvalues] @ L[ind_pos_eigenvalues] + c

        # solve for L minus
        # SHOULD BE
        # L_minus = np.linalg.solve(s[:, ind_pos_eigenvalues], -dWgivenLdt -k)
        dWgivenLdt = 0.
        L_minus = np.linalg.solve(s[:, ind_neg_eigenvalues], -dWgivenLdt -k)

        L[ind_neg_eigenvalues] = L_minus

        # RHS for BCs
        dw_right_dt = S @ L + C

        RHS_right[ind_der] = dw_right_dt[ind_der]
        RHS_right[ind_spec] = -dWgivenLdt

        RHS_right = P @ RHS_right
        '''
        
        return RHS_left, RHS_right



    def apply_boundary_conditions(
            self, 
            t, 
            q_boundary, 
            flux_boundary, 
            q_boundary_diff,
            step_size
        ):
        """Apply the boundary conditions."""
        
        # Compute the ghost states
        ghost_states = self._compute_characteristic_ghost_states(
            t, 
            q_boundary, 
            q_boundary_diff,
            step_size=step_size
        )
        
        # Compute the ghost flux
        ghost_flux = self._compute_ghost_flux(ghost_states)

        # Compute numerical boundary flux
        numerical_flux = self.numerical_flux(
            q_inside = q_boundary,
            q_outside = ghost_states,
            flux_inside = flux_boundary,
            flux_outside = ghost_flux,
            on_boundary=True
        )
        
        return numerical_flux[:, 0], numerical_flux[:, -1]

