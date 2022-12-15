import numpy as np
from discontinuous_galerkin.boundary_conditions.dirichlet_boundary_conditions import DirichletBoundaryConditions 
import pdb

def get_boundary_conditions(
    DG_vars, 
    BC_types, 
    boundary_conditions, 
    flux,
    numerical_flux, 
    BC_params=None
    ):
    """Get the boundary conditions."""

    if BC_types == 'dirichlet':
        if BC_params is None:
            BCs = DirichletBoundaryConditions(DG_vars, boundary_conditions, flux, numerical_flux)
        else:
            BCs = DirichletBoundaryConditions(DG_vars, boundary_conditions, flux, numerical_flux, **BC_params)

    return BCs

