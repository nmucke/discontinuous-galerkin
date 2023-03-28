from discontinuous_galerkin.numerical_fluxes.lax_friedrichs import LaxFriedrichsFlux 
from discontinuous_galerkin.numerical_fluxes.roe import RoeFlux
from discontinuous_galerkin.stabilizers.slope_limiters import GeneralizedSlopeLimiter
from discontinuous_galerkin.stabilizers.filters import ExponentialFilter
from discontinuous_galerkin.time_integrators.implicit_euler import ImplicitEuler
from discontinuous_galerkin.time_integrators.SSPRK import SSPRK
from discontinuous_galerkin.boundary_conditions.dirichlet_boundary_conditions import DirichletBoundaryConditions 
from discontinuous_galerkin.boundary_conditions.characteristic_dirichlet_boundary_conditions import CharacteristicDirichletBoundaryConditions
import pdb

def get_boundary_conditions(
    DG_vars, 
    BC_args, 
    numerical_flux,
    boundary_conditions, 
    flux, 
    system_jacobian=None,
    source=None,
    transform_matrices=None,
    ):
    """Get the boundary conditions."""
    
    if BC_args['treatment'] == 'naive':
        factory = {
            'dirichlet': DirichletBoundaryConditions,
        }
        BCs = factory[BC_args['type']](
            DG_vars=DG_vars, 
            boundary_conditions=boundary_conditions, 
            flux=flux,
            numerical_flux=numerical_flux,
            **BC_args
            )
    elif BC_args['treatment'] == 'characteristic':
        factory = {
            'dirichlet': CharacteristicDirichletBoundaryConditions,
        }
        BCs = factory[BC_args['type']](
            DG_vars=DG_vars, 
            boundary_conditions=boundary_conditions, 
            flux=flux,
            numerical_flux=numerical_flux,
            system_jacobian=system_jacobian,
            source=source,
            transform_matrices=transform_matrices,
            **BC_args
            )
    
    return BCs

def get_time_integrator(
    DG_vars, 
    time_integrator_args: dict=None,
    stabilizer=None,
    ):
    """Get instance of time integrator."""

    if time_integrator_args is None:
        time_integrator_args = {}

    factory = {
        'implicit_euler': ImplicitEuler,
        'SSPRK': SSPRK,
    }

    time_integrator_type = time_integrator_args['type']
    time_integrator_args = \
        {key: time_integrator_args[key] for key in time_integrator_args if key != 'type'}
    
    time_integrator_args['stabilizer'] = stabilizer

    time_integrator = factory[time_integrator_type](DG_vars, **time_integrator_args)

    return time_integrator


def get_stabilizer(
    DG_vars, 
    stabilizer_args: dict=None
    ):
    """Get instance of stabilizer."""

    if stabilizer_args is None:
        stabilizer_args = {}

    factory = {
        'slope_limiter': GeneralizedSlopeLimiter,
        'filter': ExponentialFilter,
    }

    if stabilizer_args['type'] is None:
        stabilizer = lambda x: x
    else:
        stabilizer_type = stabilizer_args['type']
        stabilizer_args = \
            {key: stabilizer_args[key] for key in stabilizer_args if key != 'type'}
        
        stabilizer = factory[stabilizer_type](DG_vars, **stabilizer_args)

    return stabilizer



def get_numerical_flux(
    DG_vars, 
    numerical_flux_args: dict=None,
    system_jacobian=None,
    eigen=None
    ):
    """Get instance of numerical flux."""

    if numerical_flux_args is None:
        numerical_flux_args = {}

    factory = {
        'lax_friedrichs': LaxFriedrichsFlux,
        'roe': RoeFlux,
    }

    numerical_flux_type = numerical_flux_args['type']

    if numerical_flux_type == 'roe':
        check_if_everything_is_implemented_for_roe_flux(
            system_jacobian=system_jacobian,
            eigen=eigen
        )

    numerical_flux_args = \
        {key: numerical_flux_args[key] for key in numerical_flux_args if key != 'type'}
    numerical_flux = factory[numerical_flux_type](DG_vars, **numerical_flux_args)
    
    return numerical_flux

def check_if_everything_is_implemented_for_roe_flux(
    system_jacobian,
    eigen    
):
    if system_jacobian is None and eigen is None:
        error_string = "The eigenvalues and eigenvectors must be implemented for the Roe flux. "
        error_string += "Please implement the eigen() or system_jacobian() methods."

        raise NotImplementedError(
            error_string
        )


