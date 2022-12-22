from discontinuous_galerkin.numerical_fluxes.lax_friedrichs import LaxFriedrichsFlux 
from discontinuous_galerkin.stabilizers.slope_limiters import GeneralizedSlopeLimiter
from discontinuous_galerkin.stabilizers.filters import ExponentialFilter
from discontinuous_galerkin.time_integrators.implicit_euler import ImplicitEuler
from discontinuous_galerkin.time_integrators.SSPRK import SSPRK
from discontinuous_galerkin.boundary_conditions.dirichlet_boundary_conditions import DirichletBoundaryConditions 
import pdb

def get_boundary_conditions(
    DG_vars, 
    BC_types, 
    boundary_conditions, 
    flux,
    numerical_flux, 
    BC_params: dict=None
    ):
    """Get the boundary conditions."""

    if BC_params is None:
        BC_params = {}

    factory = {
        'dirichlet': DirichletBoundaryConditions,
    }

    BCs = factory[BC_types](DG_vars, boundary_conditions, flux, numerical_flux, **BC_params)

    return BCs

def get_time_integrator(
    DG_vars, 
    time_integrator_type, 
    time_integrator_params: dict=None
    ):
    """Get instance of time integrator."""

    if time_integrator_params is None:
        time_integrator_params = {}

    factory = {
        'implicit_euler': ImplicitEuler,
        'SSPRK': SSPRK,
    }

    time_integrator = factory[time_integrator_type](DG_vars, **time_integrator_params)


    return time_integrator


def get_stabilizer(
    DG_vars, 
    stabilizer_type, 
    stabilizer_params: dict=None
    ):
    """Get instance of stabilizer."""

    if stabilizer_params is None:
        stabilizer_params = {}

    factory = {
        'slope_limiter': GeneralizedSlopeLimiter,
        'filter': ExponentialFilter,
    }

    if stabilizer_type is None:
        stabilizer = lambda x: x
    else:
        stabilizer = factory[stabilizer_type](DG_vars, **stabilizer_params)

    return stabilizer



def get_numerical_flux(
    DG_vars, 
    numerical_flux_type, 
    numerical_flux_params: dict=None
    ):
    """Get instance of numerical flux."""

    if numerical_flux_params is None:
        numerical_flux_params = {}

    factory = {
        'lax_friedrichs': LaxFriedrichsFlux,
    }

    numerical_flux = factory[numerical_flux_type](DG_vars, **numerical_flux_params)
    
    return numerical_flux

