import numpy as np
from discontinuous_galerkin.numerical_fluxes.lax_friedrichs import LaxFriedrichsFlux 


def get_numerical_flux(variables, numerical_flux_type, numerical_flux_params):
    """Get the numerical flux."""

    if numerical_flux_type == 'lax_friedrichs':
        if numerical_flux_params is None:
            numerical_flux = LaxFriedrichsFlux(variables)
        else:
            numerical_flux = LaxFriedrichsFlux(variables, **numerical_flux_params)

    return numerical_flux

