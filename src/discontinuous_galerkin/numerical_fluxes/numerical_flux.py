import numpy as np
from discontinuous_galerkin.numerical_fluxes.lax_friedrichs import LaxFriedrichsFlux 
import pdb

def get_numerical_flux(DG_vars, numerical_flux_type, numerical_flux_params):
    """Get the numerical flux."""

    if numerical_flux_type == 'lax_friedrichs':
        if numerical_flux_params is None:
            numerical_flux = LaxFriedrichsFlux(DG_vars)
        else:
            numerical_flux = LaxFriedrichsFlux(DG_vars, **numerical_flux_params)

    return numerical_flux

