from discontinuous_galerkin.time_integrators.implicit_euler import ImplicitEuler
from discontinuous_galerkin.time_integrators.SSPRK import SSPRK
import pdb 


def get_time_integrator(DG_vars, time_integrator_type, time_integrator_params):

    if time_integrator_type == 'implicit_euler':
        if time_integrator_params is None:
            time_integrator = ImplicitEuler(DG_vars)
        else:
            time_integrator = ImplicitEuler(DG_vars, **time_integrator_params)
    
    elif time_integrator_type == 'SSPRK':
        if time_integrator_params is None:
            time_integrator = SSPRK(DG_vars)
        else:
            time_integrator = SSPRK(DG_vars, **time_integrator_params)

    return time_integrator

