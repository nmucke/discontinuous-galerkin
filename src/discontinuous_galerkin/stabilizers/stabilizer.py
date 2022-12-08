from discontinuous_galerkin.stabilizers.slope_limiters import GeneralizedSlopeLimiter
from discontinuous_galerkin.stabilizers.filters import ExponentialFilter
import pdb 


def get_stabilizer(variables, stabilizer_type, stabilizer_params):

    if stabilizer_type == 'slope_limiter':
        if stabilizer_params is None:
            stabilizer = GeneralizedSlopeLimiter(variables)
        else:
            stabilizer = GeneralizedSlopeLimiter(variables, **stabilizer_params)

    elif stabilizer_type == 'filter':
        if stabilizer_params is None:
            stabilizer = ExponentialFilter(variables)
        else:
            stabilizer = ExponentialFilter(variables, **stabilizer_params)

    return stabilizer

