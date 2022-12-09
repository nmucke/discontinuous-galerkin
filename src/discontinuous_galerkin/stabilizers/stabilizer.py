from discontinuous_galerkin.stabilizers.slope_limiters import GeneralizedSlopeLimiter
from discontinuous_galerkin.stabilizers.filters import ExponentialFilter
import pdb 


def get_stabilizer(DG_vars, stabilizer_type, stabilizer_params):

    if stabilizer_type == 'slope_limiter':
        if stabilizer_params is None:
            stabilizer = GeneralizedSlopeLimiter(DG_vars)
        else:
            stabilizer = GeneralizedSlopeLimiter(DG_vars, **stabilizer_params)

    elif stabilizer_type == 'filter':
        if stabilizer_params is None:
            stabilizer = ExponentialFilter(DG_vars)
        else:
            stabilizer = ExponentialFilter(DG_vars, **stabilizer_params)

    return stabilizer

