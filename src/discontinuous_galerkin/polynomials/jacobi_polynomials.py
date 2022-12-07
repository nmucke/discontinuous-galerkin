import numpy as np
from scipy.special import gamma
import scipy.special as sci
import scipy.sparse as sps

def JacobiP(x, alpha, beta, N):
    """Evaluate jacobi polynomials at x"""

    xp = x

    PL = np.zeros((N+1, len(xp)))

    gamma0 = 2**(alpha + beta + 1) / (alpha + beta + 1) * gamma(alpha + 1) \
             * gamma(beta + 1) / gamma(alpha + beta + 1)
    PL[0,:] = 1.0 / np.sqrt(gamma0)
    if N == 0:
        return PL
    gamma1 = (alpha + 1) * (beta + 1) / (alpha + beta + 3) * gamma0

    PL[1,:] = ((alpha + beta + 2) * xp / 2 + (alpha - beta) / 2) \
              / np.sqrt(gamma1)
    if N == 1:
        return PL[-1:,:]

    aold = 2 / (2 + alpha + beta) * np.sqrt((alpha + 1) * (beta + 1) \
                                            / (alpha + beta + 3))

    for i in range(1,N):
        h1 = 2 * i + alpha + beta
        anew = 2 / (h1 + 2) * np.sqrt((i + 1) * (i + 1 + alpha + beta) \
              * (i + 1 + alpha) * (i + 1 + beta) / (h1 + 1) / (h1 + 3))
        bnew = -(alpha**2-beta**2)/h1/(h1+2)
        PL[i+1,:] = 1/anew*(-aold*PL[i-1,:] + np.multiply((xp-bnew),PL[i,:]))
        aold = anew

    return PL[-1:,:]

def JacobiGQ(alpha,beta,N):
    """Compute N'th order Gauss quadrature points and weights"""

    x,w = sci.roots_jacobi(N,alpha,beta)

    return x,w

def JacobiGL(alpha,beta,N):
    """Compute N'th order Gauss-Lobatto points"""

    x = np.zeros((N+1,1))

    if N==1:
        x[0]=-1
        x[1]=1
        x = x[:,0]

        return x
    x_int,w = JacobiGQ(alpha+1,beta+1,N-1)
    x = np.append(-1,np.append(x_int,1))

    return x

def Vandermonde1D(x,alpha,beta,N):
    """Initialize Vandermonde Matrix"""

    V1D = np.zeros((len(x),N+1))

    for i in range(0,N+1):
        V1D[:,i] = JacobiP(x,alpha,beta,i)
    return V1D

def GradJacobiP(r,alpha,beta,N):
    """Evaluate derivative of Jacobi polynomials"""

    dP = np.zeros((len(r),1))
    if N == 0:
        return dP
    else:
        dP[:,0] = np.sqrt(N*(N+alpha+beta+1))*JacobiP(r,alpha+1,beta+1,N-1)
    return dP

def GradVandermonde1D(r,alpha,beta,N):
    """Initialize the gradient of modal basis i at point r"""

    DVr = np.zeros((len(r),N+1))

    for i in range(0,N+1):

        DVr[:,i:(i+1)] = GradJacobiP(r,alpha,beta,i)
    return DVr

def Dmatrix1D(r,alpha,beta,N,V):
    """Initialize differentiation matrix"""

    Vr = GradVandermonde1D(r,alpha,beta,N)

    Dr = np.transpose(np.linalg.solve(np.transpose(V),np.transpose(Vr)))
    return Dr