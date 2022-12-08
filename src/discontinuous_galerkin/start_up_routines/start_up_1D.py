from discontinuous_galerkin.polynomials.jacobi_polynomials import (
    JacobiGL, 
    JacobiGQ, 
    JacobiP, 
    Vandermonde1D, 
    GradVandermonde1D,
    GradJacobiP,
)
from scipy import sparse as sps
import numpy as np
import pdb


def diracDelta(x):
    """Evaluate dirac delta function at x"""

    f = np.zeros(x.shape)
    f[np.argwhere((x<0.2e-1) & (x>-0.2e-1))] = 1
    return f

class StartUp1D():
    def __init__(
        self, 
        xmin=0,
        xmax=1,
        num_elements=10,
        polynomial_order=5,
        polynomial_type='legendre',
        num_states=1,
    ):
        self.xmin = xmin # Lower bound of domain
        self.xmax = xmax # Upper bound of domain
        self.K = num_elements # Number of elements
        self.N = polynomial_order # Polynomial order
        self.Np = self.N + 1 # Number of polynomials

        self.num_states = num_states # Number of states

        self.NODETOL = 1e-10
        self.Nfp = 1
        self.Nfaces = 2 # Number of faces on elements

        # Legendre or Chebyshev polynomials
        if polynomial_type == 'legendre':
            self.alpha = 0
            self.beta = 0
        elif polynomial_type == 'chebyshev':
            self.alpha = -0.5
            self.beta = -0.5

        self.r = JacobiGL(self.alpha, self.beta, self.N)  # Reference domain nodes
        self.V = Vandermonde1D(self.r, self.alpha, self.beta,
                               self.N)  # Vandermonde matrix
        self.invV = np.linalg.inv(self.V)  # Inverse Vandermonde matrix
        #self.Dr = self.Dmatrix1D(self.r, self.alpha, self.beta, self.N,self.V)  # Differentiation matrix
        self.Dr = self.Dmatrix1D()  # Differentiation matrix
        #self.M = np.transpose(
        #    np.linalg.solve(np.transpose(self.invV), self.invV))  # Mass matrix
        #self.invM = np.linalg.inv(self.M)  # Inverse mass matrix

        self.invM = np.dot(self.V, np.transpose(self.V))
        self.M = np.linalg.inv(self.invM)


        self.LIFT = self.lift1D()  # Surface integral

        # Generate equidistant grid
        self.Nv, self.VX, self.K, self.EtoV = self.MeshGen1D()
        self.EtoV = self.EtoV.astype(int)

        self.va = np.transpose(self.EtoV[:, 0])  # Leftmost grid points in each element
        self.vb = np.transpose(self.EtoV[:, 1])  # rightmost grid points in each element

        # Global grid
        self.x = np.ones((self.Np, 1)) * self.VX[self.va.astype(int)] + \
                 0.5 * (np.reshape(self.r, (len(self.r), 1)) + 1) \
                 * (self.VX[self.vb.astype(int)] - self.VX[self.va.astype(int)])

        # Element size
        self.deltax = np.min(np.abs(self.x[0, :] - self.x[-1, :]))
        self.dx = np.min(self.x[-self.N:, 0]-self.x[0:-1, 0])

        self.invMk = 2 / self.deltax * self.invM
        self.Mk = np.linalg.inv(self.invMk)

        fmask1 = np.where(np.abs(self.r + 1.) < self.NODETOL)[0]
        fmask2 = np.where(np.abs(self.r - 1.) < self.NODETOL)[0]

        self.Fmask = np.concatenate((fmask1, fmask2), axis=0)
        self.Fx = self.x[self.Fmask, :]

        self.EtoE, self.EtoF = self.Connect1D()

        self.vmapM, self.vmapP, self.vmapB,self.mapB,self.mapI,\
        self.mapO,self.vmapI,self.vmapO = self.BuildMaps1D()

        self.nx = self.Normals1D()
        self.nx = self.nx.flatten('F')

        self.rx, self.J = self.GeometricFactors()

        self.Fscale = 1./(self.J[self.Fmask,:])

    def Normals1D(self):
        """Compute outward pointing normals"""

        nx = np.zeros((self.Nfp * self.Nfaces, self.K))
        nx[0, :] = -1.0
        nx[1, :] = 1.0
        return nx

    def BuildMaps1D(self):
        """Connectivity and boundary tables for nodes given in the K #
           of elements, each with N+1 degrees of freedom."""

        nodeids = np.reshape(np.arange(0, self.K * self.Np), (self.Np, self.K), 'F')
        vmapM = np.zeros((self.Nfp, self.Nfaces, self.K))
        vmapP = np.zeros((self.Nfp, self.Nfaces, self.K))

        for k1 in range(0, self.K):
            for f1 in range(0, self.Nfaces):
                vmapM[:, f1, k1] = nodeids[self.Fmask[f1], k1]

        for k1 in range(0, self.K):
            for f1 in range(0, self.Nfaces):
                k2 = self.EtoE[k1, f1].astype(int)
                f2 = self.EtoF[k1, f1].astype(int)

                vidM = vmapM[:, f1, k1].astype(int)
                vidP = vmapM[:, f2, k2].astype(int)

                x1 = self.x[np.unravel_index(vidM, self.x.shape, 'F')]
                x2 = self.x[np.unravel_index(vidP, self.x.shape, 'F')]

                D = (x1 - x2) ** 2
                if D < self.NODETOL:
                    vmapP[:, f1, k1] = vidP

        vmapP = vmapP.flatten('F')
        vmapM = vmapM.flatten('F')

        mapB = np.argwhere(vmapP == vmapM)
        vmapB = vmapM[mapB]

        mapI = 0
        mapO = self.K * self.Nfaces-1
        vmapI = 0
        vmapO = self.K * self.Np-1

        return vmapM.astype(int), vmapP.astype(int), vmapB.astype(int), mapB.astype(int), mapI,mapO,vmapI,vmapO

    def Connect1D(self):
        """ Build global connectivity arrays for 1D grid based
            on standard EToV input array from grid generator"""

        TotalFaces = self.Nfaces * self.K
        Nv = self.K + 1

        vn = [0, 1]
        
        SpFToV = sps.lil_matrix((TotalFaces, Nv))

        sk = 0
        for k in range(0, self.K):
            for face in range(0, self.Nfaces):
                SpFToV[sk, self.EtoV[k, vn[face]]] = 1.
                sk = sk + 1

        SpFToF = np.dot(SpFToV, np.transpose(SpFToV)) - sps.eye(TotalFaces)
        faces = np.transpose(np.nonzero(SpFToF))
        faces[:, [0, 1]] = faces[:, [1, 0]] + 1

        element1 = np.floor((faces[:, 0] - 1) / self.Nfaces)
        face1 = np.mod((faces[:, 0] - 1), self.Nfaces)
        element2 = np.floor((faces[:, 1] - 1) / self.Nfaces)
        face2 = np.mod((faces[:, 1] - 1), self.Nfaces)

        ind = np.ravel_multi_index(np.array([face1.astype(int),
                                element1.astype(int)]), (self.Nfaces, self.K))
        EtoE = np.reshape(np.arange(0, self.K), (self.K, 1))\
                            * np.ones((1, self.Nfaces))
        EtoE[np.unravel_index(ind, EtoE.shape, 'F')] = element2
        EtoF = np.ones((self.K, 1)) * np.reshape(np.arange(0, self.Nfaces),
                                                 (1, self.Nfaces))
        EtoF[np.unravel_index(ind, EtoE.shape, 'F')] = face2
        return EtoE, EtoF

    def GeometricFactors(self):
        """Compute the matrix elements for the local mapping"""
        xr = np.dot(self.Dr, self.x)
        J = xr
        rx = np.divide(1, J)

        return rx, J
    
    def MeshGen1D(self):
        """Generate equidistant grid"""

        Nv = self.K+1

        VX = np.arange(1.,Nv+1.)

        for i in range(0,Nv):
            VX[i] = (self.xmax-self.xmin)*i/(Nv-1) + self.xmin

        EtoV = np.zeros((self.K,2))
        for k in range(0,self.K):
            EtoV[k,0] = k
            EtoV[k,1] = k+1

        return Nv, VX, self.K, EtoV
    
    def lift1D(self):
        """Compute surface integral term of DG formulation"""

        Emat = np.zeros((self.Np,self.Nfaces*self.Nfp))
        Emat[0,0] = 1
        Emat[self.Np-1,1] = 1
        LIFT = np.dot(self.V,np.dot(np.transpose(self.V),Emat))
        return LIFT

    def Dmatrix1D(self):
        """Initialize differentiation matrix"""

        Vr = GradVandermonde1D(self.r,self.alpha,self.beta,self.N)

        Dr = np.transpose(np.linalg.solve(np.transpose(self.V),np.transpose(Vr)))
        return Dr

    def identity(self,x,num_states=1):
        return x