import numpy as np
from discontinuous_galerkin.stabilizers.base_stabilizer import BaseStabilizer
import pdb

class GeneralizedSlopeLimiter(BaseStabilizer):
    """
    Slope limiter class.

    This class implements the generalized slope limiter for the discontinuous
    Galerkin method. The generalized slope limiter is a TVB modified minmod
    function. The TVB modified minmod function is a minmod function with a
    second derivative upper bound. The second derivative upper bound is used
    to prevent the slope limiter from becoming too steep. The slope limiter
    is used to prevent the discontinuous Galerkin method from becoming
    unstable.  
    """

    def __init__(self, variables, second_derivative_upper_bound=1e-5):
        """Initialize slope limiter class"""

        super(GeneralizedSlopeLimiter, self).__init__()

        self.variables = variables

        self.M = second_derivative_upper_bound


    def _minmod(self, v):
        """ Minmod function 

        v: numpy.ndarray - the array to apply the minmod function to

        returns: numpy.ndarray - the minmod function applied to v
        """

        m = v.shape[0]
        mfunc = np.zeros((v.shape[1],))
        s = np.sum(np.sign(v),0)/m

        ids = np.argwhere(np.abs(s)==1)

        if ids.shape[0]!=0:
            mfunc[ids] = s[ids] * np.min(np.abs(v[:,ids]),0)

        return mfunc

    def _minmodB(self, v):
        """ Implement the TVB modified minmod function

        v: numpy.ndarray - the array to apply the minmod function to

        returns: numpy.ndarray - the minmod function applied to v
        """

        mfunc = v[0,:]
        ids = np.argwhere(np.abs(mfunc) > self.M*self.variables.deltax*self.variables.deltax)

        if np.shape(ids)[0]>0:
            mfunc[ids[:,0]] = self._minmod(v[:,ids[:,0]])

        return mfunc

    def _SlopeLimitLin(self,ul,xl,vm1,v0,vp1):
        """ Apply slopelimited on linear function ul(Np,1) on x(Np,1)
            (vm1,v0,vp1) are cell averages left, center, and right"""

        ulimit = ul
        h = xl[self.variables.Np-1,:]-xl[0,:]

        x0 = np.ones((self.variables.Np,1))*(xl[0,:]+h/2)

        hN = np.ones((self.variables.Np,1))*h

        ux = (2/hN) * np.dot(self.variables.Dr,ul)

        ulimit = np.ones((self.variables.Np,1))*v0 + (xl-x0)*\
            self._minmodB(
                np.stack((ux[0,:],np.divide((vp1-v0),h),np.divide((v0-vm1),h)), axis=0)
                )

        return ulimit

    def _SlopeLimitN(self, u):
        """ Apply slopelimiter (Pi^N) to u assuming u an N'th order polynomial """

        uh = np.dot(self.variables.invV,u)
        uh[1:self.variables.Np,:] = 0
        uavg = np.dot(self.variables.V,uh)
        v = uavg[0:1,:]

        ulimit = u
        eps0 = 1e-8

        ue1 = u[0,:]
        ue2 = u[-1:,:]

        vk = v
        vkm1 = np.concatenate((v[0,0:1],v[0,0:self.variables.K-1]),axis=0)
        vkp1 = np.concatenate((v[0,1:self.variables.K],v[0,(self.variables.K-1):(self.variables.K)]))

        ve1 = vk - self._minmod(np.concatenate((vk-ue1,vk-vkm1,vkp1-vk)))
        ve2 = vk + self._minmod(np.concatenate((ue2-vk,vk-vkm1,vkp1-vk)))

        ids = np.argwhere((np.abs(ve1-ue1)>eps0) | (np.abs(ve2-ue2)>eps0))[:,1]
        if ids.shape[0] != 0:

            uhl = np.dot(self.variables.invV,u[:,ids])
            uhl[2:(self.variables.Np+1),:] = 0
            ul = np.dot(self.variables.V,uhl)

            ulimit[:,ids] = self._SlopeLimitLin(ul,self.variables.x[:,ids],vkm1[ids],
                                                vk[0,ids],vkp1[ids])

        return ulimit

    def apply_stabilizer(self, q):
        """ Apply slope limiter to q """

        states = []
        for i in range(self.variables.num_states):
            states.append(self._SlopeLimitN(np.reshape(
                            q[(i*(self.variables.Np*self.variables.K)):((i+1)*(self.variables.Np*self.variables.K))],
                                (self.variables.Np,self.variables.K),'F')).flatten('F'))

        return np.asarray(states).flatten()
