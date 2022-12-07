def Filter1D(self,N,Nc,s):
    """Initialize 1D filter matrix of size N.
        Order of exponential filter is (even) s with cutoff at Nc;"""

    filterdiag = np.ones((N+1))

    alpha = -np.log(np.finfo(float).eps)

    for i in range(Nc,N):
        #filterdiag[i+1] = np.exp(-alpha*((i-Nc)/(N-Nc))**s)
        filterdiag[i+1] = np.exp(-alpha*((i-1)/N)**s)

    self.filterMat = np.dot(self.V,np.dot(np.diag(filterdiag),self.invV))

def apply_filter(self,q,num_states):
    """Apply filter to state vector"""

    states = []
    for i in range(num_states):
        states.append(np.dot(self.filterMat,np.reshape(
                q[(i * (self.Np * self.K)):((i + 1) * (self.Np * self.K))],
                (self.Np, self.K), 'F')).flatten('F'))

    return np.asarray(states).flatten()