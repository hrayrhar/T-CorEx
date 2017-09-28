import numpy as np


class L1Reg:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        
    def regularize(self, W):
        ret = 0.0
        T = len(W)
        for t in range(T - 1):
            ret += np.abs(W[t+1] - W[t]).sum()
        return self.alpha * ret
    
    def get_gradient(self, W):
        ret = np.zeros(np.array(W).shape)
        T = len(W)
        for t in range(T - 1):
            ret[t+1] += np.sign(W[t+1] - W[t])
            ret[t] -= np.sign(W[t+1] - W[t])
        return self.alpha * ret

class L2Reg:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        
    def regularize(self, W):
        ret = 0.0
        T = len(W)
        for t in range(T - 1):
            ret += np.square(W[t+1] - W[t]).sum()
        return 0.5 * self.alpha * ret
    
    def get_gradient(self, W):
        ret = np.zeros(np.array(W).shape)
        T = len(W)
        for t in range(T - 1):
            ret[t+1] += (W[t+1] - W[t])
            ret[t] -= (W[t+1] - W[t])
        return self.alpha * ret
