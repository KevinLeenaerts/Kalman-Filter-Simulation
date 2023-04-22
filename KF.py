import numpy as np;

class KF:
    
    def __init__(self, F, G, Q, P, x0):
        self.n = F.shape[1]
        
        self.F = F;
        self.G = G;
        self.Q = Q;
        self.P = P;
        self.x = x0;
    
    def predict(self):
        self.x = np.dot(self.F, self.x);
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x;
    
    def update(self, z, H, R):
        I = np.eye(self.n)
        y = z - np.dot(H, self.x);
        S = R + np.dot(H, np.dot(self.P, H.T))
        K = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))

        self.x = self.x + np.dot(K, y)
        self.P = np.dot(np.dot(I - np.dot(K, H), self.P), (I - np.dot(K, H)).T) + np.dot(np.dot(K, R), K.T)
        
        return self.x
        