import numpy as np;

class Model:
    
    pos = np.array([[0.], [0.]])
    vel = np.array([[0.], [-1.]])
    acc = np.array([[1.], [-1.]])
    
    i = 0
    
    def __init__(self):
        return;
    
    def update(self, dt):
        self.i += dt;
        self.acc[0][0] = np.sin(self.i)
        self.acc[1][0] = np.tan(self.i)
        
        self.vel += self.acc * dt
        self.pos += self.vel * dt;
        
        if (self.i > 20):
            self.vel[0][0] = 0
            self.vel[1][0] = 0
        
    def get_ground_truth(self):
        return (np.row_stack((self.pos, self.vel, self.acc)))