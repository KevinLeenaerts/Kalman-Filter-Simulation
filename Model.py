import numpy as np;

class Model:
    
    pos = np.array([[0.], [0.]])
    vel = np.array([[0.], [-1.]])
    acc = np.array([[1.], [-1.]])
    
    def __init__(self):
        return;
    
    def update(self, dt):
        self.vel += self.acc * dt
        self.pos += self.vel * dt;
        
    def get_ground_truth(self):
        return (np.row_stack((self.pos, self.vel, self.acc)))