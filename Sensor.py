import numpy as np
import random;

class Sensor:
    
    def __init__(self, model, H, sigma, bias, biasSigma):
        self.model = model;
        self.H = H;
        self.sigma = sigma;
        self.bias = bias;
        self.biasSigma = biasSigma;
        
    def get_reading(self):
        return random.gauss(np.dot(self.H, self.model.get_ground_truth()), self.sigma) + random.gauss(self.bias, self.biasSigma);