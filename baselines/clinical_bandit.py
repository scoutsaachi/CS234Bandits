import numpy as np
from utils import bucketize_action

class ClinicalBandit:

    def __init__(self):
        self.weights = np.array([[-0.6752, 0.4060, 0.0443, 1.2799, -0.5695, -0.2546, 0.0118, 0.0134]]).T
        self.bias = 4.0376

    def predict(self, context, history):
        # Given the current context vector and the past history in the form of 
        # [(context), (action), reward]
        # return an action
        score = np.square(np.dot(context.T, self.weights) + self.bias)
        return bucketize_action(score)

    def predict_no_update(self, context, history):
    	return self.predict(context, history)

    def update(self, context, action, reward):
    	pass
    
