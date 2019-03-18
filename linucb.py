import numpy as np
from utils import bucketize_action

class LinUCBBandit:

    def __init__(self, feature_dim, alpha):
        self.A = [np.identity(feature_dim) for _ in range(3)]
        self.b = [np.zeros((feature_dim,1)) for _ in range(3)]
        self.alpha = alpha


    def update(self, context, action, reward):
        self.A[action] = self.A[action] + context @ context.T
        self.b[action] = self.b[action] + reward * context

    def print_weights(self):
        print(self.b)
        print(self.A)

    def predict_no_update(self, context, history):
        ucbs = []
        for a in range(3):
            A = self.A[a]
            b = self.b[a]
            A_inv = np.linalg.inv(A)
            theta = np.dot(A_inv, b)
            sqrt_term = self.alpha * np.sqrt((context.T @ A_inv) @ context)
            ucb_value = theta.T @ context + sqrt_term
            ucbs.append(ucb_value[0])
        best_action = np.argmax(ucbs)
        return best_action

    def predict(self, context, history):
        # Given the current context vector and the past history in the form of 
        # [(context), (action), reward]
        # return an action
        if len(history) > 0:
            self.update(*(history[-1]))

        return self.predict_no_update(context, history)
        
    
class WarfarinLinUCB(LinUCBBandit):
    def __init__(self):
        super().__init__(8, 0.75)