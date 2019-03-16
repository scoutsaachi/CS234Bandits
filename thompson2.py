import numpy as np
from utils import bucketize_action

class Thompson2Bandit:

    def __init__(self, feature_dim, T):

        # could put in a config or something
        R = 1
        epsilon = 1/np.log(T) 
        delta = .99995 

        self.v = R * np.sqrt((24*feature_dim*np.log(1/delta)/epsilon))
        # print (self.v **2)
        self.mu = [np.zeros(feature_dim) for _ in range(3)]
        self.f = [np.zeros(feature_dim) for _ in range(3)]
        self.B = [np.identity(feature_dim) for _ in range(3)] 

    def _get_context(self, context):
        # normalizes
        return np.squeeze(context)

    def update(self, context, action, reward):
        context = self._get_context(context)
        self.B[action] = self.B[action] + np.outer(context,context)
        self.f[action] = self.f[action] + context * reward
        # print("AFTER",self.f[action])
        # print(self.B[action])
        self.mu[action] = np.linalg.inv((self.B[action])) @ (self.f[action])
        # print("AFTER", self.mu[action])

    def print_weights(self):
        print(self.mu)
        print(self.f)
        print(self.B)


    def predict(self, context, history):
        # Given the current context vector and the past history in the form of 
        # [(context), (action), reward]
        # return an action

        if len(history) > 0:
            self.update(*(history[-1]))

        results = []
        for a in range(3):
            mu_samp = np.random.multivariate_normal(self.mu[a], (self.v**2) * np.linalg.inv(self.B[a]))
            context = self._get_context(context)
            results.append(np.dot(context.T, mu_samp))
            # print(context.T)
            # print(np.dot(context.T, mu_samp))
        # print()

        best_action = np.argmax(results)
        # print(results)
        # print(best_action)
        return best_action
    
class WarfarinThompsonSeparate(Thompson2Bandit):
    def __init__(self):
        super().__init__(8, 4386)