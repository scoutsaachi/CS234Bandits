import numpy as np
from utils import bucketize_action

class ThompsonBandit:

    def __init__(self, feature_dim, T):

        feature_dim += 3 # ugh lol

        # could put in a config or something
        R = 1
        epsilon = 1/np.log(T)  
        delta = .99992

        self.v = R * np.sqrt((24*feature_dim*np.log(1/delta)/epsilon))
        # print(self.v ** 2)
        self.mu = np.zeros(feature_dim) 
        self.f = np.zeros(feature_dim) 
        self.B = np.identity(feature_dim) 

    def _get_action_context(self, context, action):
        # adds action to context and normalizes
        actions = np.zeros(3)
        actions[action] = 1
        context = np.append(context[:8], actions)
        # context /= np.max(context)
        # print(context)
        return context

    def update(self, context, action, reward):
        # print(np.linalg.norm(self.mu, ord=2))
        context = self._get_action_context(context, action)
        self.f = self.f + context * reward
        self.B = self.B + np.outer(context,context)
        self.mu = np.linalg.inv(self.B) @ self.f

    def print_weights(self):
        print(self.mu)
        print(self.f)
        print(self.B)

    def predict_no_update(self, context, history):
        mu_samp = np.random.multivariate_normal(self.mu, (self.v**2) * np.linalg.inv(self.B))

        results = []
        for a in range(3):

            context = self._get_action_context(context, a)
            results.append(np.dot(context.T, mu_samp))
            # print(context.T)
            # print(np.dot(context.T, mu_samp))
        # print()

        best_action = np.argmax(results)
        # print(results)
        # print(best_action)
        return best_action



    def predict(self, context, history):
        # Given the current context vector and the past history in the form of 
        # [(context), (action), reward]
        # return an action

        if len(history) > 0:
            self.update(*(history[-1]))

        return self.predict_no_update(context, history)

        
    
class WarfarinThompson(ThompsonBandit):
    def __init__(self):
        super().__init__(8, 4386)