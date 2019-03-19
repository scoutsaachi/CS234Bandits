import numpy as np
from utils import bucketize_action

class KNNUCBBandit:

    def compute_distances(self, x, history):
        results = []
        for context, action, reward in history:
            rho = np.sqrt(np.sum(np.square(x - context)))
            results.append((rho, action, reward))
        return sorted(results)
    

    def compute_u(self, context, rhos, t, k):
        r = rhos[:k+1][-1][0] # max distance
        action_counts = self.compute_n(rhos, k)
        u = np.zeros(3)
        for i in range(3):
            c = action_counts[i]
            if c == 0:
                u[i] = 0
            else:
                u[i] = np.log(t)/c
        return u + r
    
    def compute_n(self, rhos, k):
        action_counts = [0,0,0]
        for _, action, _ in rhos[:k+1]:
            action_counts[action] += 1
        return action_counts

    def compute_f(self, rhos, k, a):
        reward_counts = [0,0,0]
        for _, action, reward in rhos[:k+1]:
            reward_counts[action] += reward
        action_counts = self.compute_n(rhos, k)
        n = action_counts[a]
        if n == 0:
            return 0
        return reward_counts[a]/n

    def predict(self, context, history):
        # Given the current context vector and the past history in the form of 
        # [(context), (action), reward]
        # return an action
        t = len(history)
        if t < 3:
            return t # return action which is just the history number
        rhos = self.compute_distances(context, history)
        us = np.vstack([self.compute_u(context, rhos, t, k) for k in range(t-1)])
        k_stars = np.argmax(us, axis=0)
        results = []
        for a in range(3):
            k_star = k_stars[a]
            u_val = us[k_star][a]
            f_val = self.compute_f(rhos, k_star, a)
            results.append(u_val + f_val)
        action= np.argmax(results)
        print(action, t)
        return action