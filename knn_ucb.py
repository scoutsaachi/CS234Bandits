import numpy as np
from utils import bucketize_action
import pickle as pkl

class KNNBandit:

    def __init__(self, is_kl):
        self.is_kl = is_kl
        maxv, minv = pkl.load(open("maxmin.pkl", "rb"))
        n = len(maxv)
        self.maxv = maxv.reshape((n,1))
        self.minv = minv.reshape((n,1))
        # self.maxv = np.squeeze(maxv)
        # self.minv = np.squeeze(minv)

    def compute_distances(self, x, history):
        results = []
        for context, action, reward in history:
            rho = np.sqrt(np.sum(np.square(x - self.normalize(context))))
            results.append((rho, action, reward))
        return sorted(results)

    def normalize(self, x):
        result = (x - self.minv) / (self.maxv - self.minv)
        return result

    def find_kl_max(self, p_val, upper):
        def divergence(p, q):
            return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))
        best_d = 0
        best_i = 0
        i = p_val
        # print(p_val, upper)
        while(i < 1):
            d = divergence(p_val, i)
            if d > upper:
                return best_i
            elif d > best_d:
                best_d = d
                best_i = i
            i += 0.01
        return best_i

    def predict(self, context, history):
        # Given the current context vector and the past history in the form of 
        # [(context), (action), reward]
        # return an action
        
        context = self.normalize(context)

        t = len(history)+1
        if t <= 3:
            return t-1 # return action which is just the history number
        rhos = self.compute_distances(context, history)
        best_k = None

        def safe_divide(a, b):
            return 0 if b==0 else a/b 
        
        action_counts = [0,0,0]
        reward_counts = [0,0,0]
        best_u = [None, None, None]
        best_I = [None, None, None] # k, f
        for k in range(t-1):
            rho, action, reward = rhos[k]
            action_counts[action] += 1
            reward_counts[action] += reward + 1
            for a in range(3):
                u = safe_divide(np.log(t), action_counts[a]) + rho
                f = safe_divide(reward_counts[a], action_counts[a])
                if self.is_kl:
                    upper = safe_divide(np.log(t), action_counts[a])
                    if upper == 0:
                        omega = 0
                    else:
                        omega = self.find_kl_max(f, upper)
                    I = omega + rho
                else:
                    I = f+u
                if best_u[a] is None or best_u[a] < u:
                    best_u[a] = u
                    best_I[a] = I
        action = np.argmax(best_I)
        print(action, t)
        return action


    def predict_no_update(self, context, history):
        return self.predict(context, history)

    def update(self, context, action, reward):
        pass

class KNNUCBBandit(KNNBandit):
    def __init__(self):
        super().__init__(False)

class KNNKLBandit(KNNBandit):
    def __init__(self):
        super().__init__(True)
