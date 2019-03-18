import pprint

import numpy as np

from utils import history_index


# based on http://web.stanford.edu/~bayati/papers/lassoBandit.pdf
# more based on https://www0.gsb.columbia.edu/faculty/azeevi/PAPERS/2-arm-SSY-final.pdf
class LassoBandit:
    def __init__(self):
        self.forced_sample_schedule = {
        }  # index -> action to be forced, zero indexed
        self.forced_sample_schedule_inv = {}  # action -> list of indices
        self.ball_radius = 0.1  # empirically chosen, but still not better than 0 lol
        self.context_size = 9
        self.num_actions = 3

        # self sampling params
        self.ss_K = self.num_actions  # num actions
        self.ss_q = 1  # sampling hyperparam
        self.load_forced_sample_schedule()

    def print_matrix(self, A):
        shape = A.shape
        vals = [[np.round(A[i, j], 2) for j in range(shape[1])]
                for i in range(shape[0])]
        for v in vals:
            print(v)

    def beta_hat(self, X, y):
        beta = np.linalg.pinv(X) @ y
        return beta

    # populate forced sample schedule
    def load_forced_sample_schedule(self):
        for action in range(self.ss_K):
            for n in range(8):  # yes magic #s
                for j in range(1, self.ss_q + 1):
                    sample_idx = (
                        2**n -
                        1) * self.ss_K * self.ss_q + self.ss_q * action + j - 1
                    self.forced_sample_schedule[sample_idx] = action
                    if action not in self.forced_sample_schedule_inv:
                        self.forced_sample_schedule_inv[action] = []
                    self.forced_sample_schedule_inv[action].append(sample_idx)

    def phase_one(self, aug_context, history):
        t = len(history)
        forced_sample_scores = []
        for action in range(self.num_actions):
            forced_schedule = [
                x for x in self.forced_sample_schedule_inv[action] if x < t
            ]
            forced_samples_features = history_index(
                history, 0, t_arr=forced_schedule, add_one=True)
            forced_samples_targets = history_index(
                history, 2, t_arr=forced_schedule)
            beta = self.beta_hat(forced_samples_features,
                                 forced_samples_targets)
            action_score = aug_context.T @ beta
            forced_sample_scores.append(action_score)
        return forced_sample_scores

    def predict(self, context, history):
        t = len(history)
        aug_context = np.vstack(([[1]], context))
        if t in self.forced_sample_schedule:
            return self.forced_sample_schedule[len(history)]

        forced_sample_scores = self.phase_one(aug_context, history)

        # testing (one run):
        # threshold | loss
        # 3.5       | -3093
        # 3.0       | -3030
        # 2.5       | -3142
        # 2.0       | -2830
        # 1.5       | -2831
        # 1.0       | -2736
        # 0.85      | -2834
        # 0.80      | -2277
        # 0.76      | -2897
        # 0.75      | -1929
        # 0.74      | -2078
        # 0.70      | -2057
        # 0.65      | -2035
        # 0.5       | -3089
        # 0.0       | -2657

        # testing (ten runs):
        # 0         | -2268
        # 0.05      | -2314
        # 0.1       | -2255
        # 0.2       | -2468
        # 0.5       | -2549
        # 0.6       | -2299
        # 0.75      | -2580
        # 0.8       | -2494
        # 0.9       | -2677
        # 1         | -2515
        # 1.5       | -3078
        # 2.5       | -3134

        # phase 2, get candidates in ball
        max_score = np.max(forced_sample_scores)
        # choosing the best action from all historical data
        best_action = None
        best_action_score = None
        for action in range(self.num_actions):
            if max_score - forced_sample_scores[action] > self.ball_radius:
                continue  # too far from optimal
            all_contexts = history_index(history, 0, range(t), add_one=True)
            all_labels = history_index(history, 2, range(t))
            beta = self.beta_hat(all_contexts, all_labels)
            action_score = aug_context.T @ beta
            if best_action_score == None:
                best_action = action
                best_action_score = action_score
            elif action_score > best_action_score:
                best_action = action
                best_action_score = action_score
        return best_action

    def predict_no_update(self, context, history):
        return self.predict(context, history)

    def update(self, context, action, reward):
        pass
