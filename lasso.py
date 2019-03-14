import numpy as np
import pprint

from utils import history_index


# based on http://web.stanford.edu/~bayati/papers/lassoBandit.pdf
# more based on https://www0.gsb.columbia.edu/faculty/azeevi/PAPERS/2-arm-SSY-final.pdf
class LassoBandit:
    def __init__(self):
        self.forced_sample_schedule = {
        }  # index -> action to be forced, zero indexed
        self.forced_sample_schedule_inv = {}  # action -> list of indices
        self.h = 5
        # self.initl2 = 0.05  # these can also be 0 maybe
        # self.l1 = 0.05
        # self.l2 = self.initl2
        self.context_size = 9
        self.num_actions = 3

        # self sampling params
        self.ss_K = self.num_actions  # num actions
        self.ss_q = 1  # sampling hyperparam
        self.load_forced_sample_schedule()

    def print_matrix(self, A):
        shape = A.shape
        vals = [[np.round(A[i,j],2) for j in range(shape[1])] for i in range(shape[0])]
        for v in vals:
            print(v)

    def beta_hat(self, X, y):
        beta = np.linalg.pinv(X) @ y
        return beta
        # XTX_inv = np.linalg.inv(X.T @ X)
        # return XTX_inv @ X @ y / len(X)

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

    def predict(self, context, history):
        t = len(history)
        aug_context = np.vstack(([[1]], context))
        if t in self.forced_sample_schedule:
            return self.forced_sample_schedule[len(history)]
        # MODIFICATION: we're just going to eliminate the worst action rather than do the ones within a threshold
        worst_action = None
        worst_action_score = None
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
            if worst_action_score == None:
                worst_action_score = action_score
                worst_action = action
            elif action_score < worst_action_score:
                worst_action = action
                worst_action_score = action_score
        # choosing the best action from all historical data
        best_action = None
        best_action_score = None
        for action in range(self.num_actions):
            if action == worst_action:
                continue
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
        # self.l2 = np.sqrt(
        #     (np.log(t) + np.log(self.context_size)) / t) * self.initl2
        # print("best action: %d" % best_action)
        # print(best_action, worst_action)
        return best_action


lasso_bandit = LassoBandit()
# lasso_bandit.load_forced_sample_schedule()
