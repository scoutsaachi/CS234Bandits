import numpy as np
import cvxpy as cp
from utils import history_index
# based on http://web.stanford.edu/~bayati/papers/lassoBandit.pdf
class LassoBandit:
    def __init__(self):
        self.forced_sample_schedule = {}  # index -> action to be forced, zero indexed
        self.forced_sample_schedule_inv = {} # action -> list of indices
        self.h = 5
        self.initl2 = 0.05 # these can also be 0 maybe
        self.l1 = 0.05
        self.l2 = self.initl2
        self.context_size = 8
        self.num_actions = 3

        # self sampling params
        self.ss_K = self.num_actions # num actions
        self.ss_q = 1 # sampling hyperparam
        self.load_forced_sample_schedule()

    # honestly this is so sus
    def beta_hat(self, X, y, l):  # l = lambda
        n = len(X)
        beta = cp.Variable(self.context_size)
        objective = cp.Minimize(cp.sum_squares(y - X * beta) / n + l * cp.norm(beta, 1))
        prob = cp.Problem(objective)
        result = prob.solve()
        return beta.value

    # populate forced sample schedule
    def load_forced_sample_schedule(self):
        for action in range(self.ss_K):
            for n in range(8): # yes magic #s
                for j in range(1, self.ss_q + 1):
                    sample_idx = (2 ** n - 1) * self.ss_K * self.ss_q + self.ss_q * action + j - 1
                    self.forced_sample_schedule[sample_idx] = action
                    if action not in self.forced_sample_schedule_inv:
                        self.forced_sample_schedule_inv[action] = []
                    self.forced_sample_schedule_inv[action].append(sample_idx)

    def predict(self, context, history):
        t = len(history)
        if t in self.forced_sample_schedule:
            return self.forced_sample_schedule[len(history)] # TODO: possible off by one error

        # we're just going to eliminate the worst action rather than do the ones within a threshold
        worst_action = None
        worst_action_score = None
        for action in range(self.num_actions):
            forced_schedule = [x for x in self.forced_sample_schedule_inv[action] if x < t]
            forced_samples_features = history_index(history, 0, t_arr=forced_schedule) # multiindexing probably only works in numpy
            forced_samples_targets = history_index(history, 2, t_arr=forced_schedule)
            beta = self.beta_hat(forced_samples_features, forced_samples_targets, self.l1)
            action_score = context.T @ beta
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
            beta = self.beta_hat(history_index(history, 0), history_index(history, 2), self.l2) # this doesn't work IRL but this is the idea
            action_score = context.T @ beta
            if best_action_score == None:
                best_action = action
                best_action_score = action_score
            elif action_score > best_action_score:
                best_action = action
                best_action_score = action_score
        self.l2 = np.sqrt((np.log(t) + np.log(self.context_size)) / t) * self.initl2
        print("best action: %d" % best_action)
        return best_action


lasso_bandit = LassoBandit()
# lasso_bandit.load_forced_sample_schedule()
