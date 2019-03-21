import numpy as np

from utils import maxmin_normalize, normalize, read_data_file
from collections import Counter

class BaseRunner:
    '''
    BaseRunner runs a bandit algorithm on the warfarin dataset and returns the total regret
    '''

    def __init__(self, filename, alpha, process):
        # if alpha < 0, use bernoulli (default)
        # otherwise, use risk averse reward
        self.data, self.labels = read_data_file(filename)
        if process == "maxmin":
            print("using maxmin norm")
            self.data = maxmin_normalize(self.data)
        elif process == "norm":
            print("using normalization on data")
            self.data = normalize(self.data)
        self.num_patients = len(self.labels)
        self.alpha = alpha

    def _indiv_reward_function(self, context, action, label):
        # return reward for context vector for taking the current action
        return self._compute_regret(action, label)

    def _compute_regret(self, action, label):
        if self.alpha < 0:
            return self._bernoulli_reward(action, label)
        else:
            return self._risk_averse_reward(action, label)

    def _compute_regrets(self, actions, labels):
        time_regret = []
        tot_regret = 0

        correct_frac = []
        correct_n = 0
        for i in range(len(labels)):
            action = actions[i]
            label = labels[i]
            reward = self._compute_regret(action, label)
            inf_reward = self._compute_regret(label, label)
            tot_regret +=  inf_reward - reward
            time_regret.append(tot_regret)
            correct_n += int(action == label)
            correct_frac.append(correct_n/float(i+1))
        return time_regret, correct_frac, Counter(actions)
        

    def run_bandit(self, bandit):
        # Run the initialized bandit on the dataset and return the total regret
        history = []
        actions = []
        patients = list(range(self.num_patients))
        labels = []
        np.random.shuffle(patients)
        for i in patients:
            context = np.array([self.data[i]]).T
            label = self.labels[i]
            # predict using bandit
            action = bandit.predict(context, history)
            # compute reward and update history
            reward = self._indiv_reward_function(context, action, label)
            history.append([context, action, reward])
            actions.append(action)
            labels.append(label)
        return self._compute_regrets(actions, labels)

    def _bernoulli_reward(self, action, label):
        return np.equal(action, label) - 1
    
    def _risk_averse_reward(self, action, label):
        alpha = self.alpha
        action = int(action)
        label = int(label)
        reward_table = [[1, -alpha / 2.0, -1],
                        [-alpha, 0, -alpha],
                        [-1, -alpha / 2.0, 1]]
        return reward_table[label][action]

    # def _bernoulli_rewards(self, actions, labels):
    #     # 0 if incorrect, 1 if correct
    #     return np.sum(np.equal(actions, labels) - 1)

    # def _risk_averse_rewards(self, actions, labels):
    #     # alpha is how much better (or worse)
    #     alpha = self.alpha
    #     reward_table = [[1, -alpha / 2.0, -1], [-alpha, 0, -alpha],
    #                     [-1, -alpha / 2.0, 1]]
    #     tot_rewards = 0
    #     for i in range(len(labels)):
    #         l = labels[i]
    #         a = actions[i]
    #         reward = reward_table[l][a]
    #         tot_rewards += reward
    #     return tot_rewards


class HyperRunner(BaseRunner):
    '''
    Implements HyperTSFB: an ensembling algorithm that shares information between policies
    '''

    def __init__(self, filename, alpha, process, policies, sample_indices=[]):
        super().__init__(filename, alpha, process)
        self.policies = policies
        self.sample_indices = sample_indices
        self.alphas = np.zeros(3)
        self.betas = np.zeros(3)
        self.action_counts = np.zeros(3)
        self.weights = [[[] for j in range(len(policies))] for i in range(3)]
        self.action_policy_counts = [np.zeros(len(policies)) for _ in range(3)]

    def _get_probability_of_action(self, policy, context, history, a,
                                   policy_index):
        actions = np.zeros(3)
        probs = np.zeros(3)

        N = 25 if policy_index in self.sample_indices else 1
        for _ in range(N):  # number of samples can change
            action = policy.predict_no_update(context, history)
            actions[action] += 1
        probs = actions / np.sum(actions)
        return probs[a]

    def _get_probability_of_action_given_context(self, context, history, a, t):
        actions = np.zeros(3)
        for _ in range(20):  # number of samples can change
            action, _ = self._guess_action(context, history, t)
            actions[action] += 1
        probs = actions / np.sum(actions)
        return probs[a]

    def _guess_action(self, context, history, t):
        r = []
        for a in range(3):
            r.append(np.random.beta(self.alphas[a] + 1, self.betas[a] + 1))

        r_pols = np.zeros(len(self.policies))
        for i in range(len(self.policies)):
            for a in range(3):
                if self.action_policy_counts[a][
                        i] < 30:  # they use 30 but we don't have to
                    w = np.random.uniform(0, 1)
                else:
                    # print("var: ",np.var(self.weights[a][i]) / self.action_policy_counts[a][i])
                    w = np.random.normal(
                        np.mean(self.weights[a][i]),
                        max(np.var(self.weights[a][i]) /
                        self.action_policy_counts[a][i], 0))

                r_pols[i] = r_pols[i] + self.action_counts[a] / t * r[a] * w
        best_policy = np.argmax(r_pols)

        action = self.policies[best_policy].predict_no_update(context, history)
        return action, best_policy

    def run(self):
        history = []
        actions = []
        patients = list(range(self.num_patients))
        labels = []
        np.random.shuffle(patients)
        policy_counts = []
        for time, t in enumerate(patients):
            time += 1
            context = np.array([self.data[t]]).T
            label = self.labels[t]
            action, best_policy = self._guess_action(context, history, time)
            reward = self._indiv_reward_function(context, action, label)
            policy_counts.append(best_policy)
            reward += 1  # (-1, 0) -> (0, 1)

            self.action_counts[action] += 1

            if reward == 1:
                p_a_given_x = self._get_probability_of_action_given_context(
                    context, history, action, time)
                for i in range(len(self.policies)):
                    p_a_x = self._get_probability_of_action(
                        self.policies[i], context, history, action, i)
                    if p_a_given_x > 0:
                        w = p_a_x / p_a_given_x
                        # print(i, p_a_x, w)

                        self.weights[action][i].append(w)
                        self.action_policy_counts[action][i] += 1

                self.alphas[action] += 1
            else:
                self.betas[action] += 1

            reward -= 1  # (0, 1) -> (-1, 0)

            history.append([context, action, reward])

            for p in self.policies:
                p.update(*(history[-1]))

            actions.append(action)
            labels.append(label)
        return self._compute_regrets(actions, labels), Counter(policy_counts)


class RandomRunner(BaseRunner):
    # just randomly chooses a policy every time

    def __init__(self, filename, alpha, process, policies, sample_indices=[]):
        super().__init__(filename, alpha, process)
        self.policies = policies

    def run(self):
        history = []
        actions = []
        patients = list(range(self.num_patients))
        labels = []
        np.random.shuffle(patients)
        policy_counts = []
        for t in patients:
            context = np.array([self.data[t]]).T
            label = self.labels[t]
            policy = np.random.choice(self.policies)
            policy_counts.append(policy)
            action = policy.predict_no_update(context, history)
            reward = self._indiv_reward_function(context, action, label)
            history.append([context, action, reward])
            policy.update(*(history[-1]))
            actions.append(action)
            labels.append(label)
        return self._compute_regrets(actions, labels), Counter(policy_counts)
