from utils import read_data_file
import numpy as np 

class BaseRunner:
    '''
    BaseRunner runs a bandit algorithm on the warfarin dataset and returns the total regret
    '''

    def __init__(self, filename, alpha):
        # if alpha < 0, use bernoulli (default)
        # otherwise, use risk averse reward
        self.data, self.labels = read_data_file(filename)
        self.num_patients = len(self.labels)
        self.history = []
        self.alpha = alpha
    
    def _indiv_reward_function(self, context, action, label):
        # return reward for context vector for taking the current action 
        return -1*(action != label)
    
    def _compute_regret(self, actions):
        if self.alpha < 0:
            return self._bernoulli_rewards(actions)
        else:
            return self._risk_averse_rewards(actions)
        return np.sum(np.equal(actions, self.labels)-1)

    def run_bandit(self, bandit):
        # Run the initialized bandit on the dataset and return the total regret
        regrets = []
        for _ in range(10):
            concat = np.concatenate([self.data, self.labels], axis=1)
            np.random.shuffle(concat)
            self.data = self.data[:1]
            self.labels = self.data[-1]

            actions = []
            for i in range(self.num_patients):
                context = np.array([self.data[i]]).T
                label = self.labels[i]
                # predict using bandit
                action = bandit.predict(context, self.history)
                # compute reward and update history
                reward = self._indiv_reward_function(context, action, label)
                self.history.append([context, action, reward])
                actions.append(action)
            # print(actions)
            regret = self._compute_regret(actions)
            regrets.append(regret)
        return np.average(regrets)
    

    def _bernoulli_rewards(self, actions):
        # 0 if incorrect, 1 if correct
        return np.sum(np.equal(actions, self.labels)-1)

    def _risk_averse_rewards(self, actions):
        # alpha is how much better (or worse)
        alpha = self.alpha
        reward_table = [
            [1, -alpha/2.0, -1],
            [-alpha, 0, -alpha],
            [-1, -alpha/2.0, 1]
        ]
        tot_rewards = 0
        for i in range(len(self.labels)):
            l = self.labels[i]
            a = actions[i]
            reward = reward_table[l][a]
            tot_rewards += reward
        return tot_rewards