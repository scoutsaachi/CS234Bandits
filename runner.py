from utils import read_data_file

class BaseRunner:
    '''
    BaseRunner runs a bandit algorithm on the warfarin dataset and returns the total regret
    '''

    def __init__(self, filename):
        self.data, self.labels = read_data_file(filename)
        self.num_patients = len(self.labels)
        self.history = []
    
    def _indiv_reward_function(self, context, action, label):
        # return reward for context vector for taking the current action 
        return -1*(action != label)
    
    def _compute_regret(actions):
        return np.sum(-1*(action == self.labels))

    def run_bandit(bandit):
        # Run the initialized bandit on the dataset and return the total regret
        actions = []
        for i in range(self.num_patients):
            context = self.data[i]
            label = self.labels[i]
            # predict using bandit
            action = bandit.predict(context, self.history)
            # compute reward and update history
            reward = self._indiv_reward_function(context, action, label)
            self.history.append([context, action, reward])
            actions.append(action)
        return self._compute_regret(actions)
    