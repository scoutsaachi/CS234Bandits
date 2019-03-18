import numpy as np

class RandomBandit:
    def predict(self, context, history):
        # Given the current context vector and the past history in the form of 
        # [(context), (action), reward]
        # return an action
        return np.random.choice([0,1,2])

    def predict_no_update(self, context, history):
    	return self.predict(context, history)

    def update(self, context, action, reward):
    	pass
    