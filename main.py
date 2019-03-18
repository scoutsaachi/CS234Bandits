from utils import get_argument_parser
from runner import BaseRunner, HyperRunner, RandomRunner
from baselines.constant_bandit import ConstantBandit
from baselines.clinical_bandit import ClinicalBandit
from baselines.random_bandit import RandomBandit
from linucb import WarfarinLinUCB
from thompson import WarfarinThompson
from thompson2 import WarfarinThompsonSeparate
from lasso import LassoBandit
import numpy as np

# usage: main.py clean.csv linear --alpha 0.5

BANDIT_MAP = {
    "constant": ConstantBandit,
    "clinical": ClinicalBandit,
    "linear": WarfarinLinUCB,
    "thompson": WarfarinThompson,
    "thompson2": WarfarinThompsonSeparate,
    "lasso": LassoBandit  # pretty sure this isn't working correctly
}

NON_CONSTANT_BANDITS = [  WarfarinThompsonSeparate, WarfarinLinUCB, ClinicalBandit, RandomBandit ] 

if __name__ == "__main__":
    parser = get_argument_parser()
    args = parser.parse_args()
    losses = []
    if args.bandit == "hyper":
    	for _ in range(10):
	    	policies = [policy() for policy in NON_CONSTANT_BANDITS]
	    	runner = HyperRunner(args.datafile, args.alpha, args.process, policies, [1,3])
	    	loss = runner.run()
	    	losses.append(loss)
    elif args.bandit == "randhyper":
        for _ in range(10):
	        policies = [policy() for policy in NON_CONSTANT_BANDITS]
	        runner = RandomRunner(args.datafile, args.alpha, args.process, policies, [1,3])
	        loss = runner.run()
	        losses.append(loss)
    else:
	    assert args.bandit in BANDIT_MAP
	    runner = BaseRunner(args.datafile, args.alpha, args.process)
	    for _ in range(10):
	    	bandit = BANDIT_MAP[args.bandit]()
	    	loss = runner.run_bandit(bandit)
	    	losses.append(loss)
    print(np.min(losses), np.max(losses))
    print ("Your average loss over 10 runs is: %d" % np.average(losses))
