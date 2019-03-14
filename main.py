from utils import get_argument_parser
from runner import BaseRunner
from baselines.constant_bandit import ConstantBandit
from baselines.clinical_bandit import ClinicalBandit
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

if __name__ == "__main__":
    parser = get_argument_parser()
    args = parser.parse_args()
    assert args.bandit in BANDIT_MAP
    runner = BaseRunner(args.datafile, args.alpha, args.process)
    losses = []
    for _ in range(1):
    	bandit = BANDIT_MAP[args.bandit]()
    	loss = runner.run_bandit(bandit)
    	losses.append(loss)
    print(np.min(losses), np.max(losses))
    print ("Your average loss over 10 runs is: %d" % np.average(losses))
