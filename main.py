from utils import get_argument_parser
from runner import BaseRunner
from baselines.constant_bandit import ConstantBandit
from baselines.clinical_bandit import ClinicalBandit
from linucb import WarfarinLinUCB

# usage: main.py clean.csv linear --alpha 0.5

BANDIT_MAP = {
    "constant": ConstantBandit,
    "clinical": ClinicalBandit,
    "linear": WarfarinLinUCB
}

if __name__ == "__main__":
    parser = get_argument_parser()
    args = parser.parse_args()
    print(args.alpha)
    runner = BaseRunner(args.datafile, args.alpha)
    assert args.bandit in BANDIT_MAP
    bandit = BANDIT_MAP[args.bandit]()
    loss = runner.run_bandit(bandit)
    print ("Your loss is: %d" % loss)