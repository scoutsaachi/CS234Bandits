import numpy as np

from baselines.clinical_bandit import ClinicalBandit
from baselines.constant_bandit import ConstantBandit
from baselines.random_bandit import RandomBandit
from knn_ucb import KNNUCBBandit
from lasso import LassoBandit
from linucb import WarfarinLinUCB
from runner import BaseRunner, HyperRunner, RandomRunner
from thompson import WarfarinThompson
from thompson2 import WarfarinThompsonSeparate
from utils import get_argument_parser
from knn_ucb import KNNUCBBandit,KNNKLBandit
import pickle

np.random.seed(seed=10)

# usage: main.py dataset/clean.csv linear --alpha 0.5

BANDIT_MAP = {
    "constant": ConstantBandit,
    "clinical": ClinicalBandit,
    "linear": WarfarinLinUCB,
    "thompson": WarfarinThompson,
    "thompson2": WarfarinThompsonSeparate,
    "lasso": LassoBandit,
    "knnucb": KNNUCBBandit,
    "knnkl": KNNKLBandit,
    "random": RandomBandit
}

NON_CONSTANT_BANDITS = [
    WarfarinThompsonSeparate, WarfarinLinUCB, ClinicalBandit, RandomBandit,
    LassoBandit, KNNUCBBandit
]

def run(args):
    # run once
    if args.bandit == "hyper":
        policies = [policy() for policy in NON_CONSTANT_BANDITS]
        runner = HyperRunner(args.datafile, args.alpha, args.process,
                                policies, [1, 3])
        result, policy_counts = runner.run()
    elif args.bandit == "randhyper":
        policies = [policy() for policy in NON_CONSTANT_BANDITS]
        runner = RandomRunner(args.datafile, args.alpha, args.process,
                                policies, [1, 3])
        result, policy_counts = runner.run()
    else:
        assert args.bandit in BANDIT_MAP
        runner = BaseRunner(args.datafile, args.alpha, args.process)
        bandit = BANDIT_MAP[args.bandit]()
        result = runner.run_bandit(bandit)
        policy_counts = None
    return result, policy_counts

def runten(args):
    regrets = []
    corr_fracs = []
    counts = []
    policy_counts = []
    for i in range(10):
        print("running %d" % i)
        result, policy_count = run(args)
        regret, corr_frac, count = result
        regrets.append(regret[-1])
        corr_fracs.append(corr_frac[-1])
        counts.append(count)
        policy_counts.append(policy_count)
    f = open(args.result_file, "wb")
    pickle.dump((regrets, corr_fracs,counts, policy_counts), f)
    print("Regret: ", np.mean(regrets))
    print("Correct Fraction: ", np.mean(corr_fracs))

def run_single(args):
    result, policy_count = run(args)
    regret, corr_frac, count = result
    f = open(args.result_file, "wb")
    pickle.dump((regret, corr_frac, count, policy_count), f)
    print("Regret: %s, Correct fraction %s, counter: %s" % (regret[-1], corr_frac[-1], str(count.items())))

if __name__ == "__main__":
    parser = get_argument_parser()
    args = parser.parse_args()
    if args.runten:
        runten(args)
    else:
        run_single(args)
