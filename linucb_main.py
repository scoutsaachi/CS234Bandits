from utils import get_argument_parser
from runner import BaseRunner
from linucb import LinUCBBandit

if __name__ == "__main__":
    parser = get_argument_parser()
    args = parser.parse_args()
    for alpha in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
        runner = BaseRunner(args.datafile), args.alpha
        bandit = LinUCBBandit(8, alpha)
        loss = runner.run_bandit(bandit)
        print(alpha)
        print ("Your loss is: %d" % loss)