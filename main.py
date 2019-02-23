from utils import get_argument_parser
from runner import BaseRunner
from base_bandit import BaseBandit

BANDIT_MAP = {
    "base": BaseBandit
}

if __name__ == "__main__":
    parser = get_argument_parser()
    args = parser.parse_args()
    runner = BaseRunner(args.datafile)
    assert args.bandit in BANDIT_MAP
    bandit = BANDIT_MAP[args.bandit]()
    loss = runner.run_bandit(bandit)
    print(loss)