import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np

FOLDERS = ["constant", "clinical", "linear", "thompson2", "random", "lasso"]

# ensemble plot: cut off at around 2000
# FOLDERS = ["random", "clinical", "hyper", "randhyper"]


# FOLDERS = ["constant", "hyper", "lasso",  "linear", "randhyper", "thompson", "thompson2", "random"]

def extract(filename):
    return pkl.load(open(filename, "rb"))

def extract_pickles(folders, run_ten):
    m = {}
    if run_ten:
        name = "runten_"
    else:
        name = "runone_"
    for x in folders:
        filename = "paper_results/bernoulli/%s/%s%s.pkl" % (x, name, x)
        value = extract(filename)
        m[x] = value
    return m

def graph_time(folders, graph_type="regret"):
    mapping = extract_pickles(FOLDERS, False)
    x = None
    for key,value in mapping.items():
        (regrets, corr_fracs, counts, policy_counts) = value
        if x is None:
            x = np.arange(len(regrets))
        if graph_type == "regret":
            plt.plot(x, regrets, label=key)
        elif graph_type == "corr_frac":
            plt.plot(x, corr_fracs, label=key)
    if graph_type == "regret":
        plt.title("Regret over time")
    elif graph_type == "corr_frac":
        plt.title("Correction Fraction over time")
    plt.legend()
    plt.show()

def graph_agg(folders, graph_type="regret"):
    mapping = extract_pickles(FOLDERS, True)
    means = []
    errs = []
    for f in folders:
        regret, corr_frac, count, policy_count = mapping[f]
        if graph_type == "regret":
            compute_val = regret
        elif graph_type == "corr_frac":
            compute_val = corr_frac
        means.append(np.mean(compute_val))
        errs.append(2*np.std(compute_val))
    x = np.arange(len(folders))
    plt.errorbar(x, means, yerr=errs, capsize=3, fmt=".")
    plt.xticks(x, folders)
    if graph_type == "regret":
        plt.ylabel("Regret")
        plt.title("Regret over 10 Runs")
    if graph_type == "corr_frac":
        plt.ylabel("Correct Fraction")
        plt.title("Correction Fraction over 10 Runs")
    # plt.bar(x, means, yerr=errs, tick_label=folders, capsize=3)
    plt.show()

graph_time(FOLDERS, "corr_frac")

    # for f in folder:
# print(extract_pickles(FOLDERS, False))
