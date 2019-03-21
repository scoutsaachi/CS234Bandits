import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
# from brokenaxes import brokenaxes

# FOLDERS = ["constant", "clinical", "linear", "thompson", "random", "lasso", "knnucb"]

# ensemble plot: cut off at around 2000
# FOLDERS = ["random", "clinical", "hyper", "randhyper"]

alphas = ["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1"]

def extract(filename):
    print(filename)
    return pkl.load(open(filename, "rb"))

def extract_pickles(folders, run_ten):
    m = {}
    if run_ten:
        name = "runten_"
    else:
        name = "runone_"
    for x in folders:
        filename = "paper_results/bernoulli/%s/%s%s.pkl" % (x, name, x)
        print(filename)
        value = extract(filename)
        m[x] = value
    return m

def extract_reward_pickles(folders):
    m = {}
    for f in folders:
        vals = []
        for alpha in alphas:
            filename = "paper_results/rewards_%s/runten_%s.pkl" % (f, alpha)
            vals.append(extract(filename))
        m[f] = vals
    return m

def graph_bar():
    mapped = extract_reward_pickles(['linear'])["linear"]
    means = [[], [], []]
    errs = [[], [], []]
    for val in mapped:
        _, _, count, _ = val
        for a in range(3):
            N = float(sum(list(count[0].values())))
            count_actions = [z[a]/N for z in count]
            means[a].append(np.mean(count_actions))
            errs[a].append(2*np.std(count_actions))
    barWidth = 0.25
    r1 = np.arange(len(alphas))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    x_s = [r1, r2, r3]

    legend_list = ["low", "medium", "high"]
    for a in range(3):
        y = means[a]
        x = np.arange(len(alphas))
        plt.bar(x_s[a], means[a], label=legend_list[a], width=barWidth)
    plt.legend(ncol=3, frameon=False)
    plt.xticks([r + barWidth for r in range(len(alphas))], alphas)
    plt.xlabel(r'$\alpha$')
    plt.ylabel("Fraction of actions assigned to dose")
        # plt.bar(x, means[a], yerr=errs[a], capsize=3)
    plt.title(r"Proportion of actions assigned to dose versus $\alpha$")
    plt.show()
    print(means, errs)

def graph_reward(folders):
    mapping = extract_reward_pickles(folders)
    x = np.arange(0, 1.1, 0.1)
    for f in folders:
        means = []
        errs = []
        mapped = mapping[f]
        for val in mapped:
            regret, corr_frac, count, policy_count = val
            compute_val = regret
            means.append(np.mean(compute_val))
            errs.append(2*np.std(compute_val))
        plt.errorbar(x, means, yerr=errs, capsize=3, fmt=".-", label=f)
    # plt.xticks(x, folders)
    plt.legend()
    plt.ylabel("Average Regret")
    plt.title("Regret over 10 Runs for varying alpha")
    plt.show()

def graph_time(folders, graph_type="regret", trunc=False):
    trunc_value = 1000
    mapping = extract_pickles(folders, False)
    x = None
    for key,value in mapping.items():
        (regrets, corr_fracs, counts, policy_counts) = value
        if x is None:
            x = np.arange(len(regrets))
        if graph_type == "regret":
            compute_val = regrets
        elif graph_type == "corr_frac":
            compute_val = corr_fracs
        if trunc:
            compute_val = compute_val[:trunc_value]
            x = x[:trunc_value]
        plt.plot(x, compute_val, label=key)
    if graph_type == "regret":
        plt.title("Regret over time")
        plt.ylabel("Regret")
        plt.xlabel("Time")
    elif graph_type == "corr_frac":
        plt.title("Fraction Correct over time")
        plt.ylabel("Fraction Correct")
        plt.xlabel("Time")
    plt.legend(ncol=3, frameon=False)
    plt.show()

def graph_agg(folders, graph_type="regret"):
    mapping = extract_pickles(folders, True)
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
    # plt.errorbar(x, means, yerr=errs, capsize=3, fmt=".")
    plt.bar(x, means, yerr=errs, tick_label=folders, capsize=3)
    # plt.xticks(x, folders)
    if graph_type == "regret":
        plt.ylabel("Averaged Regret")
        plt.title("Regret over 10 Runs")
    if graph_type == "corr_frac":
        plt.ylabel("Averaged Fraction Correct")
        plt.title("Fraction Correct over 10 Runs")
    # plt.bar(x, means, yerr=errs, tick_label=folders, capsize=3)
    plt.show()

def graph_times(folders):
    graph_time(folders, "regret")
    graph_time(folders, "corr_frac")
    graph_time(folders, "regret", trunc=True)
    graph_time(folders, "corr_frac", trunc=True)

def graph_aggs(folders):
    graph_agg(folders, "regret")
    graph_agg(folders, "corr_frac")

def compute_graphs():
    not_ensemble_folders = ["constant", "clinical", "linear", "thompson", "random", "lasso", "knnucb"]
    graph_times(not_ensemble_folders)

    ensemble_folders = ["random", "clinical", "hyper", "randhyper"]
    graph_times(ensemble_folders)

    all_folders = ["constant", "clinical", "linear", "thompson", "random", "lasso", "knnucb", "hyper", "randhyper"]
    graph_aggs(all_folders)

    graph_reward(["clinical", "random", "constant", "linear", "thompson"])
    graph_bar()

def compute_accuracies(folders):
    mapping = extract_pickles(folders, True)
    accuracies = {}
    for f in folders:
        _, corr_frac, _, _ = mapping[f]
        mean = np.mean(corr_frac)
        print(f, mean)

# compute_graphs()
compute_accuracies(["constant", "clinical", "linear", "thompson", "random", "lasso", "knnucb", "hyper", "randhyper"])


# print(extract_reward_pickles(["clinical"]))
# graph_time(FOLDERS, "corr_frac")

    # for f in folder:
# print(extract_pickles(FOLDERS, False))
# compute_graphs()