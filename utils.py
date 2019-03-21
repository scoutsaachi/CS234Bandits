import argparse
import csv
from collections import defaultdict
import pickle as pkl

import numpy as np


def read_data_file(filename, skip_header=True):
    skiprows = 0
    if skip_header:
        skiprows = 1
    data = np.loadtxt(filename, delimiter=",", skiprows=skiprows)
    labels = data[:, -1]
    data = data[:, 1:-1]
    return data, labels


def normalize(data):
    mean = np.mean(data, axis=1, keepdims=True)
    stdev = np.std(data, axis=1, keepdims=True)
    return (data - mean) / stdev


def maxmin_normalize(data):
    maxv = np.max(data, axis=0, keepdims=True)
    minv = np.min(data, axis=0, keepdims=True)
    pkl.dump((maxv, minv), open("maxmin.pkl", "wb"))
    # assert False
    # print(maxv)
    # print(minv)
    # assert False
    return (data - minv) / (maxv - minv)


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("datafile", help="name of the data file")
    parser.add_argument("bandit", help="name of bandit to run")
    parser.add_argument("result_file", help="name of result file")
    parser.add_argument("--alpha", type=float, default=-1)
    parser.add_argument(
        "--process", type=str, default="none")  # ["none", "maxmin", "norm"]
    parser.add_argument("--runten", action='store_true')  # run ten
    return parser


def bucketize_action(dose):
    if dose < 21:
        return 0
    elif dose <= 49:
        return 1
    else:
        return 2


def history_index(history, idx, t_arr=[], add_one=False):
    contexts = np.squeeze(np.array([history[t][idx] for t in t_arr]))
    if add_one:
        ones_arr = np.ones((len(t_arr), 1))
        contexts = np.hstack((ones_arr, contexts))
    return contexts