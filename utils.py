import csv
import argparse
from collections import defaultdict
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
    return (data - mean)/stdev

def maxmin_normalize(data):
    maxv = np.max(data, axis=1, keepdims=True)
    minv = np.min(data, axis=1, keepdims=True)
    return (data - minv)/(maxv-minv)

def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("datafile", help="name of the data file")
    parser.add_argument("bandit", help="name of bandit to run")
    parser.add_argument("--alpha", type=float, default=-1)
    parser.add_argument("--process", type=str, default="none") # ["none", "maxmin", "norm"]
    return parser

def bucketize_action(dose):
    if dose < 21:
        return 0
    elif dose <= 49:
        return 1
    else:
        return 2

def history_index(history, idx, t_arr=[]):
    if len(t_arr) > 0:
        return np.array([history[t][idx] for t in t_arr])
    else:
        return np.array([row[idx] for row in history]) # returns all timesteps if no t_arr given
