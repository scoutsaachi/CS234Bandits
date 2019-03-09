import csv
import argparse
from collections import defaultdict
import numpy as np

def read_data_file(filename, skip_header=True):
    with open(filename, "r") as f:
        csvreader = csv.reader(f, delimiter=",")
        header = skip_header
        data = []
        labels = []
        for r in csvreader:
            if header:
                header = False
                continue
            data.append([float(x) for x in r[1:-1]]) # skip ID column
            labels.append(int(r[-1]))

    return data, labels

def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("datafile", help="name of the data file")
    parser.add_argument("bandit", help="name of bandit to run")
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
