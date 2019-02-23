import csv 
import argparse


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
            data.append([float(x) for x in r[:-1]])
            labels.append(int(r[-1]))
    return data, labels

def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("datafile", help="name of the data file")
    parser.add_argument("bandit", help="name of bandit to run")
    return parser