import csv
import numpy as np

with open('clean.csv') as f:
    header_str = f.readline().strip()

data = np.loadtxt("clean.csv", delimiter=",", skiprows=1)
labels = np.expand_dims(data[:, -1], axis=1)
data = data[:, :-1]

max_val = data.max(axis=1, keepdims=True)
min_val = data.min(axis=1, keepdims=True)
std = data.std(axis=1, keepdims=True)
mean = data.mean(axis=1, keepdims=True)

normalized = (data - mean)/std
max_min = (data - min_val)/(max_val - min_val)

normalized = np.concatenate((normalized, labels), axis=1)
max_min = np.concatenate((max_min, labels), axis=1)

np.savetxt("maxmin.csv", max_min, delimiter=',', header=header_str)
np.savetxt("normalized.csv", normalized, delimiter=',', header=header_str)
