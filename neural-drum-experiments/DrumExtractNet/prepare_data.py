#!/usr/bin/env python3

import hickle
import sys
import argparse
import os
import numpy
import multiprocessing
import itertools
from scipy.io import wavfile


def parse_args():
    parser = argparse.ArgumentParser(
        prog="prepare_data",
        description="Convert prepared wav training data into hdf5 matrix for model training",
    )
    parser.add_argument("--out_file", type=str, default="data.hdf5", help="path to write hdf5 data file")
    parser.add_argument("--sample_time", type=float, default=100e-3)

    return parser.parse_args()


data_dir = "./data"
testcases = sorted([f for f in os.listdir(data_dir) if f.endswith(".wav")])


def main():
    args = parse_args()

    d = {}
    for i in range(0, len(testcases), 2):
        data_in = os.path.join(
            data_dir, testcases[i + 1]
        )  # input is the mix separation
        data_out = os.path.join(data_dir, testcases[i])  # output is the clean drum track

        in_rate, in_data = wavfile.read(data_in)
        out_rate, out_data = wavfile.read(data_out)

        sample_size = int(in_rate * args.sample_time)
        length = len(in_data) - len(in_data) % sample_size

        x = in_data[:length].reshape((-1, 1, sample_size)).astype(numpy.float32)
        y = out_data[:length].reshape((-1, 1, sample_size)).astype(numpy.float32)

        split = lambda d: numpy.split(d, [int(len(d) * 0.6), int(len(d) * 0.8)])

        x_train, x_valid, x_test = split(x)
        y_train, y_valid, y_test = split(y)

        try:
            d["x_train"] = numpy.concatenate((d["x_train"], x_train))
        except KeyError:
            d["x_train"] = x_train

        try:
            d["x_valid"] = numpy.concatenate((d["x_valid"], x_valid))
        except KeyError:
            d["x_valid"] = x_valid

        try:
            d["x_test"] = numpy.concatenate((d["x_test"], x_test))
        except KeyError:
            d["x_test"] = x_test

        try:
            d["y_train"] = numpy.concatenate((d["y_train"], y_train))
        except KeyError:
            d["y_train"] = y_train

        try:
            d["y_valid"] = numpy.concatenate((d["y_valid"], y_valid))
        except KeyError:
            d["y_valid"] = y_valid

        try:
            d["y_test"] = numpy.concatenate((d["y_test"], y_test))
        except KeyError:
            d["y_test"] = y_test

    d["mean"], d["std"] = d["x_train"].mean(), d["x_train"].std()

    # standardize
    for key in "x_train", "x_valid", "x_test":
        d[key] = (d[key] - d["mean"]) / d["std"]

    hickle.dump(d, args.out_file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
