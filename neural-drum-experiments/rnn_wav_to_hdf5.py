#!/usr/bin/env python3

import h5py
import sys
import argparse
import os
from rnn_bfcc import Bfcc
import numpy
import multiprocessing
import itertools


def parse_args():
    parser = argparse.ArgumentParser(
        prog="wav_to_hdf5",
        description="Convert prepared wav training data into hdf5 matrix for Keras training",
    )
    parser.add_argument("out_file", help="path to write hdf5 data file")
    parser.add_argument(
        "--n-pool",
        type=int,
        default=14,
        help="size of python multiprocessing pool (default 14)",
    )

    return parser.parse_args()


data_dir = "./data"
testcases = sorted([f for f in os.listdir(data_dir) if f.endswith(".wav")])
bfcc = Bfcc(hop_size=512)


def compute_single_testcase(i):
    all_ndarray_rows = []

    data_in = os.path.join(
        data_dir, testcases[i + 2]
    )  # input is the percussive separation
    data_out = os.path.join(data_dir, testcases[i])  # output is the clean drum track

    in_bfccs = bfcc.audio_to_bfccs(data_in)
    out_bfccs = bfcc.audio_to_bfccs(data_out)

    # 24 bfcc coefficients
    assert all(len(b) == 24 for b in in_bfccs)
    assert all(len(b) == 24 for b in out_bfccs)

    for bfcc_pairs in zip(in_bfccs, out_bfccs):
        all_ndarray_rows.append(numpy.concatenate((bfcc_pairs[0], bfcc_pairs[1])))

    print(
        "returning {0} ndarrays of bfcc pairs for testcase {1}".format(
            len(all_ndarray_rows), int(numpy.floor(i / 3))
        )
    )

    return all_ndarray_rows


def main():
    args = parse_args()

    pool = multiprocessing.Pool(args.n_pool)
    outputs = list(
        itertools.chain.from_iterable(
            pool.map(compute_single_testcase, range(0, len(testcases), 3))
        )
    )

    with h5py.File(args.out_file, "w") as hf:
        hf.create_dataset("data", data=outputs)

    return 0


if __name__ == "__main__":
    sys.exit(main())
