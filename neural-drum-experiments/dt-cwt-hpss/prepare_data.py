#!/usr/bin/env python3

import dtcwt
import h5py
import sys
import argparse
import os
import numpy
import itertools
from scipy.io import wavfile

FIXED_SAMPLE_RATE = 44100.0
#MAX_LEN = int(20*FIXED_SAMPLE_RATE)
MAX_LEN = 4096 # sized hops


def parse_args():
    parser = argparse.ArgumentParser(
        prog="wav_to_hdf5",
        description="Convert prepared wav training data into hdf5 matrix for Keras training",
    )
    parser.add_argument("--testcases", type=int, default=-1, help="number of testcases to load")
    parser.add_argument("out_file", help="destination hdf5 data file")

    return parser.parse_args()


data_dir = "./data"
testcases = sorted([f for f in os.listdir(data_dir) if f.endswith(".wav")])
transform = dtcwt.Transform1d()


def compute_single_testcase(i):
    harm_hi_ndarray_rows = []
    perc_hi_ndarray_rows = []

    data_mix = os.path.join(data_dir, testcases[i+1])
    data_perc = os.path.join(
        data_dir, testcases[i + 2]
    )
    data_harm = os.path.join(data_dir, testcases[i])

    _, mix_waveform = wavfile.read(data_mix)
    _, harm_waveform = wavfile.read(data_harm)
    _, perc_waveform = wavfile.read(data_perc)


    nchunk = int(numpy.floor(len(mix_waveform)//MAX_LEN))
    print(nchunk)

    for i in range(nchunk-1):
        # fix length to 20s for simplification
        mix_waveform_chunk = mix_waveform[MAX_LEN*i:MAX_LEN*(i+1)]
        perc_waveform_chunk = perc_waveform[MAX_LEN*i:MAX_LEN*(i+1)]
        harm_waveform_chunk = harm_waveform[MAX_LEN*i:MAX_LEN*(i+1)]

        # 1 level for simplification
        mix_cwt = transform.forward(mix_waveform_chunk, nlevels=1)
        harm_cwt = transform.forward(harm_waveform_chunk, nlevels=1)
        perc_cwt = transform.forward(perc_waveform_chunk, nlevels=1)

        harm_hi_ndarray_rows.append(numpy.concatenate((mix_cwt.highpasses[0], harm_cwt.highpasses[0])))
        perc_hi_ndarray_rows.append(numpy.concatenate((mix_cwt.highpasses[0], perc_cwt.highpasses[0])))

    return harm_hi_ndarray_rows, perc_hi_ndarray_rows


def main():
    args = parse_args()

    harm_hi = []
    perc_hi = []

    max_test = 3
    if args.testcases == -1:
        max_test = len(testcases)
    else:
        max_test = 3*args.testcases

    for testcase in range(0, max_test, 3):
        print('testcase {0}'.format(testcase))
        h, p = compute_single_testcase(testcase)
        harm_hi.extend(h)
        perc_hi.extend(p)

    with h5py.File(args.out_file, "w") as hf:
        hf.create_dataset("train_perc_hi", data=perc_hi)
        hf.create_dataset("train_harm_hi", data=harm_hi)

    return 0


if __name__ == "__main__":
    sys.exit(main())
