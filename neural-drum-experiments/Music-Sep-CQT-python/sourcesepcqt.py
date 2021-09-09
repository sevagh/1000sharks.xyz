#!/usr/bin/env python3

import argparse
import multiprocessing
import sys
import json
import numpy
import librosa
import scipy
import scipy.io.wavfile

from cqtsep import cqtsep


def main():
    parser = argparse.ArgumentParser(
        prog="cqtsep",
        description="CQT-based harmonic-percussive-vocal source separation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--n-pool",
        type=int,
        default=multiprocessing.cpu_count() - 1,
        help="How many threads to use in multiprocessing pool",
    )

    parser.add_argument("wav_in", help="input wav file")
    parser.add_argument("wav_out_prefix", help="output wav file prefix")

    args = parser.parse_args()

    x, fs = librosa.load(args.wav_in, sr=None, mono=False)

    pool = multiprocessing.Pool(args.n_pool)

    print("Performing separation")
    xh, xp, xv = cqtsep(x, fs, pool)

    print("Writing separated outputs with prefix: {0}".format(args.wav_out_prefix))
    scipy.io.wavfile.write(args.wav_out_prefix+"_harmonic.wav", fs, xh)
    scipy.io.wavfile.write(args.wav_out_prefix+"_percussive.wav", fs, xp)
    scipy.io.wavfile.write(args.wav_out_prefix+"_vocal.wav", fs, xv)


if __name__ == "__main__":
    main()
