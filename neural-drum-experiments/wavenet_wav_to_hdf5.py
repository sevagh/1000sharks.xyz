#!/usr/bin/env python3

import tensorflow as tf
import h5py
import sys
import argparse
import os
import numpy
import multiprocessing
import itertools
from essentia.standard import (
    MonoLoader,
    FrameGenerator,
    Windowing,
)


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

hop_size = 256
frame_size = 512
sample_rate = 44100
window = Windowing(type="hann", size=frame_size, normalized=False, zeroPhase=False)


def mu_law_encode(audio, quantization_channels=256):
    """Quantizes waveform amplitudes."""
    mu = tf.compat.v1.to_float(quantization_channels - 1)
    # Perform mu-law companding transformation (ITU-T, 1988).
    # Minimum operation is here to deal with rare large amplitudes caused
    # by resampling.
    safe_audio_abs = tf.minimum(tf.abs(audio), 1.0)
    magnitude = tf.compat.v1.log1p(mu * safe_audio_abs) / tf.compat.v1.log1p(mu)
    signal = tf.sign(audio) * magnitude
    # Quantize signal to the specified number of levels.
    return tf.compat.v1.to_int32((signal + 1) / 2 * mu + 0.5)


def mu_law_decode(output, quantization_channels=256):
    mu = quantization_channels - 1
    # Map values back to [-1, 1].
    signal = 2 * (tf.compat.v1.to_float(output) / mu) - 1
    # Perform inverse of mu-law transformation.
    magnitude = (1 / mu) * ((1 + mu) ** abs(signal) - 1)
    return tf.compat.v1.sign(signal) * magnitude


def compute_single_testcase(i):
    all_ndarray_rows = []

    data_in = os.path.join(
        data_dir, testcases[i + 2]
    )  # input is the percussive separation
    data_out = os.path.join(data_dir, testcases[i])  # output is the clean drum track

    audio_in = MonoLoader(filename=data_in, sampleRate=sample_rate)()
    audio_out = MonoLoader(filename=data_out, sampleRate=sample_rate)()

    frames_in = []
    for frame_in in FrameGenerator(
        audio_in,
        frameSize=frame_size,
        hopSize=hop_size,
        startFromZero=True,
        validFrameThresholdRatio=1,
    ):
        frames_in.append(mu_law_encode(window(frame_in)))

    frames_out = []
    for frame_out in FrameGenerator(
        audio_out,
        frameSize=frame_size,
        hopSize=hop_size,
        startFromZero=True,
        validFrameThresholdRatio=1,
    ):
        frames_out.append(mu_law_encode(window(frame_out)))

    for frame_pairs in zip(frames_in, frames_out):
        all_ndarray_rows.append(numpy.concatenate((frame_pairs[0], frame_pairs[1])))

    print(
        "returning {0} ndarrays of samples for testcase {1}".format(
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
