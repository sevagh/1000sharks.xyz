#!/usr/bin/env python3

import h5py
import sys
import argparse
import os
import numpy
from keras.models import load_model
import tensorflow
import soundfile
from essentia.standard import MonoLoader, FrameGenerator, OverlapAdd
from wavenet_wav_to_hdf5 import (
    frame_size,
    hop_size,
    sample_rate,
    window,
    mu_law_encode,
    mu_law_decode,
)

physical_devices = tensorflow.config.list_physical_devices("GPU")
tensorflow.config.experimental.set_memory_growth(physical_devices[0], enable=True)

model_file = "./model/wavenet.h5"
checkpoint_file = "./logdir/wavenet.ckpt"


def parse_args():
    parser = argparse.ArgumentParser(
        prog="inference",
        description="Apply model inference to a wav file",
    )

    parser.add_argument("input_wav", help="input clip to apply inference")
    parser.add_argument("output_wav", help="path to output wav file")
    return parser.parse_args()


def main():
    args = parse_args()

    model = None
    try:
        model = load_model(model_file)
    except Exception as e1:
        print("{0}, trying checkpoint".format(str(e1)))
        try:
            model = load_model(checkpoint_file)
        except Exception as e2:
            print("{0}, exiting".format(str(e2)))
            print(
                "could not load models, tried {0} and {1}".format(
                    model_file, checkpoint_file
                )
            )
            sys.exit(1)

    model.summary()

    audio_in = MonoLoader(filename=args.input_wav, sampleRate=sample_rate)()

    # get the window coefficients for the COLA constraint
    window_default = window(numpy.ones(frame_size, dtype=numpy.float32))
    cola_factor = numpy.sum(numpy.multiply(window_default, window_default))

    # gain?
    olap = OverlapAdd(frameSize=frame_size, hopSize=hop_size, gain=1.0 / cola_factor)

    modified_audio = numpy.array(0)

    for frame_in in FrameGenerator(
        audio_in,
        frameSize=frame_size,
        hopSize=hop_size,
        startFromZero=True,
        validFrameThresholdRatio=1,
    ):
        frame_in = numpy.reshape(mu_law_encode(frame_in), (1, 1, 512))
        frame_pred = mu_law_decode(model.predict(frame_in))
        frame_pred = numpy.reshape(frame_pred, (512,))
        modified_audio = numpy.append(modified_audio, olap(frame_pred))

    soundfile.write(args.output_wav, modified_audio, sample_rate)

    return 0


if __name__ == "__main__":
    sys.exit(main())
