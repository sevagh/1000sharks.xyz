#!/usr/bin/env python3

import h5py
import sys
import argparse
import os
from rnn_bfcc import Bfcc
import numpy
from keras.models import load_model
import tensorflow
import soundfile
from rnn_model import _custom_mae
from keras.utils.generic_utils import get_custom_objects
get_custom_objects().update({"_custom_mae": _custom_mae})

physical_devices = tensorflow.config.list_physical_devices("GPU")
tensorflow.config.experimental.set_memory_growth(physical_devices[0], enable=True)

model_file = "./model/trnnsient.h5"
checkpoint_file = "./logdir/trnnsient.ckpt"


def parse_args():
    parser = argparse.ArgumentParser(
        prog="inference",
        description="Apply tRNNsient to a wav file",
    )

    parser.add_argument("input_wav", help="input clip to apply tRNNsient inference")
    parser.add_argument("output_wav", help="path to output wav file")
    return parser.parse_args()


def main():
    args = parse_args()

    model = None
    try:
        model = load_model(model_file)
    except Exception as e1:
        print('{0}, trying checkpoint'.format(str(e1)))
        try:
            model = load_model(checkpoint_file)
        except Exception as e2:
            print('{0}, exiting'.format(str(e2)))
            print(
                "could not load models, tried {0} and {1}".format(
                    model_file, checkpoint_file
                )
            )
            sys.exit(1)

    model.summary()

    bfcc_inference = Bfcc(hop_size=512)

    # pass the model predict function as a callback to the bfcc
    # it will apply band-wise gain after calculating the target bfccs from the learned neural network
    out = bfcc_inference.apply_bfcc_gain_inference(args.input_wav, model.predict)

    soundfile.write(args.output_wav, out, bfcc_inference.sample_rate)

    return 0


if __name__ == "__main__":
    sys.exit(main())
