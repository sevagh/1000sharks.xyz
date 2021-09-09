#!/usr/bin/env python3

import h5py
import sys
import argparse
import os
import numpy
import keras
from keras import layers
from keras import regularizers
from keras.models import Sequential, load_model
from keras.layers import Dense, GRU
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas

model_file = "./model/wavenet.h5"
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

checkpoint_file = "./logdir/wavenet.ckpt"


def parse_args():
    parser = argparse.ArgumentParser(
        prog="nn_model",
        description="generate, train, and save a neural network",
    )

    parser.add_argument("data_file", help="hdf5 file with training data")
    return parser.parse_args()


def main():
    args = parse_args()

    reg = 0.0000001

    inputs = keras.Input(shape=(None, 512, 256), name="waveform_in")

    x = Dense(512, kernel_regularizer=regularizers.l2(reg), name="sample_mlp_1")(inputs)
    x = Dense(512, kernel_regularizer=regularizers.l2(reg), name="sample_mlp_2")(x)

    # one-hot output
    outputs = layers.Dense(512, name="waveform_out", activation="sigmoid")(x)

    model = None
    try:
        model = load_model(model_file)
    except IOError:
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="mae", metrics=["mae"])

    model.summary()

    monitor = EarlyStopping(monitor="loss", patience=5)

    checkpoint = ModelCheckpoint(
        checkpoint_file,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
        save_freq="epoch",
    )

    with h5py.File(args.data_file, "r") as hf:
        data = hf["data"][:3_000_000]

        X = data[:, :512]  # first 512 samples are the input waveform
        Y = data[:, 512:]  # next 512 samples are the output waveform

        # split into 90/10. then pass validation_split to keras fit
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, train_size=0.9, test_size=0.1, random_state=42
        )

        X_train = numpy.reshape(X_train, (X_train.shape[0], 1, 512))
        Y_train = numpy.reshape(Y_train, (Y_train.shape[0], 1, 512))
        X_test = numpy.reshape(X_test, (X_test.shape[0], 1, 512))
        Y_test = numpy.reshape(Y_test, (Y_test.shape[0], 1, 512))

        # train on training data (duh), with one-hot encoding of 256-bit mu-law encoded integers
        try:
            model.fit(
                tf.one_hot(X_train, 256),
                tf.one_hot(Y_train, 256),
                batch_size=16,
                epochs=100,
                callbacks=[monitor, checkpoint],
                validation_split=0.1,  # reserve 10% of the 90% of train data for external validation
                verbose=1,
            )
        except KeyboardInterrupt:
            print("interrupted by ctrl-c, saving model")
            model.save(model_file)

        train_scores = model.evaluate(X_train, Y_train)
        print(
            "train scores: %s: %.2f%%" % (model.metrics_names[1], train_scores[1] * 100)
        )

        test_scores = model.evaluate(X_test, Y_test)
        print(
            "test scores: %s: %.2f%%" % (model.metrics_names[1], test_scores[1] * 100)
        )

        print("saving model")
        model.save(model_file)


if __name__ == "__main__":
    sys.exit(main())
