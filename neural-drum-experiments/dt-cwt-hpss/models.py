#!/usr/bin/env python3

import h5py
import sys
import argparse
import os
import numpy
from tensorflow import keras
from sklearn.model_selection import train_test_split
import tensorflow

perclo_model = "./trained-models/perc_lo.h5"
harmlo_model = "./trained-models/harm_lo.h5"
perchi_model = "./trained-models/perc_hi.h5"
harmhi_model = "./trained-models/harm_hi.h5"

perclo_ckpt = "./logdir/perc_lo.ckpt"
harmlo_ckpt = "./logdir/harm_lo.ckpt"
perchi_ckpt = "./logdir/perc_hi.ckpt"
harmhi_ckpt = "./logdir/harm_hi.ckpt"


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

    inputs = keras.layers.Input(shape=(None, 2048), name="dtcwt_lowpass_in")

    # 2 hidden layers
    x = keras.layers.Dense(2048, activation="relu", name="input_dense")(inputs)
    x = keras.layers.Dense(2048, activation="relu", name="hidden_layer_1")(x)

    # continuous-valued output layer, no activation
    outputs = keras.layers.Dense(2048, name="dtcwt_lowpass_out")(x)

    #model_perclo = None
    model_perchi = None
    #model_harmlo = None
    #model_harmhi = None
    try:
        model_perchi = keras.models.load_model(perchi_model)
    except IOError:
        model_perchi = keras.Model(inputs=inputs, outputs=outputs)
        model_perchi.compile(optimizer="adam", loss="mae", metrics=["mae"])

    model_perchi.summary()

    monitor = keras.callbacks.EarlyStopping(monitor="loss", patience=5)

    checkpoint_perchi = keras.callbacks.ModelCheckpoint(
        perchi_ckpt,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
        save_freq="epoch",
    )

    with h5py.File(args.data_file, "r") as hf:
        for k in hf.keys():
            print(k)
            if k == 'train_perc_hi':
                print('training model for percussive low-pass coefficients')
                data = hf[k][:]

                print(data.shape)

                X = numpy.copy(data[:, :2048])
                Y = numpy.copy(data[:, 2048:])

                # split into 60/40. then pass validation_split to keras fit
                X_train, X_test, Y_train, Y_test = train_test_split(
                    X, Y, train_size=0.6, test_size=0.4, random_state=42
                )

                X_train = numpy.reshape(X_train, (X_train.shape[0], 1, 2048))
                Y_train = numpy.reshape(Y_train, (Y_train.shape[0], 1, 2048))
                X_test = numpy.reshape(X_test, (X_test.shape[0], 1, 2048))
                Y_test = numpy.reshape(Y_test, (Y_test.shape[0], 1, 2048))

                print(X_train.shape)
                print(Y_train.shape)
                print(X_test.shape)
                print(Y_test.shape)

                # train on training data (duh)
                try:
                    model_perchi.fit(
                        X_train,
                        Y_train,
                        batch_size=256,
                        epochs=1500,
                        callbacks=[monitor, checkpoint_perchi],
                        validation_split=0.1,  # reserve 50% of the 40% of train data for external validation
                                               # resulting in a final 60/20/20 split
                        verbose=1,
                    )
                except KeyboardInterrupt:
                    print("interrupted by ctrl-c, saving model")
                    model_perchi.save(perchi_model)

                train_scores = model_perchi.evaluate(X_train, Y_train)
                print(
                    "train scores: %s: %.2f%%" % (model_perchi.metrics_names[1], train_scores[1] * 100)
                )

                test_scores = model_perchi.evaluate(X_test, Y_test)
                print(
                    "test scores: %s: %.2f%%" % (model_perchi.metrics_names[1], test_scores[1] * 100)
                )

                print("saving model")
                model_perchi.save(perchi_model)


if __name__ == "__main__":
    sys.exit(main())
