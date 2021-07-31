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
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K
from sklearn.model_selection import train_test_split
import tensorflow
from keras.utils.generic_utils import get_custom_objects

model_file = "./model/trnnsient.h5"
physical_devices = tensorflow.config.list_physical_devices("GPU")
tensorflow.config.experimental.set_memory_growth(physical_devices[0], enable=True)

checkpoint_file = "./logdir/trnnsient.ckpt"


def _custom_mae(y_true, y_pred):
    # real y pred is the gain times the input which is the first half of y pred
    y_pred_actual = y_pred[:, :, :24] * y_pred[:, :, 24:]
    y_true_actual = y_true[:, :, 24:]

    y_true_actual = K.cast(y_true_actual, y_pred_actual.dtype)
    diff = K.abs((y_true_actual - y_pred_actual) / K.clip(K.abs(y_true_actual),
                                                   K.epsilon(),
                                                   None))
    return 100. * K.mean(diff, axis=-1)


get_custom_objects().update({"_custom_mae": _custom_mae})


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

    inputs = keras.Input(shape=(None, 24), name="bfcc_in")
    x = layers.Dense(14, activation="tanh", name="input_dense")(inputs)
    gru1 = layers.GRU(
        48,
        activation="tanh",
        return_sequences=True,
        recurrent_activation="sigmoid",
        kernel_regularizer=regularizers.l2(reg),
        recurrent_regularizer=regularizers.l2(reg),
        name="gru_1",
    )(x)

    tmp = keras.layers.concatenate([x, gru1])
    gru2 = layers.GRU(
        96,
        activation="relu",
        return_sequences=True,
        recurrent_activation="sigmoid",
        kernel_regularizer=regularizers.l2(reg),
        recurrent_regularizer=regularizers.l2(reg),
        name="gru_2",
    )(tmp)

    # gain between 0 and 1 like the original
    outputs = layers.Dense(24, activation="sigmoid", name="bfcc_gain")(gru2)

    out = keras.layers.concatenate([inputs, outputs])

    model = None
    try:
        model = load_model(model_file)
    except IOError:
        model = keras.Model(inputs=inputs, outputs=out)
        model.compile(optimizer="adam", loss=_custom_mae, metrics=["mae"])

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
        data = hf["data"][:]

        data /= 255.0 # scale the whole thing down

        print(data[:2])

        X = numpy.copy(data[:, :24])  # first 24 bfccs are the input
        Y = numpy.copy(data[:, :])    # second 24 bfccs are the output

        # split into 90/10. then pass validation_split to keras fit
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, train_size=0.9, test_size=0.1, random_state=42
        )

        X_train = numpy.reshape(X_train, (X_train.shape[0], 1, 24))
        Y_train = numpy.reshape(Y_train, (Y_train.shape[0], 1, 48))
        X_test = numpy.reshape(X_test, (X_test.shape[0], 1, 24))
        Y_test = numpy.reshape(Y_test, (Y_test.shape[0], 1, 48))

        # train on training data (duh)
        try:
            model.fit(
                X_train,
                Y_train,
                batch_size=256,
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
