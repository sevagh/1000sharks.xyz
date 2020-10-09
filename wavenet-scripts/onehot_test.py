import tensorflow as tf
import scipy
from scipy.io.wavfile import read as wav_read
import librosa
import sys
import matplotlib.pyplot as plt
from datetime import datetime

quantization_channels=2**8
batch_size = 1


def mu_law_encode(audio):
    '''Quantizes waveform amplitudes.'''
    with tf.name_scope('encode'):
        mu = tf.to_float(quantization_channels - 1)
        # Perform mu-law companding transformation (ITU-T, 1988).
        # Minimum operation is here to deal with rare large amplitudes caused
        # by resampling.
        safe_audio_abs = tf.minimum(tf.abs(audio), 1.0)
        magnitude = tf.log1p(mu * safe_audio_abs) / tf.log1p(mu)
        signal = tf.sign(audio) * magnitude
        # Quantize signal to the specified number of levels.
        return tf.to_int32((signal + 1) / 2 * mu + 0.5)


def _one_hot(input_batch):
    '''One-hot encodes the waveform amplitudes.
    This allows the definition of the network as a categorical distribution
    over a finite set of possible amplitudes.
    '''
    with tf.name_scope('one_hot_encode'):
        encoded = tf.one_hot(
            input_batch,
            depth=quantization_channels,
            dtype=tf.float32)
        shape = [1, -1, quantization_channels]
        encoded = tf.reshape(encoded, shape)
    return encoded


def time_to_batch(value, dilation, name=None):
    with tf.name_scope('time_to_batch'):
        shape = tf.shape(value)
        pad_elements = dilation - 1 - (shape[1] + dilation - 1) % dilation
        padded = tf.pad(value, [[0, 0], [0, pad_elements], [0, 0]])
        reshaped = tf.reshape(padded, [-1, dilation, shape[2]])
        transposed = tf.transpose(reshaped, perm=[1, 0, 2])
        return tf.reshape(transposed, [shape[0] * dilation, -1, shape[2]])


def batch_to_time(value, dilation, name=None):
    with tf.name_scope('batch_to_time'):
        shape = tf.shape(value)
        prepared = tf.reshape(value, [dilation, -1, shape[2]])
        transposed = tf.transpose(prepared, perm=[1, 0, 2])
        return tf.reshape(transposed,
                          [tf.div(shape[0], dilation), -1, shape[2]])

if __name__ == '__main__':
    x, _ = librosa.load(sys.argv[1], mono=True) 

    # keep audio small
    print(x.shape)
    x = x[:1024]
    print(x.shape)

    mu_law_encoded = mu_law_encode(x)
    encoded = _one_hot(mu_law_encoded)

    with tf.Session() as sess:
        #encoded_eval = encoded.eval()
        #mu_law_eval = mu_law_encoded.eval()
        #for i in range(x.shape[0]):
        #    print('waveform value: {0}'.format(x[i]))
        #    print('mu-law encoded value: {0}'.format(mu_law_eval[i]))
        #    print('one-hot encoded values: {0}'.format(encoded_eval[0][i]))

        for dilation in [1, 4, 16]:
            print('DILATION: {0}'.format(dilation))
            print(encoded)
            transformed = time_to_batch(encoded, dilation)
            print(transformed)
            restored = batch_to_time(transformed, dilation)
            print(restored)

            e = encoded.eval()
            t = transformed.eval()
            r = restored.eval()

            print(e[0][:16])
            print(t[0][:4])
            print(t[0][:1])
