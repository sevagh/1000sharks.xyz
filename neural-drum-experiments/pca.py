#!/usr/bin/env python

import sys
from scipy.io import wavfile
from sklearn.decomposition import PCA
import numpy as np


def pca_reduce(signal, n_components, block_size=1024):
    # First, zero-pad the signal so that it is divisible by the block_size
    samples = len(signal)
    hanging = block_size - np.mod(samples, block_size)
    padded = np.lib.pad(signal, (0, hanging), 'constant', constant_values=0)

    # Reshape the signal to have 1024 dimensions
    reshaped = padded.reshape((len(padded) // block_size, block_size))

    # Second, do the actual PCA process
    pca = PCA(n_components=n_components)
    pca.fit(reshaped)

    transformed = pca.transform(reshaped)
    reconstructed = pca.inverse_transform(transformed).reshape((len(padded)))
    return pca, transformed, reconstructed


def main(infile):
    fs, x = wavfile.read(infile)
    _, _, reconstructed = pca_reduce(x, 512, 1024)
    wavfile.write('recon.wav', fs, reconstructed)

    return 0


if __name__ == '__main__':
    try:
        infile = sys.argv[1]
    except IndexError:
        print('usage {0} /path/to/wav/file'.format(sys.argv[0]), file=sys.stderr)
        sys.exit(1)

    sys.exit(main(infile))
