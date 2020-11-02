# Ignore a bunch of deprecation warnings
import warnings
warnings.filterwarnings("ignore")

import ddsp
import ddsp.training
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import soundfile
import matplotlib.pyplot as plt
import numpy


sample_rate = 16000 #DEFAULT_SAMPLE_RATE  # 16000
n_frames = 1000
hop_size = 64
n_samples = n_frames * hop_size

# Amplitude [batch, n_frames, 1].
# Make amplitude linearly decay over time.
amps = np.linspace(1.0, -3.0, n_frames)
amps = amps[np.newaxis, :, np.newaxis]

# Harmonic Distribution [batch, n_frames, n_harmonics].
# Make harmonics decrease linearly with frequency.
n_harmonics = 20
harmonic_distribution = np.ones([n_frames, 1]) * np.linspace(1.0, -1.0, n_harmonics)[np.newaxis, :]
harmonic_distribution = harmonic_distribution[np.newaxis, :, :]

# Fundamental frequency in Hz [batch, n_frames, 1].
f0_hz = 440.0 * np.ones([1, n_frames, 1])

# Create synthesizer object.
additive_synth = ddsp.synths.Additive(n_samples=n_samples,
                                      scale_fn=ddsp.core.exp_sigmoid,
                                      sample_rate=sample_rate)

# Generate some audio.
audio = additive_synth(amps, harmonic_distribution, f0_hz)

a = audio.numpy()[0] # get numpy array from the tensor expression

try:
    soundfile.write('synth_demo_py.wav', a, sample_rate)
except:
    pass

_, _, _, im = plt.specgram(a, Fs=sample_rate, NFFT=1024, noverlap=256)
plt.show()
