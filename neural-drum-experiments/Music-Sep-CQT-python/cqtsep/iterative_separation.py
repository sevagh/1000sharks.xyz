import numpy
from librosa.decompose import hpss
from librosa.core import stft, istft, cqt, icqt
from librosa.util import fix_length
from scipy.signal import medfilt
from .transient_shaper import multiband_transient_shaper


'''
separate a mix into its harmonic, percussive, and vocal constituents
using iterative median filtering on the CQT and STFT, and transient shaping
'''
def cqtsep(
    x,
    fs,
    pool,
):
    print('Iteration 1: harmonic separation with CQT (bins_per_octave = 96)')
    # first iteration, high frequency resolution CQT
    C1 = cqt(x, sr=fs, bins_per_octave=96, hop_length=3434)#, n_bins=7*bins_per_octave)
    Cmag1 = numpy.abs(C1)

    # fitzgerald soft masking
    Mh1, Mp1 = hpss(Cmag1, power=2.0, margin=1.0, kernel_size=(17, 7), mask=True)

    Ch1 = numpy.multiply(Mh1, C1)
    Cp1 = numpy.multiply(Mp1, C1)

    xh1 = fix_length(icqt(Ch1, sr=fs, bins_per_octave=96, hop_length=3434), len(x))
    xp1 = fix_length(icqt(Cp1, sr=fs, bins_per_octave=96, hop_length=3434), len(x))

    print('Iteration 2: vocal separation with CQT (bins_per_octave = 24)')
    # second iteration, vocal with low frequency resolution CQT
    xim2 = xp1

    C2 = cqt(xim2, sr=fs, bins_per_octave=24, hop_length=860)#, n_bins=7*bins_per_octave)
    Cmag2 = numpy.abs(C2)

    # fitzgerald soft masking
    Mh2, Mp2 = hpss(Cmag2, power=2.0, margin=1.0, kernel_size=(17, 7), mask=True)

    Ch2 = numpy.multiply(Mh2, C2)
    Cp2 = numpy.multiply(Mp2, C2)

    xh2 = fix_length(icqt(Ch2, sr=fs, bins_per_octave=24, hop_length=860), len(x))
    xp2 = fix_length(icqt(Cp2, sr=fs, bins_per_octave=24, hop_length=860), len(x))

    print('Iteration 3: percussive separation with STFT (window = 1024)')
    # third iteration, percussive stft
    xim3 = xp1 + xp2;

    S3 = stft(
        xim3,
        n_fft=2048,
        win_length=1024,
        hop_length=256,
    )
    Smag3 = numpy.abs(S3)

    # fitzgerald soft masking
    _, Mp3 = hpss(S3, power=2.0, margin=1.0, kernel_size=(17, 17), mask=True)

    Sp3 = numpy.multiply(Mp3, S3)

    xp3 = fix_length(istft(Sp3, win_length=1024, hop_length=256, dtype=x.dtype), len(x))

    print('Iteration 4: harmonic refinement with CQT (bins_per_octave = 12)')
    # fourth iteration, refine harmonic
    x_vocal = xh2
    x_percussive = xp2 + xp3
    x_harmonic = xh1

    C4 = cqt(x_harmonic, sr=fs, bins_per_octave=12, hop_length=432)#, n_bins=7*bins_per_octave)
    Cv4 = cqt(x_vocal, sr=fs, bins_per_octave=12, hop_length=432)#, n_bins=7*bins_per_octave, )
    Cp4 = cqt(x_percussive, sr=fs, bins_per_octave=12, hop_length=432)#, n_bins=7*bins_per_octave)

    Cmag4 = numpy.abs(C4)
    Cvmag4 = numpy.abs(Cv4)
    Cpmag4 = numpy.abs(Cp4)

    H4 = numpy.power(Cmag4, 2.0)
    V4 = numpy.power(Cvmag4, 2.0)
    P4 = numpy.power(Cpmag4, 2.0)
    tot4 = H4 + V4 + P4
    Mh4 = numpy.divide(H4, tot4)

    Ch4 = numpy.multiply(Mh4 , C4)
    xh4 = fix_length(icqt(Ch4, sr=fs, bins_per_octave=12, hop_length=432), len(x))

    print('Iteration 5: suppress transients in xh and xv')
    xh = multiband_transient_shaper(xh4, fs, pool, attack=False)
    xp = xp3+xp2
    xv = multiband_transient_shaper(xh2, fs, pool, attack=False)

    return xh, xp, xv
