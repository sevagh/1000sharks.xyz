#!/usr/bin/env python3

import matplotlib.pyplot as plt
import dtcwt
import pywt
from scipy.io import wavfile
import sys
import numpy


if __name__ == '__main__':
    in_mix = None
    in_harm = None
    in_perc = None
    in_rate = None
    out_file = None
    try:
        in_rate, in_mix = wavfile.read(sys.argv[1])
        _, in_perc = wavfile.read(sys.argv[2])
        _, in_harm = wavfile.read(sys.argv[3])
        out_file = sys.argv[4]
    except IndexError:
        print('usage: dtcwthpss.py mix perc harm', file=sys.stderr)
        sys.exit(1)

    samples_cap = int(numpy.floor(len(in_mix)/2)*2)
    in_mix = in_mix[:samples_cap]
    in_harm = in_harm[:samples_cap]
    in_perc = in_perc[:samples_cap]

    mA, mD = pywt.dwt(in_mix, 'db2')
    pA, pD = pywt.dwt(in_perc, 'db2')
    hA, hD = pywt.dwt(in_harm, 'db2')

    # Show level 1 highpass coefficient magnitudes
    plt.figure()
    plt.plot(mA)
    plt.title('dwt A, mix')

    plt.figure()
    plt.plot(mD)
    plt.title('dwt D, mix')

    plt.figure()
    plt.plot(pA)
    plt.title('dwt A, perc')

    plt.figure()
    plt.plot(pD)
    plt.title('dwt D, perc')

    plt.figure()
    plt.plot(hA)
    plt.title('dwt A, harm')

    plt.figure()
    plt.plot(hD)
    plt.title('dwt D, harm')
    plt.show()

    #frankenstein = dtcwt.Pyramid(harm_t.lowpass, mix_t.highpasses)

    #inv_recon = transform.inverse(frankenstein)
    #inv_recon /= max(numpy.abs(inv_recon))

    #wavfile.write(out_file, in_rate, inv_recon)
