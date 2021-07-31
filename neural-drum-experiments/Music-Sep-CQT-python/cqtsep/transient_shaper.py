import numpy
import itertools
from scipy.signal import butter, lfilter

# bark frequency bands between 20 and 20khz, human hearing stuff
_FREQ_BANDS = [
    20,
    119,
    224,
    326,
    438,
    561,
    698,
    850,
    1021,
    1213,
    1433,
    1685,
    1978,
    2322,
    2731,
    3227,
    3841,
    4619,
    5638,
    6938,
    8492,
    10705,
    14105,
    20000,
]


def bandpass(lo, hi, x, fs, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return lfilter(b, a, x)


def _attack_envelope(
    x, fs, fast_attack_ms, slow_attack_ms, release_ms, power_memory_ms, attack=True,
):
    g_fast = numpy.exp(-1.0 / (fs * fast_attack_ms / 1000.0))
    g_slow = numpy.exp(-1.0 / (fs * slow_attack_ms / 1000.0))
    g_release = numpy.exp(-1.0 / (fs * release_ms / 1000.0))
    g_power = numpy.exp(-1.0 / (fs * power_memory_ms / 1000.0))

    fb_fast = 0
    fb_slow = 0
    fb_pow = 0

    N = len(x)

    fast_envelope = numpy.zeros(N)
    slow_envelope = numpy.zeros(N)
    attack_gain_curve = numpy.zeros(N)

    x_power = numpy.zeros(N)
    x_deriv_power = numpy.zeros(N)

    for n in range(N):
        x_power[n] = (1 - g_power) * x[n] * x[n] + g_power * fb_pow
        fb_pow = x_power[n]

    x_deriv_power[0] = x_power[0]

    # simple differentiator filter
    for n in range(1, N):
        x_deriv_power[n] = x_power[n] - x_power[n - 1]

    for n in range(N):
        if fb_fast > x_deriv_power[n]:
            fast_envelope[n] = (1 - g_release) * x_deriv_power[n] + g_release * fb_fast
        else:
            fast_envelope[n] = (1 - g_fast) * x_deriv_power[n] + g_fast * fb_fast
        fb_fast = fast_envelope[n]

        if fb_slow > x_deriv_power[n]:
            slow_envelope[n] = (1 - g_release) * x_deriv_power[n] + g_release * fb_slow
        else:
            slow_envelope[n] = (1 - g_slow) * x_deriv_power[n] + g_slow * fb_slow
        fb_slow = slow_envelope[n]

        attack_gain_curve[n] = fast_envelope[n] - slow_envelope[n]

    attack_gain_curve /= numpy.max(attack_gain_curve)

    if attack:
        return x * attack_gain_curve
    else:
        return x * (1.0 - attack_gain_curve)


def _single_band_transient_shaper(
    band,
    x,
    fs,
    fast_attack_ms,
    slow_attack_ms,
    release_ms,
    power_memory_ms,
    filter_order,
    attack=True,
):
    lo = _FREQ_BANDS[band]
    hi = _FREQ_BANDS[band + 1]

    y = bandpass(lo, hi, x, fs, filter_order)

    # per bark band, apply a differential envelope attack/transient enhancer
    y_shaped = _attack_envelope(
        y, fs, fast_attack_ms, slow_attack_ms, release_ms, power_memory_ms, attack=attack
    )

    return y_shaped


def multiband_transient_shaper(
    x,
    fs,
    pool,
    fast_attack_ms=1,
    slow_attack_ms=15,
    release_ms=20,
    power_memory_ms=1,
    filter_order=3,
    attack=True,
):
    # bark band decomposition
    band_results = list(
        pool.starmap(
            _single_band_transient_shaper,
            zip(
                range(0, len(_FREQ_BANDS) - 1, 1),
                itertools.repeat(x),
                itertools.repeat(fs),
                itertools.repeat(fast_attack_ms),
                itertools.repeat(slow_attack_ms),
                itertools.repeat(release_ms),
                itertools.repeat(power_memory_ms),
                itertools.repeat(filter_order),
                itertools.repeat(attack),
            ),
        )
    )

    y_t = numpy.zeros(len(x))
    for banded_attacks in band_results:
        y_t += banded_attacks

    return y_t
