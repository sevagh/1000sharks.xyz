from essentia.standard import (
    BFCC,
    MonoLoader,
    Windowing,
    Spectrum,
    FrameGenerator,
    IDCT,
    FFT,
    IFFT,
    OverlapAdd,
)
import numpy
import sys

# the 24 default bark bands
bark_bands = [
    0,
    100,
    200,
    300,
    400,
    510,
    630,
    770,
    920,
    1080,
    1270,
    1480,
    1720,
    2000,
    2320,
    2700,
    3150,
    3700,
    4400,
    5300,
    6400,
    7700,
    9500,
    12000,
    15500,
]

eband5ms = [
    0,
    50,
    2,
    3,
    4,
    255,
]


#static const opus_int16 eband5ms[] = {
#/*0  200 400 600 800  1k 1.2 1.4 1.6  2k 2.4 2.8 3.2  4k 4.8 5.6 6.8  8k 9.6 12k 15.6 20k*/
#  0,  1,  2,  3,  4,  5,  6,  7,  8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100
#};
#
#void interp_band_gain(float *g, const float *bandE) {
#  int i;
#  memset(g, 0, FREQ_SIZE);
#  for (i=0;i<NB_BANDS-1;i++)
#  {
#    int j;
#    int band_size;
#    band_size = (eband5ms[i+1]-eband5ms[i])<<FRAME_SIZE_SHIFT;
#    for (j=0;j<band_size;j++) {
#      float frac = (float)j/band_size;
#      g[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j] = (1-frac)*bandE[i] + frac*bandE[i+1];
#    }
#  }
#}

def _interp_band_gain(bfcc_gain, nfft):
    dft_gain = numpy.zeros(nfft)

    # for number of bands, all except the last pair
    # since we interpolate by summing with the next band
    print(bfcc_gain)
    for (i, band_limits) in enumerate(_pairs(bark_bands)):
        print('pair {0} limits {1}'.format(i, band_limits))
        #print(bfcc_gain[i])
        #print(bfcc_gain[i+1])
        band_size = band_limits[1] - band_limits[0];
        for j in range(band_size):
            frac = float(j)/float(band_size)

            gain_idx = 0# what the fuck is this??

            dft_gain[gain_idx] = (1.0-frac)*bfcc_gain[i-1] + (frac)*bfcc_gain[i]


temperature = 0.3

# still not sure wtf to do here...
def _bark_band_equalizer(fft, sample_rate, nfft, bfcc_cepstral_gains):
    fft_prime = numpy.copy(fft)

    # apply per-band filtering in the fft domain
    # first generate interpolated band gain the way rnnoise does
    for (i, band_limits) in enumerate(_pairs(bark_bands)):
        for j in range(nfft):
            freq = j * sample_rate / nfft
            if band_limits[0] <= freq and freq < band_limits[1]:
                #print('idx {0} freq {1} belongs to band {2}'.format(j, freq, (band_limits[0], band_limits[1])))
                fft_prime[j] = (1-temperature)*fft_prime[j] + temperature*fft_prime[j]*bfcc_cepstral_gains[i]

    return fft_prime


def _pairs(seq):
    i = iter(seq)
    try:
        prev = next(i)
    except StopIteration:
        return
    for item in i:
        yield prev, item
        prev = item


class Bfcc:
    # use the defaults of the BFCC class for the IDCT
    _NUM_BANDS = 24
    _DCT_TYPE = 2
    _LIFTERING = 0
    _NUM_BFCCs = _NUM_BANDS
    _HF_CUTOFF = 15500.0

    def __init__(self, hop_size=512, sample_rate=44100):
        self.hop_size = hop_size
        self.frame_size = 2 * hop_size
        self.fft_size = 4 * hop_size

        self.sample_rate = sample_rate

        self.spectrum_size = int(self.fft_size / 2) + 1

        self.window = Windowing(
            type="hann",
            size=self.frame_size,
            zeroPadding=self.fft_size - self.frame_size,
            normalized=False,
            zeroPhase=False,
        )

        # get the window coefficients for the COLA constraint
        window_default = self.window(numpy.ones(self.frame_size, dtype=numpy.float32))
        cola_factor = numpy.sum(numpy.multiply(window_default, window_default))

        self.spectrum = Spectrum(size=self.fft_size)

        self.bfcc = BFCC(
            inputSize=self.spectrum_size,
            type="magnitude",
            logType="dbamp",
            numberBands=Bfcc._NUM_BANDS,
            dctType=Bfcc._DCT_TYPE,
            numberCoefficients=Bfcc._NUM_BFCCs,
            liftering=Bfcc._LIFTERING,
            highFrequencyBound=Bfcc._HF_CUTOFF,
        )

        # get the window coefficients for the COLA constraint
        window_default = self.window(numpy.ones(self.frame_size, dtype=numpy.float32))
        cola_factor = numpy.sum(numpy.multiply(window_default, window_default))

        self.fft = FFT(size=self.fft_size)
        self.ifft = IFFT(size=self.fft_size)

        # we obviously can't use the same olap object cause it stores inner state!
        self.olap = OverlapAdd(
            frameSize=2 * self.frame_size, hopSize=self.hop_size, gain=2.0 / cola_factor
        )

    def audio_to_bfccs(self, filename):
        ret = []
        audio = MonoLoader(filename=filename, sampleRate=self.sample_rate)()

        expected_frames = int(numpy.floor(audio.shape[0] / (self.frame_size / 2))) - 1
        print(
            "splitting file {0} of len {1} into {2} overlapping frames of size {3}".format(
                filename, audio.shape, expected_frames, self.frame_size
            )
        )

        num_frames = 0
        for frame in FrameGenerator(
            audio,
            frameSize=self.frame_size,
            hopSize=self.hop_size,
            startFromZero=True,
            validFrameThresholdRatio=1,
        ):
            ret.append(self.bfcc(self.spectrum(self.window(frame)))[1])
            num_frames += 1

        if expected_frames != num_frames:
            print("expected {0}, got {1} frames".format(expected_frames, num_frames))

        return ret

    def apply_bfcc_gain_inference(self, filename, nn_model_func):
        audio = MonoLoader(filename=filename, sampleRate=self.sample_rate)()

        modified_audio = numpy.array(0)

        for frame in FrameGenerator(
            audio,
            frameSize=self.frame_size,
            hopSize=self.hop_size,
            startFromZero=True,
            validFrameThresholdRatio=1,
        ):
            waudio = self.window(frame)

            my_bfcc = self.bfcc(self.spectrum(waudio))[1]

            my_bfcc2 = numpy.reshape(my_bfcc, (1, 1, 24))

            # apply our trained tRNNsient model to get the desired bfccs
            # the outputs of the neural network already contain an implicit normalization by 255.0
            desired_bfcc_gain = nn_model_func(my_bfcc2)

            # second half is the gain
            desired_bfcc_gain = numpy.reshape(desired_bfcc_gain[:, :, :24], (24,))

            # the normalizations should cancel out

            # apply fft on windowed audio
            fft = self.fft(waudio)
            # apply per-band gains to transform old_audio into old_audio' from learned nn parameters
            fft_prime = _bark_band_equalizer(
                fft, self.sample_rate, self.fft_size, desired_bfcc_gain
            )
            waudio_prime = self.ifft(fft_prime)

            # need a weighted overlap add for the return result
            modified_audio = numpy.append(modified_audio, self.olap(waudio_prime))

        return modified_audio
