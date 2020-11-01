// stkclarinet.cpp
//
// based on MUMT-307 homework
// FM Clarinet algorithm
//
// Sevag Hanssian
//
// Usage:
// 	$ ./stkclarinet [optional f0] [optional duration]
//
// Examples:
// 	$ ./stkclarinet
// 		default, 0.5s duration, f0 = 300Hz
// 	$ ./stkclarinet 300 1.0
// 		f0 = 300Hz (same as default), duration = 1.0s
// 	$ ./stkclarinet 1337
// 		f0 = 1337Hz, duration = 0.5s (default)

#include <stk/ADSR.h>
#include <stk/SineWave.h>
#include <stk/RtWvOut.h>

// small value close to 0 for ADSR
static constexpr float EPS = 0.000000001;

// unchanging params
static constexpr int IMIN = 2;
static constexpr int IMAX = 4;

// from doge book, chowning clarinet FM states that f0 = fc/3
static constexpr float FC_FM_RATIO = 1.5;
static constexpr float FC_F0_RATIO = 3;

int main(int argc, char **argv)
{
    float f_0 = 900.0/FC_F0_RATIO; // 900 = default fc
    float duration_secs = 0.5;

    if (argc > 1) {
        f_0 = std::stof(argv[1]);
        if (argc == 3) {
            duration_secs = std::stof(argv[2]);
        }
    }

    stk::Stk::setSampleRate(44100.0);

    stk::ADSR adsr1; // F1 shape
    stk::ADSR adsr2; // F2 shape

    adsr1.setAllTimes(0.2, EPS, 1.0, 0.1);
    adsr2.setAllTimes(EPS, EPS, 1.0, 0.2);

    // Create looping wavetables with sine waveforms.
    stk::SineWave carrier;
    stk::SineWave modulator;

    // Create a one-channel realtime output object.
    stk::RtWvOut output;
    output.start();

    // Set our FM parameters.
    float f_c = f_0*FC_F0_RATIO;
    float f_m = f_c/FC_FM_RATIO;

    modulator.setFrequency(f_m);
    long releaseCount = (long) (stk::Stk::sampleRate() * duration_secs);

    // Start the runtime loop.
    long counter = 0;
    adsr1.keyOn();
    adsr2.keyOn();

    while (true) {
        auto F2_envelope_tick = adsr2.lastOut() * f_m*(IMAX-IMIN);
        auto fm_input = F2_envelope_tick + f_m*IMIN;
        carrier.setFrequency(f_c + (fm_input * modulator.tick()));
        auto F1_envelope_tick = adsr1.lastOut();

        // finally, end result, sum of left and right blocks
        output.tick(F1_envelope_tick * carrier.tick());

        // Update the envelope and check for release time.
        adsr1.tick();
        adsr2.tick();

        // we're done here
        if (counter++ == releaseCount) {
            adsr1.keyOff();
            adsr2.keyOff();
            break;
        }
    }

    return 0;
}
