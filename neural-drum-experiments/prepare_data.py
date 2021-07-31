#!/usr/bin/env python3

import sys
import argparse
import os
import numpy
import subprocess
from essentia.standard import MonoLoader
import soundfile

# median-filtering harmonic percussive source separation
from librosa.effects import percussive

global_data_dir = "./data"


def parse_args():
    parser = argparse.ArgumentParser(
        prog="prepare_data",
        description="Prepare training datasets for tRNNsient from periphery stems",
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=44100,
        help="sample rate (default: 44100 Hz)",
    )

    parser.add_argument(
        "--zen-path",
        type=str,
        default="",
        help="path to zen (default, '', falls back to librosa.percussive which is slower)",
    )

    parser.add_argument(
        "stem_dirs", nargs="+", help="directories containing periphery stems"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = os.path.abspath(global_data_dir)

    seq = 0
    for sd in args.stem_dirs:
        for song in os.scandir(sd):
            for dir_name, _, file_list in os.walk(song):
                instruments = [
                    os.path.join(dir_name, f) for f in file_list if f.endswith(".wav")
                ]
                if instruments:
                    print("Found directory containing wav files: %d" % seq)
                    print(os.path.basename(dir_name).replace(" ", "_"))
                    loaded_wavs = [None] * len(instruments)
                    drum_track_index = -1
                    for i, instrument in enumerate(instruments):
                        if "drum" in instrument.lower():
                            drum_track_index = i
                        # automatically resamples for us
                        loaded_wavs[i] = MonoLoader(
                            filename=instrument, sampleRate=args.sample_rate
                        )()

                    # ensure all stems have the same length
                    assert (
                        len(loaded_wavs[i]) == len(loaded_wavs[0])
                        for i in range(1, len(loaded_wavs))
                    )

                    # first create the full mix
                    full_mix = sum(loaded_wavs)

                    seqstr = "%03d" % seq

                    mix_path = os.path.join(data_dir, "{0}_mix.wav".format(seqstr))
                    soundfile.write(mix_path, full_mix, args.sample_rate)

                    # write the drum track
                    soundfile.write(
                        os.path.join(data_dir, "{0}_drums.wav".format(seqstr)),
                        loaded_wavs[drum_track_index],
                        args.sample_rate,
                    )

                    # then create the percussive separation using harmonic-percussive source separation with median filtering
                    # either pass the path of https://github.com/sevagh/Zen (for better performance GPU median filtering)
                    # or fall back to librosa
                    success = False
                    if args.zen_path:
                        try:
                            hpss_out = subprocess.check_output(
                                "{0} offline -i {1} --hps --out-prefix {2} --only-percussive".format(
                                    args.zen_path,
                                    mix_path,
                                    os.path.join(data_dir, seqstr),
                                ),
                                shell=True,
                            )
                            success = True
                        except subprocess.CalledProcessError:
                            pass

                    if success:
                        print(
                            "used zen for percussive separation of seq {0}".format(seq)
                        )
                    else:
                        print(
                            "applying librosa.effects.percussive for seq {0}".format(
                                seq
                            )
                        )

                        y_perc = percussive(full_mix, margin=5.0)
                        soundfile.write(
                            os.path.join(data_dir, "{0}_perc.wav".format(seqstr)),
                            y_perc,
                            args.sample_rate,
                        )

                    seq += 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
