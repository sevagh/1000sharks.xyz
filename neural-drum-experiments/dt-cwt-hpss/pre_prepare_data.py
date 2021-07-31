#!/usr/bin/env python3

import sys
import argparse
import os
import numpy
import subprocess
from essentia.standard import MonoLoader
import soundfile
import multiprocessing

global_data_dir = "./data"


def parse_args():
    parser = argparse.ArgumentParser(
        prog="pre_prepare_data",
        description="Prepare training datasets from periphery stems",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=44100,
        help="sample rate (default: 44100 Hz)",
    )
    parser.add_argument(
        "--n-pool",
        type=int,
        default=multiprocessing.cpu_count() - 1,
        help="size of multiprocessing pool",
    )
    parser.add_argument(
        "--testcases",
        type=int,
        default=-1,
        help="how many testcases to discover (default: -1, i.e. all)"
    )
    parser.add_argument(
        "stem_dirs", nargs="+", help="directories containing periphery stems"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = os.path.abspath(global_data_dir)
    pool = multiprocessing.Pool(args.n_pool)

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
                        os.path.join(data_dir, "{0}_percussive.wav".format(seqstr)),
                        loaded_wavs[drum_track_index],
                        args.sample_rate,
                    )

                    # first create the full mix
                    harmonic_mix = sum(loaded_wavs[:drum_track_index]) + sum(loaded_wavs[drum_track_index+1:])

                    # write the harmonic track (everything without drums)
                    soundfile.write(
                        os.path.join(data_dir, "{0}_harmonic.wav".format(seqstr)),
                        harmonic_mix,
                        args.sample_rate,
                    )

                    seq += 1

                    # end early for 1 single testcase
                    if args.testcases > 0:
                        if seq >= args.testcases:
                            return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
