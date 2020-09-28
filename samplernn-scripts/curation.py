import argparse
import sys
import os
import librosa
import soundfile
import essentia.standard as es
import acoustid as ai
import numpy


# returns correlation between lists
# from https://medium.com/@shivama205/audio-signals-comparison-23e431ed2207
def correlation(listx, listy):
    if len(listx) == 0 or len(listy) == 0:
        # Error checking in main program should prevent us from ever being
        # able to get here.
        raise Exception('Empty lists cannot be correlated.')
    if len(listx) > len(listy):
        listx = listx[:len(listy)]
    elif len(listx) < len(listy):
        listy = listy[:len(listx)]
    
    covariance = 0
    for i in range(len(listx)):
        covariance += 32 - bin(listx[i] ^ listy[i]).count("1")
    covariance = covariance / float(len(listx))
    
    return covariance/32


def main():
    parser = argparse.ArgumentParser(
        prog="curation.py",
        description="1000sharks SampleRNN music curation tool"
    )

    parser.add_argument("paths", help="Path to dirs containing generated wav clips", nargs="+")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Sample rate (hz, int), default 16000 based on SampleRNN")
    args = parser.parse_args()

    all_audio = {}

    for p in args.paths:
        for wav_file in os.listdir(p):
            full_path = os.path.join(p, wav_file)
            print('trimming silence and computing chromaprint for {0}'.format(full_path))

            x, _ = soundfile.read(full_path, dtype='float32')

            # trim trailing and leading silence
            x_trimmed, _ = librosa.effects.trim(x, top_db=50, frame_length=256, hop_length=64)

            fingerprint = es.Chromaprinter(sampleRate=args.sample_rate)(x_trimmed)
            int_fingerprint = ai.chromaprint.decode_fingerprint(bytes(fingerprint, encoding='utf-8'))[0]

            all_audio[full_path] = {
                    'raw_audio': x_trimmed,
                    'chromaprint': int_fingerprint,
            }

    correlation_scores = {}

    # naive O(n^2) comparison
    for filename1, data1 in all_audio.items():
        for filename2, data2 in all_audio.items():
            print('comparing chromaprint correlation for {0}, {1}'.format(filename1, filename2))
            if filename1 == filename2:
                # don't compare a file to itself
                continue
            try:
                chromaprint_correlation = correlation(data1['chromaprint'], data2['chromaprint'])
            except:
                continue
            correlation_scores[chromaprint_correlation] = (filename1, filename2)


    # sort by the most highly correlated pairs of audio
    sorted_correlation_scores = dict(sorted(correlation_scores.items(), reverse=True))
    total_audio = None
    for v in sorted_correlation_scores.values():
        print('concatenating audio by similarity')
        # if we've already taken a clip before, ignore
        if v[0] not in all_audio.keys() or v[1] not in all_audio.keys():
            continue

        # keep a running accumulation of similar clips
        if total_audio is None:
            total_audio = all_audio[v[0]]['raw_audio']
        else:
            total_audio = numpy.concatenate((total_audio, all_audio[v[0]]['raw_audio']))
        total_audio = numpy.concatenate((total_audio, all_audio[v[1]]['raw_audio']))

        # delete the data we don't need anymore
        del all_audio[v[0]]
        del all_audio[v[1]]

    print('writing output file')
    soundfile.write("total_curated.wav", total_audio, args.sample_rate)
    return 0


if __name__ == '__main__':
    sys.exit(main())
