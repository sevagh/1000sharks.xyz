#!/usr/bin/env bash

echo "Fetching training data - youtube-dl wav files for Mestis and Periphery albums"

# youtube playlists for Mestis - Eikasia, Polysemy, Basal Ganglia
mestis_album_1="PLNOrZEIoYAMgLJeZeCUEhABLPz7yqkyfI"
mestis_album_2="PLfoVvOUi1CqV0O-yMdOvTff_vp8hOQnWi"
mestis_album_3="PLRK89uMjq03BMsxBKFGBcDAh2G7ACwJMK"

youtube-dl -ci -f "bestaudio" -x --audio-format wav -i ${mestis_album_1}
youtube-dl -ci -f "bestaudio" -x --audio-format wav -i ${mestis_album_2}
youtube-dl -ci -f "bestaudio" -x --audio-format wav -i ${mestis_album_3}

# youtube playlists for instrumental Periphery albums - Periphery III, I, II, IV, Omega, Juggernaut
periphery_album_1="PLSTnbYVfZR03JGmoJri6Sgvl4f0VAi9st"
periphery_album_2="PL7DVODcLLjFplM5Rw-bNUyrwAECIPRK26"
periphery_album_3="PLuEYu7jyZXdde7ePWV1RUvrpDKB8Gr6ex"
periphery_album_45="PLEFyfJZV-vtKeBedXTv82yxS7gRZkzfWr"
periphery_album_6="PL6FJ2Ri6gSpOWcbdq--P5J0IRcgH-4RVm"

youtube-dl -ci -f "bestaudio" -x --audio-format wav -i ${periphery_album_1}
youtube-dl -ci -f "bestaudio" -x --audio-format wav -i ${periphery_album_2}
youtube-dl -ci -f "bestaudio" -x --audio-format wav -i ${periphery_album_3}
youtube-dl -ci -f "bestaudio" -x --audio-format wav -i ${periphery_album_45}
youtube-dl -ci -f "bestaudio" -x --audio-format wav -i ${periphery_album_6}

mkdir -p periphery-raw
mkdir -p mestis-raw

find . -maxdepth 1 -mindepth 1 -type f -iname '*PERIPHERY*.wav' -exec mv {} periphery-raw/ \;
find . -maxdepth 1 -mindepth 1 -type f -iname '*MESTIS*.wav' -exec mv {} mestis-raw/ \;
find . -maxdepth 1 -mindepth 1 -type f -iname '*Javier*.wav' -exec mv {} mestis-raw/ \;
find . -maxdepth 1 -mindepth 1 -type f -iname '*Suspiro*.wav' -exec mv {} mestis-raw/ \;
find . -maxdepth 1 -mindepth 1 -type f -name '*.wav' -exec rm {} \;

mkdir -p mestis-processed
mkdir -p periphery-processed

echo "Processing each wav file to 16kHz mono"

for f in mestis-raw/*.wav; do
	ffmpeg -i "${f}" -ac 1 -ar 16000 "mestis-processed/$(basename "$f")";
done

for f in periphery-raw/*.wav; do
	ffmpeg -i "${f}" -ac 1 -ar 16000 "periphery-processed/$(basename "$f")";
done

mkdir -p periphery-chunks
mkdir -p mestis-chunks
mkdir -p mixed-chunks

for f in mestis-processed/*.wav; do
	python ../chunk_audio.py --input_file "${f}" --output_dir mestis-chunks --chunk_length 8000 --overlap 1000
	python ../chunk_audio.py --input_file "${f}" --output_dir mixed-chunks --chunk_length 8000 --overlap 1000
done

for f in periphery-processed/*.wav; do
	python ../chunk_audio.py --input_file "${f}" --output_dir periphery-chunks --chunk_length 8000 --overlap 1000
	python ../chunk_audio.py --input_file "${f}" --output_dir mixed-chunks --chunk_length 8000 --overlap 1000
done
