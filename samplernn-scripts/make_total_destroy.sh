#!/usr/bin/env bash

echo "Fetching training data - Make Total Destroy"

youtube-dl -ci -f "bestaudio" -x --audio-format wav -i "https://www.youtube.com/watch?v=1_VZji3YFo4"

ffmpeg -i 'Periphery - Make Total Destroy (Instrumental)-1_VZji3YFo4.wav' -ac 1 -ar 16000 ./make_total_destroy.wav

python ./chunk_audio.py --input_file ./make_total_destroy.wav --output_dir make-total-destroy/ --chunk_length 8000 --overlap 1000

python train.py --id make-total-destroy --data_dir make-total-destroy/ --num_epochs 100 --batch_size 32 --sample_rate 16000
