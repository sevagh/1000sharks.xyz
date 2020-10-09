#!/usr/bin/env bash

echo "Fetching training data - skeletal domain"

#youtube-dl -ci -f "bestaudio" -x --audio-format wav -i "https://www.youtube.com/watch?v=2Op6wx6qPnA"

ffmpeg -i 'Cannibal Corpse - A Skeletal Domain (Full Album)-2Op6wx6qPnA.wav' -ac 1 -ar 16000 ./skeletal_domain.wav
