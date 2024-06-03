#!/usr/bin/env bash

# Ensure ffmpeg is installed
if ! command -v ffmpeg &> /dev/null
then
    echo "ffmpeg could not be found, please install it first."
    exit
fi

# Directory containing the WAV files
input_dir="public"
output_dir="public/compressed"

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# Loop over all WAV files in the input directory
for wav_file in "$input_dir"/*.wav; do
    # Get the base name of the file without the extension
    base_name=$(basename "$wav_file" .wav)
    
    # Output OGG file path
    output_file="$output_dir/${base_name}.ogg"
    
    # Convert WAV to OGG with a bitrate of 96 kbps
    ffmpeg -i "$wav_file" -b:a 96k "$output_file"
done

echo "Conversion complete. Compressed files are located in the '$output_dir' directory."
