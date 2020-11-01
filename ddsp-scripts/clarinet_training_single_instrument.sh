#!/usr/bin/env bash

# dataset prep
#python -m ddsp.training.data_preparation.ddsp_prepare_tfrecord \
#	--input_audio_filepatterns=/home/sevagh/ddsp-chowning-clarinet/datasets/*wav \
#	--output_tfrecord_path=/home/sevagh/ddsp-chowning-clarinet/datasets/tfrecords/train.tfrecord \
#	--num_shards=128 \
#	--alsologtostderr \
#	--allow_memory_growth
#
#exit 0

# training
python -m ddsp.training.ddsp_run \
  --mode=train \
  --alsologtostderr \
  --save_dir="/home/sevagh/ddsp-chowning-clarinet/clarinet-autoencoder-training-v2/" \
  --gin_file=ddsp/training/gin/models/solo_instrument.gin \
  --gin_file=ddsp/training/gin/datasets/tfrecord.gin \
  --gin_param="TFRecordProvider.file_pattern='/home/sevagh/ddsp-chowning-clarinet/datasets/tfrecords/train.tfrecord*'" \
  --gin_param="batch_size=16" \
  --gin_param="train_util.train.num_steps=100000" \
  --allow_memory_growth

# evaluate
#python -m ddsp.training.ddsp_run \
#  --mode=eval \
#  --alsologtostderr \
#  --save_dir="/home/sevagh/ddsp-chowning-clarinet/clarinet-autoencoder-training/" \
#  --gin_file=ddsp/training/gin/datasets/tfrecord.gin \
#  --gin_file=ddsp/training/gin/eval/basic_f0_ld.gin \
#  --gin_param="TFRecordProvider.file_pattern='/home/sevagh/ddsp-chowning-clarinet/datasets/tfrecords/train.tfrecord*'" \
#  --allow_memory_growth

# sample
#python -m ddsp.training.ddsp_run \
#  --mode=sample \
#  --alsologtostderr \
#  --save_dir="/home/sevagh/ddsp-chowning-clarinet/clarinet-autoencoder-training/" \
#  --gin_file=ddsp/training/gin/datasets/tfrecord.gin \
#  --gin_file=ddsp/training/gin/eval/basic_f0_ld.gin \
#  --gin_file="/home/sevagh/ddsp-chowning-clarinet/clarinet-autoencoder-training/operative_config-0.gin" \
#  --gin_param="TFRecordProvider.file_pattern='/home/sevagh/ddsp-chowning-clarinet/datasets/tfrecords/train.tfrecord*'" \
#  --allow_memory_growth

# junk params
  #--gin_param="train_util.train.steps_per_save=3000" \
  #--gin_param="trainers.Trainer.checkpoints_to_keep=10" \
  #--gin_param="Additive.n_samples=192000" \
  #--gin_param="Additive.sample_rate=48000" \
  #--gin_param="FilteredNoise.n_samples=192000"
