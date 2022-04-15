#!/bin/sh
# mkdir datasets
cd datasets
curl -L -o cer_dataset_16k_flattened_resampled.tar.gz "https://www.dropbox.com/s/rav3hndwb3eu0hm/cer_dataset_16k_flattened_resampled.tar.gz?dl=01"
tar -xf cer_dataset_16k_flattened_resampled.tar.gz
# File is typically in /notebooks/datasets/cer_dataset_16k_flattened_resampled

# git clone https://github.com/Yeok-c/YAMNET_TF2