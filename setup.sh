#!/bin/sh
mkdir datasets
cd datasets
curl -L -o cer_dataset_16k_resampled_split.tar.gz "https://www.dropbox.com/s/oplh9e2vq72uqnu/cer_dataset_16k_resampled_split.tar.gz?dl=1"
tar -xf cer_dataset_16k_resampled_split.tar.gz