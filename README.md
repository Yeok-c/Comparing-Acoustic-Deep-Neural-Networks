# WIP Repo for Comparing Acoustic Deep Neural Networks
## Architectures Compared:
- MobileNetV1 (YAMNET, Transfer learned)
- MobileNetV2
- MobileNetV3
- VGG16 (VGGish, Transfer learned)
- EfficientNetV1B0
- EfficientNetV2B0
- Conv-Mixer (WIP)

## Setup
- Run setup.sh to download dataset and install necessary packages

## Basic Train
- Run train_all.py to iteratively train all networks, do edit the dataset path within the script
- Run train_hp_tuners.py to perform basic hyperparameter search

## Single Model Train
- train.py requires several arguments. Check train_all.py for default values 

      python train.py -d <dataset_path> -u <number of dense units> -a <model architecture> -e <epochs> 
      -s <random_seed> -p <patch_hop_distance> -t <time_now (or other argument for marking training log)> -r <ratio of files> 
      
## Real time inference
- Run record_audio.py on terminal to record wave files
- Run display_spec_rt_2.ipynb to visualize mel-spec and predictions in real time

## Visualize Weights
- For conventional DNN models, run display_weights.ipynb to see convolutional filters
- For Attention based based models, attention visualization is done within their respective notebooks
