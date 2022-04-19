# WIP Repo for Comparing Acoustic Deep Neural Networks
# Architectures Compared:
- MobileNetV1 (YAMNET)
- MobileNetV2
- MobileNetV3
- VGG16 (VGGish)
- EfficientNetV1B0
- EfficientNetV2B0
- Conv-Mixer (WIP)

# Setup
- Run setup.sh to download dataset and install necessary packages

# Basic Train
- Run train_all.py to iteratively train all networks 
- Run train_hp_tuners.py to perform basic hyperparameter search

# Real time inference
- Run record_audio.py on terminal to record wave files
- Run display_spec_rt_2.ipynb to visualize mel-spec and predictions in real time
