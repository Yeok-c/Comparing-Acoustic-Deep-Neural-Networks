#!/bin/sh
mkdir datasets
curl -L -o cer_dataset_16k_resampled_split.tar.gz "https://www.dropbox.com/s/oplh9e2vq72uqnu/cer_dataset_16k_resampled_split.tar.gz?dl=1"
tar -xf cer_dataset_16k_resampled_split.tar.gz -C ./datasets/
rm cer_dataset_16k_resampled_split.tar.gz

curl -L -o ./src/models/vggish_tf2/vggish_audioset_weights.h5 "https://www.dropbox.com/s/nwh04df77tfkgfx/vggish_audioset_weights.h5?dl=1"
apt-get install libsndfile1-dev -y
pip install -r requirements.txt
python train_all_models_example.py