import os, datetime
# os.system("activate usc39")
DATETIME_NOW = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
SEED = 1
NUM_DENSE_UNITS = 128

# Network_architectures = ['VGGISH', 'EfficientNetV1_B0', 'EfficientNetV2_B0',
#                          'YAMNET', 'MobileNetV2', 'MobileNetV3']
Network_architectures = ['VGGISH']

for MODEL_NAME in Network_architectures:
    # print(MODEL_NAME)
#     command="python train.py -d C:\\Users\\User\\Documents\\cer_dataset_16k_resampled_split\\ -u {} -a {} -e 6 -s {} -p 1 -t {} -r 0.005".format(
    command="python train_hp_tuners.py -d C:\\Users\\User\\Documents\\cer_dataset_16k_resampled_split\\ -u {} -a {} -e 20 -s {} -p 1 -t {} -r 0.4".format(
            NUM_DENSE_UNITS, MODEL_NAME, SEED, DATETIME_NOW)
    print(command)
    os.system(command)