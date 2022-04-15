import os, datetime
# os.system("activate usc39")
DATETIME_NOW = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
SEED = 1

Network_architectures = ['VGGISH', 'EfficientNetV1_B0', 'EfficientNetV2_B0',
                         'YAMNET', 'MobileNetV2', 'MobileNetV3']

for MODEL_NAME in Network_architectures:
    # print(MODEL_NAME)
    command="python train.py -d C:\\Users\\User\\Documents\\cer_dataset_16k_flattened_resampled\\ -a {} -e 6 -s {} -p 1 -t {} -r 0.005".format(
            MODEL_NAME, SEED, DATETIME_NOW)
    print(command)
    os.system(command)