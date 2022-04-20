import os, datetime
# os.system("activate usc39")
DATETIME_NOW = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
SEED = 1
NUM_DENSE_UNITS = 128
FILE_RATIO = 0.05

Network_architectures = ['VGGISH','YAMNET', 'MobileNetV2', 'MobileNetV3', 'EfficientNetV1_B0', 'EfficientNetV2_B0']

for MODEL_NAME in Network_architectures:
        # print(MODEL_NAME)
        command="""python train.py -d ./datasets/cer_dataset_16k_resampled_split/ \
        -u {} -a {} -e 8 -s {} -p 1 -t {} -r {}""".format(
                        NUM_DENSE_UNITS, MODEL_NAME, SEED, DATETIME_NOW, FILE_RATIO)
        print(command)
        os.system(command)