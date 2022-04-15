'''
Yeok C 2022
Resamples each wav file (including those in subfolders) to 16k 
- Parent folder of all other folders should be in DATASET_ORIGIN_PATHS
- Script will search LEVELS_DOWN levels of subfolders for wav files
- Output files will be put in DATASET_DESTINATION_PATH under a single level 


Example input folder structure
├── Folder_01
│   └── Subfolder_01
│       └── wav_1
│       └── wav_2
│       └── wav_3
│   └── Subfolder_02
│       └── wav_4
├── Folder_02
│   └── Subfolder_01
│       └── wav_1
│   └── Subfolder_02
│       └── wav_2
│       └── wav_3
└── Folder_03
    └── Subfolder_01
        └── wav_1

Example output folder structure
├── Folder_01
│   └── wav_1
│   └── wav_2
│   └── wav_3
│   └── wav_4
├── Folder_02
│   └── wav_1
│   └── wav_2
│   └── wav_3
└── Folder_03
    └── wav_1

'''

import glob, os

LEVELS_DOWN = 5
DATASET_ORIGIN_PATHS =  ["/home/yeok/cer_dataset_16k/"]
for i in range(LEVELS_DOWN): DATASET_ORIGIN_PATHS.append(DATASET_ORIGIN_PATHS[i]+"**/")
DATASET_DESTINATION_PATH = "/home/yeok/cer_dataset_16k_flattened/"

filenames_origin_all, filenames_destination_all = [],[]
for parent_dir in DATASET_ORIGIN_PATHS:
    filenames = glob.glob(os.path.join(parent_dir, "*.wav"))
    filenames_origin_all.extend(filenames)

def get_flat_path(filename):
    subdir_path = filename.split("/")[4]
    return subdir_path

def get_filename_only(filename):
    filename_only = filename.split("/")[-1]
    return filename_only

for filename in filenames_origin_all:
    # print(get_flat_path(filename))
    filenames_destination_all.append(DATASET_DESTINATION_PATH + get_flat_path(filename) + "/" + get_filename_only(filename))

# Make dirs for output
os.mkdir(DATASET_DESTINATION_PATH)
for folder in glob.glob(os.path.join(DATASET_ORIGIN_PATHS[0], "*")):
    os.mkdir(DATASET_DESTINATION_PATH+folder.split('/')[-1])

for origin_path,destination_path in zip(filenames_origin_all, filenames_destination_all):
    # origin_path = origin_path.replace("/", "\\")
    # destination_path = destination_path.replace("/", "\\")
    print("Resampling {}".format(origin_path))
    os.system("ffmpeg -i " + origin_path + " -ac 1 -ar 16000 " + destination_path + " -y -loglevel warning")
    # print(origin_path)