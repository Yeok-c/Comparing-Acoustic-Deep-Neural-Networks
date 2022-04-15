import glob, os

DATASET_ORIGIN_PATHS =  ["D:\\datasets\\modified_car_engine_sounds\\original\\"]
DATASET_DESTINATION_PATH = "D:\\datasets\\modified_car_engine_sounds\\resampled_16k\\"

filenames_origin_all, filenames_destination_all = [],[]
for parent_dir in DATASET_ORIGIN_PATHS:
    filenames = glob.glob(os.path.join(parent_dir, "*.wav"))
    filenames_origin_all.extend(filenames)

def get_path_insubdir(filename):
    subdir_path = filename.split(DATASET_ORIGIN_PATHS[0])[-1]
    return subdir_path

for filename in filenames_origin_all:
    # print(get_path_insubdir(filename))
    filenames_destination_all.append(DATASET_DESTINATION_PATH + get_path_insubdir(filename))

# print(filenames_origin_all)

import shutil
def ignore_files(dir, files):
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]
 
# calling the shutil.copytree() method and
# passing the src,dst,and ignore parameter
if os.path.exists(DATASET_DESTINATION_PATH) == False: 
    shutil.copytree(DATASET_ORIGIN_PATHS[0],
                    DATASET_DESTINATION_PATH,
                    ignore=ignore_files)

# Wrap because file name or path may contain spaces
def wrap_with_doublequote(input):
    return "\"" + input + "\""

for origin_path,destination_path in zip(filenames_origin_all, filenames_destination_all):
    origin_path = origin_path.replace("/", "\\")
    destination_path = destination_path.replace("/", "\\")
    print("Resampling {}".format(origin_path))
    os.system("ffmpeg -i " + wrap_with_doublequote(origin_path) + " -ac 1 -ar 16000 " 
                    + wrap_with_doublequote(destination_path) + " -y -loglevel warning")
    # print(origin_path)
