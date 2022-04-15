import glob, os
import shutil

DATASET_ORIGIN_PATH =  "D:\\datasets\\ESC-50-master\\audio\\"
DATASET_DESTINATION_PATH = "D:\\datasets\\ESC-50-master\\audio_resampled_16k\\"

filenames_origin_all, filenames_destination_all = [],[]
filenames = glob.glob(os.path.join(DATASET_ORIGIN_PATH, "*.wav"))
filenames_origin_all.extend(filenames)

def get_path_insubdir(filename):
    subdir_path = filename.split("\\")[-1]
    return subdir_path

for filename in filenames_origin_all:
    # print(get_path_insubdir(filename))
    filenames_destination_all.append(DATASET_DESTINATION_PATH + get_path_insubdir(filename))

def ignore_files(dir, files):
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]
 
# calling the shutil.copytree() method and
# passing the src,dst,and ignore parameter
shutil.copytree(DATASET_ORIGIN_PATH,
                DATASET_DESTINATION_PATH,
                ignore=ignore_files)

for origin_path,destination_path in zip(filenames_origin_all, filenames_destination_all):
    origin_path = origin_path.replace("/", "\\")
    destination_path = destination_path.replace("/", "\\")
    print("Resampling {}".format(origin_path))
    os.system("ffmpeg -i " + origin_path + " -ac 1 -ar 16000 " + destination_path + " -y -loglevel warning")
    # print(origin_path)