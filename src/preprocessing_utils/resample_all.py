import os, glob
parent_dir = "D:\\datasets\\cer_dataset_16k_flattened\\"
destination_dir =  "D:\\datasets\\cer_dataset_16k_flattened_resampled\\"
classes = os.listdir(parent_dir)
NUM_CLASSES = len(classes)
filenames_origin_all = []
for CLASS in classes[5:]:
    print(CLASS)
    filenames_class=[]
    for layer in range(5): # look maximum 3 layers down
        path = os.path.join(parent_dir, CLASS, "**\\"*layer,"*.wav")
        filenames = glob.glob(os.path.join(parent_dir, CLASS, "**\\"*layer,"*.wav"))
        try:
            filenames_class.extend(filenames)
            # print(filenames_class)
        except: # if no files
            pass
    print("Number of files in {}: {}".format(CLASS, len(filenames_class)))    
    filenames_origin_all.extend(filenames_class)

print("Number of files: ", len(filenames_origin_all))

def get_path_insubdir(filename):
    subdir_path = filename.split(parent_dir)[-1]
    return subdir_path

filenames_destination_all=[]
for filename in filenames_origin_all:
    # print(get_path_insubdir(filename))
    filenames_destination_all.append(destination_dir + get_path_insubdir(filename))

import shutil
def ignore_files(dir, files):
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]
 
# calling the shutil.copytree() method and
# passing the src,dst,and ignore parameter
# try:
#     shutil.copytree(parent_dir, destination_dir,ignore=ignore_files)
# except:
#     pass

def wrap_with_doublequote(input):
    return "\"" + input + "\""

for origin_path,destination_path in zip(filenames_origin_all, filenames_destination_all):
    origin_path = origin_path.replace("/", "\\")
    destination_path = destination_path.replace("/", "\\")
    # print("Resampling {}".format(origin_path))
    command = "ffmpeg -i " + wrap_with_doublequote(origin_path) + " -ac 1 -ar 16000 " + wrap_with_doublequote(destination_path) + " -y -loglevel warning"
    print(command)
    os.system(command)