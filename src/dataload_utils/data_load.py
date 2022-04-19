import os, glob
import tensorflow as tf
import models.yamnet_tf2.features as features_lib
import models.yamnet_tf2.yamnet_modified as yamnet
import models.yamnet_tf2.params as yamnet_params
from .data_aug import augment_spec
import librosa

class Dataset_loader:
    def __init__(self, parent_dir, params):
        self.parent_dir = parent_dir
        self.params = params
        self.__filenames_all__, self.__classes__ = self.get_filenames_and_classnames_list()
        self.__num_classes__ = len(self.__classes__)
        
    @tf.function
    def _get_label(self, filename):
        label = tf.strings.split(filename, self.parent_dir+"Class_")[-1]
        label = tf.strings.split(label, "_")[0]
        label = int(label)
        features = self._get_features(filename)
        length = len(features)
        try: 
            label = tf.repeat(label, length)
        except:
            pass
        label = tf.cast(label, dtype='int32')
        label = tf.one_hot(label, self.__num_classes__)
        return label

    def _decode_audio(self, audio_bin):
        audio, _ = tf.audio.decode_wav(contents=audio_bin, desired_channels=1)
        return tf.squeeze(audio, axis=-1)

    def _get_waveform_no_label(self, file_path):
        audio_binary = tf.io.read_file(file_path)
        waveform = self._decode_audio(audio_binary)
        waveform = tf.cast(waveform, dtype=tf.float32)
        return waveform

    def _get_features(self, filename):
        waveform = self._get_waveform_no_label(filename)
        waveform_padded = features_lib.pad_waveform(waveform, self.params)
        log_mel_spectrogram, features = features_lib.waveform_to_log_mel_spectrogram_patches(
            waveform_padded, self.params)
        return tf.expand_dims(features, axis=-1)
        # return log_mel_spectrogram

    def get_embeddings_and_features(self, filename):
        waveform = self._get_waveform_no_label(filename)
        waveform_padded = features_lib.pad_waveform(waveform, self.params)
        log_mel_spectrogram, features = features_lib.waveform_to_log_mel_spectrogram_patches(
            waveform_padded, self.params)
        return log_mel_spectrogram, tf.expand_dims(features, axis=-1)
    
    def get_embeddings_and_features_librosa(self, filename):
        waveform, sr = librosa.load(filename, sr=16000)
        # waveform = self._get_waveform_no_label(filename)
        waveform_padded = features_lib.pad_waveform(waveform, self.params)
        log_mel_spectrogram, features = features_lib.waveform_to_log_mel_spectrogram_patches(
            waveform_padded, self.params)
        return log_mel_spectrogram, tf.expand_dims(features, axis=-1)

    def get_filenames_and_classnames_list(self):
        classes = os.listdir(self.parent_dir)
        NUM_CLASSES = len(classes)
        filenames_all = []
        for CLASS in classes:
            filenames_class=[]
            for layer in range(3): # look maximum 3 layers down
                path = os.path.join(self.parent_dir, CLASS, "**/"*layer,"*.wav")
                filenames = glob.glob(os.path.join(self.parent_dir, CLASS, "**/"*layer,"*.wav"))
                try:
                    filenames_class.extend(filenames)
                    # print(filenames_class)
                except: # if no files
                    pass
            print("Number of files in {}: {}".format(CLASS, len(filenames_class)))    
            filenames_all.extend(filenames_class)
        print("Number of files: ", len(filenames_all))
        return filenames_all, classes

    # Create a dataset of filenames
    def get_dataset(self, filenames, augment=False, flat_map=True):
        dataset = tf.data.Dataset.from_tensor_slices(filenames)

        # Map the filenames to the features
        # Unaugmented should always exist 
        dataset_samples_noaugment = dataset.map(
            map_func = lambda x: tf.py_function(self._get_features, inp=[x], Tout=tf.float32, name=None),
            num_parallel_calls=tf.data.AUTOTUNE
        )  
        dataset_labels = dataset.map(lambda x: self._get_label(x))

        # Flatmap the features to squeeze dim=0 and then zip the samples and labels
        if flat_map==True:
            dataset_samples_noaugment = dataset_samples_noaugment.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
            dataset_labels = dataset_labels.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))

        dataset = dataset.zip((dataset_samples_noaugment, dataset_labels))
        if augment==True:
            dataset = dataset.map(
                map_func = lambda x,y: (tf.py_function(augment_spec, inp=[x, 15, 30], Tout=tf.float32, name=None), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        return dataset
