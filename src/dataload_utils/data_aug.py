import tensorflow as tf
import tensorflow_io as tfio

@tf.function
def mix_up(ds_one, ds_two, alpha=0.2):
    def _sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
        gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
        gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
        return gamma_1_sample / (gamma_1_sample + gamma_2_sample)

    # Unpack two datasets
    images_one, labels_one = ds_one
    images_two, labels_two = ds_two
    batch_size = tf.shape(images_one)[0]

    # Sample lambda and reshape it to do the mixup
    l = _sample_beta_distribution(batch_size, alpha, alpha)
    x_l = tf.reshape(l, (batch_size, 1, 1, 1))
    y_l = tf.reshape(l, (batch_size, 1))

    # Perform mixup on both images and labels by combining a pair of images/labels
    # (one from each dataset) into one image/label
    images = images_one * x_l + images_two * (1 - x_l)
    labels = labels_one * y_l + labels_two * (1 - y_l)
    return (images, labels)

def augment_spec(spec, freq_mask_upper_bound=10, time_mask_upper_bound=10):
    # Sample from 2 uniform distributions 
    param1 = tf.random.uniform(shape=[], minval=1, maxval=freq_mask_upper_bound, dtype=tf.int32)
    param2 = tf.random.uniform(shape=[], minval=1, maxval=time_mask_upper_bound, dtype=tf.int32)
    spec = tf.squeeze(spec, axis=-1)
    spec = tfio.audio.freq_mask(spec, param=param1)
    spec = tfio.audio.time_mask(spec, param=param2)
    spec = tf.expand_dims(spec, axis=-1)
    return spec