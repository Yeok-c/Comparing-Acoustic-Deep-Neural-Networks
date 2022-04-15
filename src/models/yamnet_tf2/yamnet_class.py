from tensorflow.keras import Model, layers
from . import params as yamnet_params

class Yamnet():
  def __init__(self,num_classes):
    self.params = yamnet_params.Params(sample_rate=16000, patch_hop_seconds=0.5)
    self.num_classes = num_classes

  def _batch_norm(self, name, params):
    def _bn_layer(layer_input):
      return layers.BatchNormalization(
        name=name,
        center=params.batchnorm_center,
        scale=params.batchnorm_scale,
        epsilon=params.batchnorm_epsilon)(layer_input)
    return _bn_layer

  def _conv(self, name, kernel, stride, filters, params):
    def _conv_layer(layer_input):
      output = layers.Conv2D(name='{}/conv'.format(name),
                            filters=filters,
                            kernel_size=kernel,
                            strides=stride,
                            padding=params.conv_padding,
                            use_bias=False,
                            activation=None)(layer_input)
      output = self._batch_norm('{}/conv/bn'.format(name), params)(output)
      output = layers.ReLU(name='{}/relu'.format(name))(output)
      return output
    return _conv_layer

  def _separable_conv(self, name, kernel, stride, filters, params):
    def _separable_conv_layer(layer_input):
      output = layers.DepthwiseConv2D(name='{}/depthwise_conv'.format(name),
                                      kernel_size=kernel,
                                      strides=stride,
                                      depth_multiplier=1,
                                      padding=params.conv_padding,
                                      use_bias=False,
                                      activation=None)(layer_input)
      output = self._batch_norm('{}/depthwise_conv/bn'.format(name), params)(output)
      output = layers.ReLU(name='{}/depthwise_conv/relu'.format(name))(output)
      output = layers.Conv2D(name='{}/pointwise_conv'.format(name),
                            filters=filters,
                            kernel_size=(1, 1),
                            strides=1,
                            padding=params.conv_padding,
                            use_bias=False,
                            activation=None)(output)
      output = self._batch_norm('{}/pointwise_conv/bn'.format(name), params)(output)
      output = layers.ReLU(name='{}/pointwise_conv/relu'.format(name))(output)
      return output
    return _separable_conv_layer

  def model(self):
    features = layers.Input(shape=(self.params.patch_frames, self.params.patch_bands, 1))
    # net = layers.Lambda(lambda x: tf.image.resize(x, (28, 28)))(features)
    net = features
    # net = layers.Reshape(
    #     (params.patch_frames, params.patch_bands, 1),
    #     input_shape=(params.patch_frames, params.patch_bands))(features)
    _YAMNET_LAYER_DEFS = [
        # (layer_function, kernel, stride, num_filters)
        (self._conv,          [3, 3], 2,   32),
        (self._separable_conv, [3, 3], 1,   64),
        (self._separable_conv, [3, 3], 2,  128),
        (self._separable_conv, [3, 3], 1,  128),
        (self._separable_conv, [3, 3], 2,  256),
        (self._separable_conv, [3, 3], 1,  256),
        (self._separable_conv, [3, 3], 2,  512),

        (self._separable_conv, [3, 3], 1,  512),
        (self._separable_conv, [3, 3], 1,  512),
        (self._separable_conv, [3, 3], 1,  512),
        (self._separable_conv, [3, 3], 1,  512),
        (self._separable_conv, [3, 3], 1,  512),
        (self._separable_conv, [3, 3], 2, 1024),
        (self._separable_conv, [3, 3], 1, 1024)
        ]
    for (i, (layer_fun, kernel, stride, filters)) in enumerate(_YAMNET_LAYER_DEFS):
      net = layer_fun('layer{}'.format(i + 1), kernel, stride, filters, self.params)(net)
    embeddings = layers.GlobalAveragePooling2D()(net)
    logits = layers.Dense(units=self.params.num_classes, use_bias=True)(embeddings)
    predictions = layers.Activation(activation=self.params.classifier_activation)(logits)
    frames_model = Model(
        name='yamnet_frames', inputs=features,
        outputs=predictions
    )
    return frames_model