from tensorflow.keras import layers, Model
import tensorflow as tf

num_classes = 6
def get_model(MODEL):
    if MODEL == 'VGGISH':
        from .vggish_tf2 import vggish as vgk
        base_model = vgk.VGGish(include_top=True)
        base_model.load_weights("C:/Users/User/Documents/comparing_acoustic_deep_neural_networks/src/models/vggish_tf2/model/vggish_audioset_weights.h5")
        x = layers.GlobalAveragePooling2D()(base_model.layers[-6].output)
        # Up until pooling layer
    
    if MODEL == 'EfficientNetV1_B0':
        base_model = tf.keras.applications.efficientnet.EfficientNetB0(
            include_top=False, weights=None, input_shape=(96, 64, 1))
        x = layers.GlobalAveragePooling2D()(base_model.layers[-1].output)

    if MODEL == 'EfficientNetV2_B0':
        base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(
            include_top=False, weights=None, input_shape=(96, 64, 1))
        x = layers.GlobalAveragePooling2D()(base_model.layers[-1].output)    
    
    if MODEL == 'YAMNET':
        from .yamnet_tf2 import yamnet_class as yamnet
        Yamnet = yamnet.Yamnet(6)
        base_model = Yamnet.model()
        base_model.load_weights("C:/Users/User/Documents/comparing_acoustic_deep_neural_networks/src/models/yamnet_tf2/yamnet.h5")
        x = layers.GlobalAveragePooling2D()(base_model.layers[-4].output)
    
    if MODEL == 'MobileNetV2':
        base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(
            input_shape=(96,64,1), alpha=1.0, include_top=False, weights=None)
        x = layers.GlobalAveragePooling2D()(base_model.layers[-1].output)
        
    if MODEL == 'MobileNetV3':        
        base_model = tf.keras.applications.MobileNetV3Large(
            input_shape=(96,64,1), alpha=1.0, include_top=True, weights=None, include_preprocessing=False)
        # x = layers.GlobalAveragePooling2D()(MobileNetV1Small.layers[-4].output)
        x = layers.Dropout(0.4)(base_model.layers[-3].output)
    
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(1024)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64)(x)
    
    logits = layers.Dense(units=num_classes, use_bias=True)(x)
    predictions = layers.Activation(activation='Softmax')(logits) # <- Sigmoid in some original implementations though
    model = Model(base_model.input, predictions) 
    # model.summary()    # model.summary()
    return model