from sklearn.model_selection import learning_curve
from tensorflow.keras import layers, Model
import tensorflow as tf


num_classes = 4
def get_model(MODEL, dense_units=128, dropout_rate=0.1):
    if MODEL == 'VGGISH':
        import models.vggish_tf2 as vgk
        base_model = vgk.VGGish(include_top=True)
        base_model.load_weights("./src/models/vggish_tf2/vggish_audioset_weights.h5")
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
        from models.yamnet_tf2 import yamnet
        Yamnet = yamnet.Yamnet(num_classes)
        base_model = Yamnet.model()
        base_model.load_weights("./src/models/yamnet_tf2/yamnet.h5")
        x = layers.GlobalAveragePooling2D()(base_model.layers[-4].output)
    
    if MODEL == 'MobileNetV2':
        base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(
            input_shape=(96,64,1), alpha=1.0, include_top=False, weights=None)
        x = layers.GlobalAveragePooling2D()(base_model.layers[-1].output)
        
    if MODEL == 'MobileNetV3':        
        base_model = tf.keras.applications.MobileNetV3Large(
            input_shape=(96,64,1), alpha=1.0, include_top=True, weights=None, include_preprocessing=False)
        # x = layers.GlobalAveragePooling2D()(MobileNetV1Small.layers[-4].output)
        x = base_model.layers[-3].output
    
    x = layers.Dropout(dropout_rate*2)(x)
    x = layers.Dense(dense_units)(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(64)(x)
    
    logits = layers.Dense(units=num_classes, use_bias=True)(x)
    predictions = layers.Activation(activation='Softmax')(logits) # <- Sigmoid in some original implementations though
    model = Model(base_model.input, predictions) 
    # model.summary()    # model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.CategoricalCrossentropy(), #from_logits=True
        metrics=['accuracy'],
    )
    return model

def model_builder_yamnet(hp):
    import models.yamnet_tf2 as yamnet
    Yamnet = yamnet.Yamnet(num_classes)
    base_model = Yamnet.model()
    base_model.load_weights("./src/models/yamnet_tf2/yamnet.h5")
    x = layers.GlobalAveragePooling2D()(base_model.layers[-4].output)
    
    dense_units = hp.Int('units', min_value=128, max_value=1024, step=128)
    dropout_rate = hp.Float('dropout', min_value=0.1, max_value=0.3, step=0.1)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    x = layers.Dropout(dropout_rate*2)(x)
    x = layers.Dense(dense_units)(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(64)(x)
    
    logits = layers.Dense(units=num_classes, use_bias=True)(x)
    predictions = layers.Activation(activation='Softmax')(logits) # <- Sigmoid in some original implementations though
    model = Model(base_model.input, predictions) 
    # model.summary()    # model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(), #from_logits=True
        metrics=['accuracy'],
    )
    return model


def model_builder_vggish(hp):
    import models.vggish_tf2 as vgk
    base_model = vgk.VGGish(include_top=True)
    base_model.load_weights("./src/models/vggish_tf2/vggish_audioset_weights.h5")
    x = layers.GlobalAveragePooling2D()(base_model.layers[-6].output)
    # Up until pooling layer
        
    dense_units = hp.Int('units', min_value=128, max_value=1024, step=128)
    dropout_rate = hp.Float('dropout', min_value=0.1, max_value=0.3, step=0.1)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    x = layers.Dropout(dropout_rate*2)(x)
    x = layers.Dense(dense_units)(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(64)(x)
    
    logits = layers.Dense(units=num_classes, use_bias=True)(x)
    predictions = layers.Activation(activation='Softmax')(logits) # <- Sigmoid in some original implementations though
    model = Model(base_model.input, predictions) 
    # model.summary()    # model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(), #from_logits=True
        metrics=['accuracy'],
    )
    return model