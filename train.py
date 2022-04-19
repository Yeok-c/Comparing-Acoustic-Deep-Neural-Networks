import sys
import getopt

def myfunc(argv):
    arg_input = ""
    arg_output = ""
    arg_user = ""
    arg_help = "{0} -d <dataset_dir*> -u <num_dense_units> -a <architecture*> -e <epochs(default=40)> -s <seed(default=42)> -p <patch_hop_distance(default=0.25)>".format(argv[0])
    
    try:
        opts, args = getopt.getopt(argv[1:], "hd:u:a:e:s:p:t:r:", ["help", "dataset_dir=", "dense_units=""architecture=", 
                                                          "epochs:", "seed=", "patch_hop_distance=",
                                                          "timenow=", "file_ratio="])
    except:
        print(arg_help)
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)  # print the help message
            sys.exit(2)
        elif opt in ("-d", "--dataset_dir"): # Necessary
            arg_dataset_dir = arg
        elif opt in ("-u", "--dense_units"): # Necessary
            arg_dense_units = int(arg)
        elif opt in ("-a", "--architecture"): # Necessary
            arg_architecture = arg
        elif opt in ("-e", "--epochs"):
            arg_epochs = int(arg)
        elif opt in ("-s", "--seed"):
            arg_seed = int(arg)
        elif opt in ("-p", "--patch_hop_distance"):
            arg_hop_distance = float(arg)
        elif opt in ("-t", "--timenow"):
            arg_timenow = arg
        elif opt in ("-r", "--ratio_of_files"):
            arg_ratio = float(arg)

    return arg_dataset_dir, arg_hop_distance, arg_dense_units, arg_epochs, arg_architecture, arg_seed, arg_timenow, arg_ratio
    
if __name__ == "__main__":
    DATASET_DIR, PATCH_HOP_DISTANCE, DENSE_UNITS, EPOCHS, MODEL_NAME, SEED, DATETIME_NOW, FILE_RATIO = myfunc(sys.argv)
    
    print('Dataset directory:', DATASET_DIR)
    print('Number of epochs:', EPOCHS)
    print('Number of Dense Units in Classifier:', DENSE_UNITS)
    print('Patch Hop Distance for Embedding:', PATCH_HOP_DISTANCE)
    print('Network Architecture:', MODEL_NAME)
    print('Random Seed: ', SEED)
    print('Date Time Now: ', DATETIME_NOW)
    print('Ratio Of Files: ', FILE_RATIO)
    
    
    import os, sys, glob
    import numpy as np
    import tensorflow as tf
    import tensorflow_io as tfio
    import random
    # import datetime

    sys.path.append('src')
    from models.get_models import get_model
    import models.yamnet_tf2.params as params
    params = params.Params(sample_rate=16000, patch_hop_seconds=PATCH_HOP_DISTANCE) # 0.25

    # from dataload_utils.data_load import get_dataset, get_filenames_and_classnames_list
    import dataload_utils.data_load as data_load
    from dataload_utils.data_aug import mix_up

    # SEED = 42
    random.seed(SEED)
    tf.random.set_seed(SEED)

    # parent_dir = "C:\\Users\\User\\Documents\\cer_dataset_16k_flattened_resampled\\"
    dataset_loader = data_load.Dataset_loader(DATASET_DIR, params)
    filenames_all = dataset_loader.__filenames_all__
    classes = dataset_loader.__classes__
    num_classes = dataset_loader.__num_classes__
    print("classes: {}, num_classes: {}".format(classes, num_classes))

    # To do real shuffling
    AUTOTUNE = tf.data.AUTOTUNE
    batch_size=64
    random.shuffle(filenames_all)
    filenames_all=filenames_all[:int(len(filenames_all)*FILE_RATIO)]
    filenames_train = filenames_all[:int(len(filenames_all)*0.7)]
    filenames_eval = filenames_all[int(len(filenames_all)*0.7):int(len(filenames_all)*0.9)]
    filenames_test = filenames_all[int(len(filenames_all)*0.9):]

    # Training set preparation
    dataset_aug = dataset_loader.get_dataset(filenames_train, augment=True)
    train_dataset = dataset_aug.shuffle(batch_size*2).batch(batch_size) # Batch before doing mixup

    # Mixup -
    random.shuffle(filenames_train)
    dataset_no_aug = dataset_loader.get_dataset(filenames_train, augment=False)

    zipped_ds = tf.data.Dataset.zip((
        dataset_aug.shuffle(batch_size*2).batch(batch_size), 
        dataset_no_aug.shuffle(batch_size*2).batch(batch_size)
        ))

    train_dataset = zipped_ds.map(
        map_func = lambda ds_one, ds_two: mix_up(ds_one, ds_two, alpha=0.2), 
        num_parallel_calls=AUTOTUNE
        )

    eval_dataset = dataset_loader.get_dataset(filenames_eval, augment=False).shuffle(batch_size*2).batch(batch_size)
    test_dataset = dataset_loader.get_dataset(filenames_test, augment=False, flat_map=False).shuffle(batch_size*2)#.batch(batch_size)

    train_dataset = train_dataset.cache().prefetch(AUTOTUNE)
    eval_dataset = eval_dataset.cache().prefetch(AUTOTUNE)
    test_dataset = test_dataset.cache().prefetch(AUTOTUNE)

    # length = len(list(dataset_train_eval))
    # print("Total length of dataset: ", length)

    # Paths
    training_path = "./training/{}".format(DATETIME_NOW)

    model_training_path = training_path + "/{}".format(MODEL_NAME)
    ckp_path = model_training_path + "/checkpoints/cp.ckpt"
    log_path = model_training_path + "/logs/fit"    
    hd5_path = model_training_path + "/model.hd5"
    cfm_path = model_training_path + "/confusion_matrix.png"

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckp_path,
                                                    save_weights_only=True,
                                                    verbose=1)
    # Create a tensorboard callback                         
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1)

    # Declare model
    model = get_model(MODEL_NAME, dense_units = DENSE_UNITS)
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.CategoricalCrossentropy(), #from_logits=True
        metrics=['accuracy'],
    )
    if MODEL_NAME=="YAMNET" or MODEL_NAME=="VGGISH":
        # Transfer learn
        # make all layers untrainable by freezing weights (except for last layer)
        for l, layer in enumerate(model.layers[:-7]):
            layer.trainable = False

        # First first time
        model.fit(train_dataset, validation_data = eval_dataset, epochs=EPOCHS-5, 
            verbose=1, callbacks=[cp_callback,tensorboard_callback])
        
        # unfreeze all layers
        for l, layer in enumerate(model.layers[:-7]):
            layer.trainable = True
            
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),
            loss=tf.keras.losses.CategoricalCrossentropy(), #from_logits=True
            metrics=['accuracy'],
        )
        model.fit(train_dataset, validation_data = eval_dataset, initial_epoch = EPOCHS-5, epochs=EPOCHS, 
            verbose=1, callbacks=[cp_callback,tensorboard_callback])
        
    else:
        # Fit model from scratch
        model.fit(train_dataset, validation_data = eval_dataset, epochs=EPOCHS, 
            verbose=1, callbacks=[cp_callback,tensorboard_callback])

    # Evaluate performance of model with test fold (that it wasn't trained on)
    model.load_weights(ckp_path)
    loss, acc = model.evaluate(test_dataset, verbose=2)
    

    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    # Get y_preds = predictions made by model
    y_preds,y_trues = [],[]
    for x_test, y_true in list(test_dataset):
        y_pred = np.argmax(model.predict(x_test), axis=1)
        y_true = np.argmax(y_true, axis=1)
        y_preds.extend(y_pred)
        y_trues.extend(y_true)
    y_trues = np.array(y_trues)

    y_preds = np.array(y_preds)

    accuracy = accuracy_score(y_trues, y_preds)
    print("Testing accuracy: ", accuracy)


    cm, ax = plt.subplots(figsize=(10,10))
    try:
        cm = ConfusionMatrixDisplay.from_predictions(
            y_trues, y_preds, normalize='true', 
            display_labels=classes, xticks_rotation=90,
            ax=ax
        )
    except:
        cm = ConfusionMatrixDisplay.from_predictions(
            y_trues, y_preds, normalize='true', 
            xticks_rotation=90,
            ax=ax
        )
    ax.set_title("{}, Acc: {:02f}".format(model_training_path.split("/")[-1], accuracy))
    cm.figure_.savefig(cfm_path,dpi=300)