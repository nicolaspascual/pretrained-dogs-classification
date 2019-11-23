from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import LabelEncoder

import arg_parser
import tensorflow as tf

from utils_source_model import load_VGG16_imagenet, load_VGG16_places
from utils_target_task import load_target_task, load_target_task_imgs_labels, std_mc_acc
from utils_fine_tuning import data_generators, plot_learning_curves

tf.logging.set_verbosity(tf.logging.ERROR)


def encode_labels(y_train, y_test):
    """Encode the labels in a format suitable for sklearn
    """
    le = LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    return y_train, y_test


if __name__ == '__main__':
    # Check python version
    import sys

    if sys.version_info < (3, 0):
        sys.stdout.write("Sorry, requires Python 3.x\n")
        sys.exit(1)

    # parse parameters
    args = vars(arg_parser.parser.parse_args())
    source_model = args['source_model']
    target_task = args['target_task']

    # import model
    if source_model == 'VGG16_ImageNet':
        model, img_width, img_height = load_VGG16_imagenet(fc_layers=False)
    elif source_model == 'VGG16_Places':
        model, img_width, img_height = load_VGG16_places(fc_layers=False)
    else:
        sys.stdout.write("Source model specified ", source_model,
                         "not recognized. Try: ", "VGG16_ImageNet", ", VGG16_Places")
        sys.exit(1)

    # load target task
    if target_task == 'mit67' or target_task == 'catsdogs' or target_task == 'textures':
        target_classes, train_data_dir, validation_data_dir, nb_train_samples, \
        nb_validation_samples = load_target_task(target_task)
        _, _, _, test_labels = load_target_task_imgs_labels(target_task, n_classes=-1)
    else:
        sys.stdout.write("Target task specified ", target_task,
                         "not recognized. Try: mit67, catsdogs, textures")
        sys.exit(1)

    _, train_labels, _, test_labels = load_target_task_imgs_labels(target_task, n_classes=-1)
    train_labels, test_labels = encode_labels(train_labels, test_labels)
    # Set training parameters
    batch_size = 64
    epochs = 30

    # Freeze the layers which you don't want to train. VGG16 has 18 conv and pooling layers 
    for layer in model.layers[:18]:
        layer.trainable = False
    # 0 input_layer.InputLayer
    # 1 convolutional.Conv2D
    # 2 convolutional.Conv2D
    # 3 pooling.MaxPooling2D
    # 4 convolutional.Conv2D
    # 5 convolutional.Conv2D
    # 6 pooling.MaxPooling2D
    # 7 convolutional.Conv2D
    # 8 convolutional.Conv2D
    # 9 convolutional.Conv2D
    # 10 pooling.MaxPooling2D
    # 11 convolutional.Conv2D
    # 12 convolutional.Conv2D
    # 13 convolutional.Conv2D
    # 14 pooling.MaxPooling2D
    # 15 convolutional.Conv2D
    # 16 convolutional.Conv2D
    # 17 convolutional.Conv2D
    # 18 pooling.MaxPooling2D

    # Adding custom layers on top
    x = model.output
    x = Flatten()(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    predictions = Dense(target_classes, activation="softmax")(x)

    # creating the final model 
    model_final = Model(inputs=model.input, output=predictions)

    # compile the model 
    model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=0.0005, momentum=0.9),
                        metrics=["accuracy"])

    # Initiate the train and test generators with data augumentation 
    train_generator, validation_generator = data_generators(train_data_dir,
                                                            validation_data_dir, img_height, img_width, batch_size)

    # Save the model according to certain conditions  
    # checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

    # Train the model 
    history = model_final.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=[early])  # ,checkpoint])

    # Plot curves
    plot_learning_curves(history)

    # Get predicted labels of validation set
    predicted_labels = model_final.predict_generator(validation_generator, steps=nb_validation_samples / batch_size)
    predicted_labels = predicted_labels.argmax(axis=-1)
    le = LabelEncoder()
    le.fit(test_labels)
    predicted_labels = le.inverse_transform(predicted_labels)

    # Print results
    print(std_mc_acc(test_labels, predicted_labels))

#
