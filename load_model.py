from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.utils.training_utils import multi_gpu_model
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras import regularizers
import tensorflow as tf

def load_model(input_shape, gpus_number):
    """
        Reduced VGG16
    """
    common_options_cnn = {
        'activation': 'relu', 'input_shape': input_shape, 'padding': 'same'
    }
    model = Sequential()
    model.add(Conv2D(64, (3, 3), **common_options_cnn))
    model.add(Conv2D(64, (3, 3), **common_options_cnn))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=3))


    model.add(Conv2D(85, (3, 3), **common_options_cnn))
    model.add(Conv2D(85, (3, 3), **common_options_cnn))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=3))

    model.add(Conv2D(106, (3, 3), **common_options_cnn))
    model.add(Conv2D(106, (3, 3), **common_options_cnn))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=3))

    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dropout(0.8))
    model.add(Dense(1000, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(500, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(120, activation=(tf.nn.softmax)))
    if gpus_number > 1:
        model = multi_gpu_model(model, gpus=gpus_number)
    optimizer = RMSprop(lr=1e-3)# 1e-3 this or higher discarded: 1e-1, 1e-2
    model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
    return model

def load_model_5(input_shape):
    """
        Added dropout
        Baseline + dropout only on FC layers + increasing lr + 2019-10-04_08-37
    """

    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))


    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.4))


    model.add(Dense(120, activation=(tf.nn.softmax)))
    model.add(Dropout(0.2))


    optimizer = RMSprop(lr=5e-3)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def load_model_6(input_shape):
    """
        Baseline + dropout + reducing lr 2019-10-03_22-19
    """

    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))


    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))


    model.add(Dense(120, activation=(tf.nn.softmax)))

    optimizer = RMSprop(lr=5e-5)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def load_model_5(input_shape):
    '''
        Model used to get to overfit, but after using data augmentation it stopped working
    '''
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(120, activation=(tf.nn.softmax)))

    optimizer = RMSprop(lr=5e-4)
    model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
    return model


def load_model_4(input_shape):
    """
        Reduce overfit but no so much, reduced structure and added custom lr
        result/train/2019-10-03_09-17
    """
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dense(120, activation=(tf.nn.softmax)))

    optimizer = RMSprop(lr=5e-4)
    model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
    return model

def load_model_3(input_shape):
    """
        VGG19 copy used for the first experiments until 02/10/2019 :: 14:00
    """
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(120, activation=(tf.nn.softmax)))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model




def load_model_2(input_shape):
    """
            Architecture + l2 reg
    """
    model = Sequential()
    lambda_reg = 0.0005
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=regularizers.l2(lambda_reg)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(lambda_reg)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(200, activation='relu', kernel_regularizer=regularizers.l2(lambda_reg)))
    model.add(Dense(120, activation=(tf.nn.softmax), kernel_regularizer=regularizers.l2(lambda_reg)))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def load_model_1(input_shape):
    """
            Architecture + dropout
    """
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(120, activation=(tf.nn.softmax)))
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
    return model
