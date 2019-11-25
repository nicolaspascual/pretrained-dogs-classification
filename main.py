import datetime
import sys
from os import path

import tensorflow as tf
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from keras.models import Model, Sequential
from load_data import load_data
from load_source_task import load_VGG16_places
from sklearn.preprocessing import LabelEncoder
from utils import plot_accuracy, plot_loss, save_history, save_model

out_file_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')

if len(sys.argv) > 1:
    out_file_name = sys.argv[1]
if len(sys.argv) > 2:
    gpus_number = int(sys.argv[2])
else:
    gpus_number = 1

tf.logging.set_verbosity(tf.logging.ERROR)



model, img_width, img_height = load_VGG16_places(fc_layers=False)


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
predictions = Dense(120, activation="softmax")(x)

# creating the final model 
model_final = Model(inputs=model.input, output=predictions)

# compile the model 
model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=0.0005, momentum=0.9),
                    metrics=["accuracy"])

# Initiate the train and test generators with data augumentation 
train_generator, valid_generator, test_generator = load_data('./data/')

# Save the model according to certain conditions  
# checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

# Train the model 
history = model_final.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.n // batch_size,
    epochs=epochs,
    validation_data=valid_generator,
    validation_steps=valid_generator.n // batch_size,
    callbacks=[early])  # ,checkpoint])

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
score = model.evaluate_generator(generator=test_generator, steps=STEP_SIZE_TEST)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


plot_accuracy(history, path.join('./result/', out_file_name))
plot_loss(history, path.join('./result/', out_file_name))
save_model(model, path.join('./out/', out_file_name))
save_history(history, path.join('./out/', out_file_name))
