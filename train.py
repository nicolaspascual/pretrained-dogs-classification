import datetime
import sys
from os import path
from keras.callbacks import EarlyStopping
from dl_tensor_board import TrainValTensorBoard
out_file_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')

if len(sys.argv) > 1:
    out_file_name = sys.argv[1]
if len(sys.argv) > 2:
    gpus_number = int(sys.argv[2])
else:
    gpus_number = 1

from load_data import load_data
train_generator, valid_generator, test_generator = load_data('./data/')
input_shape = train_generator.image_shape

from load_model import load_model
model = load_model(input_shape, gpus_number)


#Start training
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.005, patience=10, verbose=0, mode='auto')
tensorboard_callback = TrainValTensorBoard(log_dir=path.join('./out/', out_file_name, 'tensorboard.log'))
history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=100,
                    callbacks=[early_stopping, tensorboard_callback]

)
#Evaluate the model with test set
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
score = model.evaluate_generator(generator=test_generator, steps=STEP_SIZE_TEST)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

from utils import plot_accuracy, plot_loss, save_model, save_history

plot_accuracy(history, path.join('./result/', out_file_name))
plot_loss(history, path.join('./result/', out_file_name))
save_model(model, path.join('./out/', out_file_name))
save_history(history, path.join('./out/', out_file_name))
