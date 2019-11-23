from keras.preprocessing.image import ImageDataGenerator
from os import path

IMAGE_ROWS, IMAGE_COLS = 224, 300

common_options = {
    'rescale': 1./255
}
'''
'rotation_range': 45,
    'shear_range': 0.2,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'vertical_flip': False
'''
train_options = {
    'shear_range': 0.2,
    'zoom_range': 0.2,
    'horizontal_flip': True
}

def load_data(base_folder):

    train_generator = ImageDataGenerator(**{**common_options, **train_options}).flow_from_directory(
        directory=path.join(base_folder, 'train/'),
        target_size=(IMAGE_ROWS, IMAGE_COLS),
        color_mode='rgb',
        batch_size=64,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )

    valid_generator = ImageDataGenerator(**common_options).flow_from_directory(
        directory=path.join(base_folder, 'validation/'),
        target_size=(IMAGE_ROWS, IMAGE_COLS),
        color_mode='rgb',
        batch_size=32,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )

    test_generator = ImageDataGenerator(**common_options).flow_from_directory(
        directory=path.join(base_folder, 'test/'),
        target_size=(IMAGE_ROWS, IMAGE_COLS),
        color_mode='rgb',
        batch_size=1,
        class_mode='categorical',
        shuffle=False,
        seed=42
    )

    return train_generator, valid_generator, test_generator
