import os
from keras import applications
import keras.backend as K
from l2norm import l2norm
import numpy as np
from load_source_task import load_VGG16_places
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from load_data import load_data
from pathlib import Path
from glob import glob
from random import shuffle
tf.logging.set_verbosity(tf.logging.ERROR)

def encode_labels(y_train,y_test):
    """Encode the labels in a format suitable for sklearn
    """
    le = LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    return y_train, y_test

def get_paths_and_labels(base_folder, n=None):
    imgs = np.array(glob(f'{base_folder}/**/*.jpg'))
    labels = np.array([Path(img).parent.name for img in imgs])
    if n:
        idx = list(range(len(imgs)))
        shuffle(idx)
        idx = idx[:n]
        imgs = imgs[idx]
        labels = labels[idx]
    return imgs, labels


model, img_width, img_height = load_VGG16_places(fc_layers=False)


# Define input and target tensors, that is where we want
#to enter data, and which activations we wish to extract

#target_tensors = ['block1_conv1/Relu:0','block1_conv2/Relu:0','block2_conv1/Relu:0','block2_conv2/Relu:0','block3_conv1/Relu:0','block3_conv2/Relu:0','block3_conv3/Relu:0','block4_conv1/Relu:0','block4_conv2/Relu:0','block4_conv3/Relu:0','block5_conv1/Relu:0','block5_conv2/Relu:0','block5_conv3/Relu:0','fc1/Relu:0','fc2/Relu:0']
#target_tensors = ['fc1/Relu:0']
'''
target_tensors = ['block1_conv1','block1_conv2',
                    'block2_conv1','block2_conv2',
                    'block3_conv1','block3_conv2','block3_conv3',
                    'block4_conv1','block4_conv2','block4_conv3',
                    'block5_conv1','block5_conv2','block5_conv3',
                    'fc1','fc2']
'''
target_tensors = ['block3_conv2', 'block4_conv3', 'block5_conv1', 'block5_conv2', 'block5_conv3']

#load target task
train_imgs, train_labels = get_paths_and_labels('./data/train')
test_imgs, test_labels = get_paths_and_labels('./data/test')
train_labels, test_labels = encode_labels(train_labels, test_labels)

#Parameters for the extraction procedure
batch_size = 128
input_reshape = (224, 300)
# L2-norm on the train set
l2norm_features = l2norm(model, train_imgs, batch_size, target_tensors, input_reshape, 'VGG16_Places', 'train_l2')

from sklearn import svm
#Train SVM with the obtained features.
clf = svm.LinearSVC()
print('Training SVM...')
clf.fit(X = l2norm_features, y = train_labels)
print('Done training SVM on extracted features of training set')

# L2-norm on the test set
l2norm_features = l2norm(model, test_imgs, batch_size, target_tensors, input_reshape, 'VGG16_Places', 'test_l2')
print('Done extracting features of test set')

#Test SVM with the test set.
predicted_labels = clf.predict(l2norm_features)
print('Done testing SVM on extracted features of test set')

def std_mc_acc(ground_truth, prediction):
    """Standard average multiclass prediction accuracy
    """
    y_ok = prediction == ground_truth
    acc = []
    for unique_y in np.unique(ground_truth):
        acc.append(np.sum(y_ok[ground_truth == unique_y]) * 1.0 / np.sum(ground_truth == unique_y))
    return np.mean(acc)


#Print results
print(std_mc_acc(test_labels, predicted_labels))


