from sklearn.preprocessing import LabelEncoder
import numpy as np
from pathlib import Path
from glob import glob
from random import shuffle
from load_source_task import load_VGG16_places


def encode_labels(y_train, y_test):
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




from fne import full_network_embedding

model, img_width, img_height = load_VGG16_places(fc_layers=False)

target_tensors = ['block3_conv2', 'block4_conv3', 'block5_conv1', 'block5_conv2', 'block5_conv3']
#target_tensors = ['block1_conv1', 'block1_conv2',
#                  'block2_conv1', 'block2_conv2',
#                  'block3_conv1', 'block3_conv2', 'block3_conv3',
#                  'block4_conv1', 'block4_conv2', 'block4_conv3',
#                  'block5_conv1', 'block5_conv2', 'block5_conv3',
#                  'fc1', 'fc2']
input_reshape = (224, 300)

# load target task
train_imgs, train_labels = get_paths_and_labels('./data/train')
test_imgs, test_labels = get_paths_and_labels('./data/test')
train_labels, test_labels = encode_labels(train_labels, test_labels)

# Parameters for the extraction procedure
batch_size = 128

# Call FNE method on the train set
fne_features, fne_stats_train = full_network_embedding(model, 'VGG16_Places', train_imgs, batch_size,
                                                        target_tensors, input_reshape)

np.save('fne_train.npy', fne_features)
np.save('fne_stats_train.npy', fne_stats_train)

print('Done extracting features of training set. Embedding size:', fne_features.shape)


from sklearn import svm

# Train SVM with the obtained features.
clf = svm.LinearSVC()
clf.fit(X=fne_features, y=train_labels)
print('Done training SVM on extracted features of training set')

# Call FNE method on the test set, using stats from training
fne_features, fne_stats_test = full_network_embedding(model, 'VGG16_Places', test_imgs, batch_size,
                                                        target_tensors,
                                                        input_reshape, stats=fne_stats_train)
np.save('fne_test.npy', fne_features)
np.save('fne_stats_test.npy', fne_stats_train)
print('Done extracting features of test set')

# Test SVM with the test set.
predicted_labels = clf.predict(fne_features)
print('Done testing SVM on extracted features of test set')

def std_mc_acc(ground_truth, prediction):
    """Standard average multiclass prediction accuracy
    """
    y_ok = prediction == ground_truth
    acc = []
    for unique_y in np.unique(ground_truth):
        acc.append(np.sum(y_ok[ground_truth == unique_y]) * 1.0 / np.sum(ground_truth == unique_y))
    return np.mean(acc)

# Print results
print(std_mc_acc(test_labels, predicted_labels))
