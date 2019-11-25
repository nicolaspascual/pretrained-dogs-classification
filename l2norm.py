import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model


def load_image_and_resize(input_reshape, abs_path, channels_first=False):
    """
    Some models receive the input as (N x H x W x 3) and another ones as (N x 3 x H x W).
    So, depending on the model you need to change the image shape. That is what channels_first
    parameter controls.
    """
    image = load_img(abs_path, target_size=input_reshape)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    if channels_first:
        image = np.rollaxis(image, 3, 1)
    return image


def l2norm(base_model, image_paths, batch_size, target_tensors, input_reshape, source_model, out_base):
    ''' 
    Extract features from images in a list, using a pre-trained model 
    (input parameter model) with its computational graph loaded. Tensors used 
    to extract features are defined by target_tensors input parameter.

    Args:
        model (tf.GraphDef): Serialized TensorFlow protocol buffer (GraphDef) containing the pre-trained model graph
                             from where to extract the FNE. You can get corresponding tf.GraphDef from default Graph
                             using `tf.Graph.as_graph_def`.
        image_paths (list(str)): List of images to extract features from.
        batch_size (int): Number of images to be concurrently computed on the same batch.
        target_tensors (list(str)): List of tensor names from model to extract features from.
        input_reshape (tuple): A tuple containing the desired shape (height, width) used to resize the image.

    Returns:
       2D ndarray : List of features per image. Of shape <num_imgs,num_feats>
    '''
    if source_model == 'VGG16_ImageNet':
        from keras.applications.vgg16 import preprocess_input
    elif source_model == 'VGG16_Places':
        from places_utils import preprocess_input
    elif source_model == 'testnet':
        def preprocess_input(input_image):
            return input_image
    else:
        import sys
        sys.stdout.write("Source model specified ", source_model,
                         "not recognized. Try: ", "VGG16_ImageNet", ", VGG16_Places")
        sys.exit(1)

    for t_idx, tensor_name in enumerate(target_tensors):
        model = Model(inputs=base_model.input, outputs=base_model.get_layer(tensor_name).output)
        for idx in range(0, len(image_paths), batch_size):
            batch_images_path = image_paths[idx:idx + batch_size]
            img_batch = np.zeros((len(batch_images_path), *input_reshape, 3), dtype=np.float32)
            for i, img_path in enumerate(batch_images_path):
                # image = load_img(img_path, target_size=input_reshape)
                # image = img_to_array(image)
                # image = np.expand_dims(image, axis=0)
                # image = preprocess_input(image)
                # img_batch[i] = image
                abs_path = os.path.abspath(img_path)
                img_batch[i] = load_image_and_resize(input_reshape, abs_path)


            features_batch = model.predict(img_batch, batch_size=batch_size)

            # If its a conv layer, do SPATIAL AVERAGE POOLING
            if len(features_batch.shape) == 4:
                features_batch = np.mean(np.mean(features_batch, axis=2), axis=1)
            if idx == 0:
                features_layer = features_batch.copy()
            else:
                features_layer = np.concatenate((features_layer, features_batch.copy()), axis=0)
            features_layer = image_normalization_L2(features_layer)

        if t_idx == 0:
            features = features_layer.copy()
        else:
            features = np.concatenate((features, features_layer.copy()), axis=1)

    # Store output
    outputs_path = '../outputs'
    if not os.path.exists(outputs_path):
        os.makedirs(outputs_path)
    np.save(os.path.join(outputs_path, f'{out_base}_l2norm.npy'), features)

    # To load output do:
    # l2norm = np.load('l2norm.npy')
    #
    # Return
    return features


def image_normalization_L2(data_matrix):
    """Normalize the data matrix for each image
    """
    l2_norm = np.sqrt(np.sum(np.power(data_matrix, 2), axis=1))[:, np.newaxis]
    return np.nan_to_num(data_matrix / l2_norm)
