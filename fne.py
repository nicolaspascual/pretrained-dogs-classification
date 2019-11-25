import os
import sys
import numpy as np

from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model


def load_image_and_resize(abs_path, input_reshape):
    image = load_img(abs_path, target_size=input_reshape)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

def full_network_embedding(base_model, source_model, image_paths, batch_size, target_tensors, input_reshape,
                           stats=np.empty((0, 0))):
    ''' 
    Generates the Full-Network embedding[1] of a list of images using a pre-trained
    model (input parameter model) with its computational graph loaded. Tensors used 
    to compose the FNE are defined by target_tensors input parameter. The input_tensor
    input parameter defines where the input is fed to the model.

    By default, the statistics used to standardize are the ones provided by the same 
    dataset we wish to compute the FNE for. Alternatively these can be passed through
    the stats input parameter.

    This function aims to generate the Full-Network embedding in an illustrative way.
    We are aware that it is possible to integrate everything in a tensorflow operation,
    however this is not our current goal.

    [1] Garcia-Gasulla, D., Vilalta, A., Parés, F., Ayguadé, E., Labarta, J., Cortés, U., & Suzumura, T. (2018, November). An out-of-the-box full-network embedding for convolutional neural networks. In 2018 IEEE International Conference on Big Knowledge (ICBK) (pp. 168-175). IEEE.
   
    Args:
        model (tf.GraphDef): Serialized TensorFlow protocol buffer (GraphDef) containing the pre-trained model graph
                             from where to extract the FNE. You can get corresponding tf.GraphDef from default Graph
                             using `tf.Graph.as_graph_def`.
        image_paths (list(str)): List of images to generate the FNE for.
        batch_size (int): Number of images to be concurrently computed on the same batch.
        input_tensor (str): Name of tensor from model where the input is fed to
        target_tensors (list(str)): List of tensor names from model to extract features from.
        input_reshape (tuple): A tuple containing the desired shape (height, width) used to resize the image.
        stats (2D ndarray): Array of feature-wise means and stddevs for standardization.

    Returns:
       2D ndarray : List of features per image. Of shape <num_imgs,num_feats>
       2D ndarry: Mean and stddev per feature. Of shape <2,num_feats>
    '''
    # Print available layers
    # print([tensor.name for tensor in tf.get_default_graph().as_graph_def().node])

    if source_model == 'VGG16_ImageNet':
        from keras.applications.vgg16 import preprocess_input
    elif source_model == 'VGG16_Places':
        from places_utils import preprocess_input
    elif source_model == 'testnet':
        def preprocess_input(input_image):
            return input_image
    else:
        sys.stdout.write("Source model specified ", source_model,
                         "not recognized. Try: ", "VGG16_ImageNet", ", VGG16_Places")
        sys.exit(1)

    for t_idx, tensor_name in enumerate(target_tensors):
        # print(target_tensors)
        # print(tensor_name)

        model = Model(inputs=base_model.input, outputs=base_model.get_layer(tensor_name).output)

        for idx in range(0, len(image_paths), batch_size):
            batch_images_path = image_paths[idx:idx + batch_size]
            img_batch = np.zeros((len(batch_images_path), *input_reshape, 3), dtype=np.float32)
            for i, img_path in enumerate(batch_images_path):
                img_batch[i] = load_image_and_resize(img_path, input_reshape)

            features_batch = model.predict(img_batch, batch_size=batch_size)

            # If its a conv layer, do SPATIAL AVERAGE POOLING
            if len(features_batch.shape) == 4:
                features_batch = np.mean(np.mean(features_batch, axis=2), axis=1)
            if idx == 0:
                features_layer = features_batch.copy()
            else:
                features_layer = np.concatenate((features_layer, features_batch.copy()), axis=0)

        if t_idx == 0:
            features = features_layer.copy()
        else:
            features = np.concatenate((features, features_layer.copy()), axis=1)

    # STANDARDIZATION STEP
    # Compute statistics if needed
    if len(stats) == 0:
        stats = np.zeros((2, features.shape[1]), dtype=np.float32)
        stats[0, :] = np.mean(features, axis=0, dtype=np.float32)
        stats[1, :] = np.std(features, axis=0, dtype=np.float32)
    # Apply statistics, avoiding nans after division by zero
    features = np.divide(features - stats[0], stats[1], out=np.zeros_like(features, dtype=np.float32),
                         where=stats[1] != 0)

    if len(np.argwhere(np.isnan(features))) != 0:
        raise Exception('There are nan values after standardization!')
    # DISCRETIZATION STEP
    th_pos = 0.15
    th_neg = -0.25
    features[features > th_pos] = 1
    features[features < th_neg] = -1
    features[[(features >= th_neg) & (features <= th_pos)][0]] = 0

    # Store output
    outputs_path = '../outputs'
    if not os.path.exists(outputs_path):
        os.makedirs(outputs_path)
    np.save(os.path.join(outputs_path, 'fne.npy'), features)
    np.save(os.path.join(outputs_path, 'stats.npy'), stats)

    # To load output do:
    # fne = np.load('fne.npy')
    # fne_stats = np.load('stats.npy')

    # Return
    return features, stats
