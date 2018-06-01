from keras.applications.vgg19 import VGG19
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import keras

import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.python.tools import freeze_graph
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.training import saver as saver_lib

config = {
    # Training params
    "train_data_dir": "/tmp/keras/flower_photos/train",  # training data
    "val_data_dir": "/tmp/keras/flower_photos/val",  # validation data
    "train_batch_size": 16,  # training batch size
    "epochs": 3,  # number of training epochs
    "num_train_samples": 3670,  # number of training examples
    "num_val_samples": 500,  # number of test examples

    # Where to save models (Tensorflow + TensorRT)
    "graphdef_file": "/tmp/keras/flower_photos/keras_vgg19_graphdef.pb",
    "frozen_model_file": "/tmp/keras/flower_photos/keras_vgg19_frozen_model.pb",
    "snapshot_dir": "/tmp/keras/flower_photos/snapshot",
    "engine_save_dir": "/tmp/keras/flower_photos/",

    # Needed for TensorRT
    "image_dim": 224,  # the image size (square images)
    "inference_batch_size": 1,  # inference batch size
    "input_layer": "input_1",  # name of the input tensor in the TF computational graph
    "out_layer": "dense_2/Softmax",  # name of the output tensorf in the TF conputational graph
    "output_size": 5,  # number of classes in output (5)
    "precision": "fp32",  # desired precision (fp32, fp16)

    "test_image_path": "/tmp/keras/flower_photos/test"
}

def prediction():
    return

prediction()