import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.python.tools import freeze_graph
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.training import saver as saver_lib
import pycuda.driver as cuda
# import pycuda.autoinit--ru
import numpy as np
from random import randint # generate a random test case
from PIL import Image
import time #import system tools
import os
import uff
import tensorrt as trt
from tensorrt.parsers import uffparser
import functools

# Configuration parameters
config = {
    # Training params
    "data_dir": "/tmp/tensorflow/mnist/input_data",  # datasets
    "train_batch_size": 10,  # training batch size
    "epochs": 5000,  # number of training epochs
    "num_train_samples": 50000,  # number of training examples
    "num_val_samples": 5000,  # number of test examples
    "num_test_samples":10000,
    "learning_rate": 1e-4,  # learning rate

    # Where to save logs
    "summary_writer": "/tmp/tensorflow/mnist/log",
    "test_summary_writer": "/tmp/tensorflow/mnist/log/validation",

    # Where to save models (Tensorflow + TensorRT)
    "graphdef_file": "/tmp/tensorflow/mnist/tf_mnist_graphdef.pb",
    "frozen_model_file": "/tmp/tensorflow/mnist/tf_mnist_frozen_model.pb",
    "snapshot_dir": "/tmp/tensorflow/mnist/snapshot",
    "checkpoint_dir":"/tmp/tensorflow/mnist/log",
    "engine_save_dir": "/tmp/tensorflow/mnist",

    # Needed for TensorRT
    "image_dim": 28,  # the image size (square images)
    "inference_batch_size": 1,  # inference batch size
    "input_layer": "Placeholder",  # name of the input tensor in the TF computational graph
    "output_layer": "fc2/Relu",  # name of the output tensorf in the TF conputational graph
    "number_classes": 10,  # number of classes in output (5)
    "precision": "fp32",  # desired precision (fp32, fp16)
}

# It is critical to ensure the version of UFF matches the required UFF version by TensorRT
trt.utils.get_uff_version()
parser = uffparser.create_uff_parser()

def get_uff_required_version(parser):
	return str(parser.get_uff_required_version_major()) + '.' + str(parser.get_uff_required_version_minor()) + '.' + str(parser.get_uff_required_version_patch())

if trt.utils.get_uff_version() != get_uff_required_version(parser):
	raise ImportError("""ERROR: UFF TRT Required version mismatch""")

# Training A Model In TensorFlow
# Defining some hyper parameters and then define some helper functions to make the code a bit less verbose
STARTER_LEARNING_RATE = config['learning_rate']
BATCH_SIZE = config['train_batch_size']
NUM_CLASSES = config['number_classes']
MAX_STEPS = config['epochs']
IMAGE_SIZE = config['image_dim']
INPUT_LAYERS = [config['input_layer']]
OUTPUT_LAYERS = [config['output_layer']]
INFERENCE_BATCH_SIZE = config['inference_batch_size']
INPUT_C = 1
INPUT_H = config['image_dim']
INPUT_W = config['image_dim']


def WeightsVariable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1, name='weights'))

def BiasVariable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape, name='biases'))

def Conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    filter_size = W.get_shape().as_list()
    pad_size = filter_size[0]//2
    pad_mat = np.array([[0,0],
                       [pad_size,pad_size],
                       [pad_size,pad_size],
                       [0,0]])
    x = tf.pad(x, pad_mat)

    x = tf.nn.conv2d(x,
                     W,
                     strides=[1, strides, strides, 1],
     padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def MaxPool2x2(x, k=2):
    # MaxPool2D wrapper
    pad_size = k//2
    pad_mat = np.array([[0,0],
                       [pad_size,pad_size],
                       [pad_size,pad_size],
                       [0,0]])
    return tf.nn.max_pool(x,
                          ksize=[1, k, k, 1],
                          strides=[1, k, k, 1],
                          padding='VALID')

# Define a network and then define our loss metrics, training and test steps, our input nodes, and a data loader.
def network(images):
	# Convolution 1
	with tf.name_scope('conv1'):
		weights = WeightsVariable([5, 5, 1, 32])
		biases = BiasVariable([32])
		conv1 = tf.nn.relu(Conv2d(images, weights, biases))
		pool1 = MaxPool2x2(conv1)

	# Convolution 2
	with tf.name_scope('conv2'):
		weights = WeightsVariable([5, 5, 32, 64])
		biases = BiasVariable([64])
		conv2 = tf.nn.relu(Conv2d(pool1, weights, biases))
		pool2 = MaxPool2x2(conv2)
		pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

	# Fully Connected 1
	with tf.name_scope('fc1'):
		weights = WeightsVariable([7 * 7 * 64, 1024])
		biases = BiasVariable([1024])
		fc1 = tf.nn.relu(tf.matmul(pool2_flat, weights) + biases)

	# Fully Connected 2
	with tf.name_scope('fc2'):
		weights = WeightsVariable([1024, 10])
		biases = BiasVariable([10])
		fc2 = tf.reshape(tf.matmul(fc1, weights) + biases, shape=[-1, 10], name='Relu')

	return fc2


def loss_metrics(logits, labels):
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
		labels=labels,
		logits=logits,
		name='softmax')
	return tf.reduce_mean(cross_entropy, name='softmax_mean')


def training(loss):
	tf.summary.scalar('loss', loss)
	global_step = tf.Variable(0, name='global_step', trainable=False)
	learning_rate = tf.train.exponential_decay(STARTER_LEARNING_RATE,
											   global_step,
											   100000,
											   0.75,
											   staircase=True)
	tf.summary.scalar('learning_rate', learning_rate)
	optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
	train_op = optimizer.minimize(loss, global_step=global_step)
	return train_op


def evaluation(logits, labels):
	correct = tf.nn.in_top_k(logits, labels, 1)
	return tf.reduce_sum(tf.cast(correct, tf.int32))

def timeit(func):
    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        startTime = time.time()
        retargs = func(*args, **kwargs)
        elapsedTime = time.time() - startTime
        print('function [{}] finished in {} ms'.format(
            func.__name__, int(elapsedTime * 1000)))
        return retargs
    return newfunc

@timeit
def do_eval(sess,
			eval_correct,
			images_placeholder,
			labels_placeholder,
			data_set,
			summary):

    true_count = 0
    steps_per_epoch = data_set.num_examples // BATCH_SIZE
    num_examples = steps_per_epoch * BATCH_SIZE

    for step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set,
                                   images_placeholder,
                                   labels_placeholder)
        log, correctness = sess.run([summary, eval_correct],
                                    feed_dict=feed_dict)
        true_count += correctness

    precision = float(true_count) / num_examples
    tf.summary.scalar('precision', tf.constant(precision))
    print('Num examples %d, Num Correct: %d Precision @ 1: %0.04f' % (num_examples, true_count, precision))
    return log

def placeholder_inputs(batch_size):
	images_placeholder = tf.placeholder(tf.float32,
										shape=(None, INPUT_H, INPUT_W, INPUT_C))
	labels_placeholder = tf.placeholder(tf.int32, shape=(None))
	return images_placeholder, labels_placeholder


def fill_feed_dict(data_set, images_pl, labels_pl):
	images_feed, labels_feed = data_set.next_batch(BATCH_SIZE)
	feed_dict = {
		images_pl: np.reshape(images_feed, (-1, INPUT_H, INPUT_W, INPUT_C)),
		labels_pl: labels_feed,
	}
	return feed_dict

# Define our training pipeline in function that will return a frozen model with the training nodes removed.
def run_training(isTrain, data_sets):
    with tf.Graph().as_default():
        images_placeholder, labels_placeholder = placeholder_inputs(BATCH_SIZE)
        logits = network(images_placeholder)
        loss = loss_metrics(logits, labels_placeholder)
        train_op = training(loss)
        eval_correct = evaluation(logits, labels_placeholder)

        # Merges all summaries collected in the default graph.
        summary = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        # Specify the fraction of GPU memory allowed for TensorFlow. TensorRT can use the remaining memory.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)

        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        summary_writer = tf.summary.FileWriter(config['summary_writer'],graph=tf.get_default_graph())
        test_writer = tf.summary.FileWriter(config['test_summary_writer'],graph=tf.get_default_graph())
        sess.run(init)
        if isTrain:
            for step in range(MAX_STEPS):
                start_time = time.time()
                feed_dict = fill_feed_dict(data_sets.train,
                                           images_placeholder,
                                           labels_placeholder)
                _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
                duration = time.time() - start_time
                if step % 100 == 0:
                    print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                    summary_str = sess.run(summary, feed_dict=feed_dict)

                    # Adds a `Summary` protocol buffer to the event file.
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()
                if (step + 1) % 1000 == 0 or (step + 1) == MAX_STEPS:
                    Checkpoint_file = os.path.join(config['summary_writer'], "model.ckpt")
                    saver.save(sess, Checkpoint_file, global_step=step)
                    print('Validation Data Eval:')
                    log = do_eval(sess,
                                  eval_correct,
                                  images_placeholder,
                                  labels_placeholder,
                                  data_sets.validation,
                                  summary)
                    test_writer.add_summary(log, step)

            saver = saver_lib.Saver(write_version=saver_pb2.SaverDef.V2)

            # save model weights in TF checkpoint
            checkpoint_path = saver.save(sess, config['snapshot_dir'], global_step=0,
                                         latest_filename='checkpoint_state')

            graphdef = tf.get_default_graph().as_graph_def()
            frozen_graph = tf.graph_util.convert_variables_to_constants(sess, graphdef, OUTPUT_LAYERS)
            inference_graph = tf.graph_util.remove_training_nodes(frozen_graph)

            # write the graph definition to a file.
            # You can view this file to see your network structure and
            # to determine the names of your network's input/output layers.
            graph_io.write_graph(inference_graph, '.', config['graphdef_file'])

            # specify which layer is the output layer for your graph.
            # In this case, we want to specify the softmax layer after our
            # last dense (fully connected) layer.
            out_names = config['output_layer']

            # freeze your inference graph and save it for later! (Tensorflow)
            freeze_graph.freeze_graph(
                config['graphdef_file'],
                '',
                False,
                checkpoint_path,
                out_names,
                "save/restore_all",
                "save/Const:0",
                config['frozen_model_file'],
                False,
                ""
            )
            return inference_graph
        else:
            # saver = tf.train.import_meta_graph('snapshot-0.meta')
            # saver.restore(sess,tf.train.latest_checkpoint(config['checkpoint_dir']))
            ckpt = tf.train.get_checkpoint_state(config['checkpoint_dir'])

            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Test Data Testing:')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        data_sets.test,
                        summary)



# Load the TensorFlow MNIST data loader and run training
MNIST_DATASETS = input_data.read_data_sets(config['data_dir'])
tf_model = run_training(False, MNIST_DATASETS)
